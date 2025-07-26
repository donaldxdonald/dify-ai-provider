import {
  APICallError,
  type JSONValue,
  type LanguageModelV1,
  type LanguageModelV1CallOptions,
  type LanguageModelV1FinishReason,
  type LanguageModelV1ObjectGenerationMode,
  type LanguageModelV1StreamPart,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  generateId,
  postJsonToApi,
  type ParseResult,
} from "@ai-sdk/provider-utils";
import {
  difyStreamEventSchema,
  errorResponseSchema,
} from "../dify-chat-schema";
import type { DifyStreamEvent } from "../dify-chat-schema";
import { DifyCompletionModelId, DifyCompletionSettings } from "./settings";
import { workflowCompletionResponseSchema } from "./schema";

interface ModelConfig {
  provider: string;
  baseURL: string;
  headers: () => Record<string, string>;
  fetch?: FetchFunction;
}

const difyFailedResponseHandler = createJsonErrorResponseHandler({
  errorSchema: errorResponseSchema,
  errorToMessage: (data) => {
    console.log("Dify API error:", data);
    return `Dify API error: ${data.message}`;
  },
});

// For TypeScript compatibility
interface ExtendedLanguageModelV1CallOptions
  extends LanguageModelV1CallOptions {
  messages?: Array<{
    role: string;
    content: string | Array<string | { type: string; [key: string]: any }>;
  }>;
}

export class DifyCompletionLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = "v1";
  readonly modelId: string;
  readonly defaultObjectGenerationMode: LanguageModelV1ObjectGenerationMode =
    undefined;

  private readonly generateId: () => string;
  private readonly completionEndpoint: string;
  private readonly config: ModelConfig;

  constructor(
    modelId: DifyCompletionModelId,
    private settings: DifyCompletionSettings,
    config: ModelConfig
  ) {
    this.modelId = modelId;
    this.config = config;
    this.generateId = generateId;
    this.completionEndpoint = `${this.config.baseURL}/workflows/run`;

    // Make sure we set a default response mode
    if (!this.settings.responseMode) {
      this.settings.responseMode = "streaming";
    }
  }

  get provider(): string {
    return this.config.provider;
  }

  async doGenerate(
    options: ExtendedLanguageModelV1CallOptions
  ): Promise<Awaited<ReturnType<LanguageModelV1["doGenerate"]>>> {
    const { abortSignal } = options;
    const requestBody = this.getRequestBody(options);

    const { responseHeaders, value: data } = await postJsonToApi({
      url: this.completionEndpoint,
      headers: combineHeaders(this.config.headers(), options.headers),
      body: requestBody,
      abortSignal,
      failedResponseHandler: difyFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        workflowCompletionResponseSchema
      ),
      fetch: this.config.fetch,
    });

    return {
      text: data.data?.outputs?.result || "",
      toolCalls: [], // Dify workflows don't support tool calls
      finishReason: "stop" as LanguageModelV1FinishReason,
      usage: {
        promptTokens: 0,
        completionTokens: 0,
      },
      rawCall: this.createRawCall(options),
      providerMetadata: {
        difyWorkflowData: {
          workflowRunId: data.workflow_run_id as JSONValue,
          taskId: data.task_id as JSONValue,
        },
      },
      rawResponse: {
        headers: responseHeaders,
        body: data,
      },
      request: { body: JSON.stringify(requestBody) },
      response: {
        id: data.workflow_run_id || this.generateId(),
        timestamp: new Date(),
      },
    };
  }

  async doStream(
    options: ExtendedLanguageModelV1CallOptions
  ): Promise<Awaited<ReturnType<LanguageModelV1["doStream"]>>> {
    const { abortSignal } = options;
    const requestBody = this.getRequestBody(options);
    const body = { ...requestBody, response_mode: "streaming" };

    const { responseHeaders, value: responseStream } = await postJsonToApi({
      url: this.completionEndpoint,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: difyFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
        difyStreamEventSchema
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    let workflowRunId: string | undefined;
    let taskId: string | undefined;

    return {
      stream: responseStream.pipeThrough(
        new TransformStream<
          ParseResult<DifyStreamEvent>,
          LanguageModelV1StreamPart
        >({
          transform(chunk, controller) {
            if (!chunk.success) {
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const data = chunk.value;

            // Store workflow run and task IDs for metadata
            if (data.workflow_run_id)
              workflowRunId = data.workflow_run_id as any;
            if (data.task_id) taskId = data.task_id;

            // Handle known event types
            switch (data.event) {
              case "workflow_finished": {
                // Add block scope to prevent variable leakage
                let totalTokens = 0;

                // Type guard for data.data
                if (
                  "data" in data &&
                  data.data &&
                  typeof data.data === "object" &&
                  "total_tokens" in data.data &&
                  typeof data.data.total_tokens === "number"
                ) {
                  totalTokens = data.data.total_tokens;
                }

                controller.enqueue({
                  type: "finish",
                  finishReason: "stop",
                  providerMetadata: {
                    difyWorkflowData: {
                      workflowRunId: workflowRunId as JSONValue,
                      taskId: taskId as JSONValue,
                    },
                  },
                  usage: {
                    promptTokens: 0,
                    completionTokens: totalTokens,
                  },
                });
                break;
              }

              case "text_chunk": {
                // Type guard for text property
                if (
                  "data" in data &&
                  data.data &&
                  typeof data.data === "object" &&
                  "text" in data.data &&
                  typeof data.data.text === "string"
                ) {
                  controller.enqueue({
                    type: "text-delta",
                    textDelta: data.data.text,
                  });
                }
                break;
              }

              case "workflow_started": {
                // Type guard for workflow_run_id property
                if (
                  "workflow_run_id" in data &&
                  typeof data.workflow_run_id === "string"
                ) {
                  controller.enqueue({
                    type: "response-metadata",
                    id: data.workflow_run_id,
                  });
                }
                break;
              }

              // Ignore other event types
            }
          },
        })
      ),
      rawCall: this.createRawCall(options),
      rawResponse: { headers: responseHeaders },
      request: { body: JSON.stringify(body) },
    };
  }

  /**
   * Get the request body for the Dify Workflow API
   */
  private getRequestBody(options: ExtendedLanguageModelV1CallOptions) {
    // In AI SDK v4, messages are in options.prompt instead of options.messages
    const messages = options.messages || options.prompt;

    if (!messages || !messages.length) {
      throw new APICallError({
        message: "No messages provided",
        url: this.completionEndpoint,
        requestBodyValues: options,
      });
    }

    const latestMessage = messages[messages.length - 1];

    if (latestMessage.role !== "user") {
      throw new APICallError({
        message: "The last message must be a user message",
        url: this.completionEndpoint,
        requestBodyValues: { latestMessageRole: latestMessage.role },
      });
    }

    // Handle file/image attachments
    const hasAttachments =
      Array.isArray(latestMessage.content) &&
      latestMessage.content.some((part: any) => {
        return typeof part !== "string" && part.type === "image";
      });

    if (hasAttachments) {
      throw new APICallError({
        message: "Dify provider does not currently support image attachments",
        url: this.completionEndpoint,
        requestBodyValues: { hasAttachments: true },
      });
    }

    // Extract the query from the latest user message
    let query = "";
    if (typeof latestMessage.content === "string") {
      query = latestMessage.content;
    } else if (Array.isArray(latestMessage.content)) {
      // Handle AI SDK v4 format with text objects in content array
      query = latestMessage.content
        .map((part: any) => {
          if (typeof part === "string") {
            return part;
          } else if (part.type === "text") {
            return part.text;
          }
          return "";
        })
        .filter(Boolean)
        .join(" ");
    }

    const userId = options.headers?.["user-id"] ?? "you_should_pass_user-id";
    const { "user-id": _, ...cleanHeaders } = options.headers || {};
    options.headers = cleanHeaders;

    return {
      inputs: {
        query,
        ...this.settings.inputs,
      },
      response_mode: this.settings.responseMode,
      user: userId,
    };
  }

  /**
   * Create the rawCall object for response
   */
  private createRawCall(options: ExtendedLanguageModelV1CallOptions) {
    return {
      rawPrompt: options.messages || options.prompt,
      rawSettings: { ...this.settings },
    };
  }

  supportsUrl?(_url: URL): boolean {
    return false;
  }
}
