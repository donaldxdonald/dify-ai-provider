import { z } from "zod";

export const workflowCompletionResponseSchema = z.object({
  task_id: z.string(),
  workflow_run_id: z.string().optional(),
  data: z.object({
    id: z.string(),
    workflow_id: z.string(),
    outputs: z.record(z.string(), z.any()).optional(),
  }),
});
