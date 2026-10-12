#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for the loaded model + context
typedef struct EvalCoreHandle EvalCoreHandle;

// Return codes
typedef enum {
    EVAL_CORE_OK = 0,
    EVAL_CORE_ERR_MODEL_LOAD = -1,
    EVAL_CORE_ERR_CONTEXT_CREATE = -2,
    EVAL_CORE_ERR_TOKENIZE = -3,
    EVAL_CORE_ERR_BATCH_DECODE = -4,
    EVAL_CORE_ERR_INVALID_ARGS = -5,
    EVAL_CORE_ERR_OOM = -6,
} EvalCoreStatus;

// Load model from GGUF file and create context.
// model_path: path to .gguf file
// n_ctx: context window size
// n_gpu_layers: number of layers to offload to GPU (0 = CPU only)
// n_threads: number of CPU threads
// Returns handle on success, NULL on failure (check eval_core_last_error())
EvalCoreHandle* eval_core_load_model(const char* model_path,
                                      int n_ctx,
                                      int n_gpu_layers,
                                      int n_threads);

// Get last error message (thread-local static buffer)
const char* eval_core_last_error(void);

// Run batched evaluation.
// handle: from eval_core_load_model
// prompts: array of n prompt strings (instruction part only)
// articles: array of n article strings
// n: number of (prompt, article) pairs
// out_labels: pre-allocated int32_t array of size n, filled with 0..3
//             mapping: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
// Returns EVAL_CORE_OK on success, error code on failure.
EvalCoreStatus eval_core_score_batch(EvalCoreHandle* handle,
                                      const char* const* prompts,
                                      const char* const* articles,
                                      int n,
                                      int32_t* out_labels);

// Free handle and all associated resources.
void eval_core_free(EvalCoreHandle* handle);

#ifdef __cplusplus
}
#endif