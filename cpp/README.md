# eval_core — Batched LLM Inference for AG News Classification

C++ shared library (`libeval_core.dylib` / `.so`) that wraps llama.cpp for high-throughput batched evaluation of prompts against the AG News dataset.

## Why This Exists

`llama_cpp_python` evaluates one prompt at a time — each call clears KV cache, re-tokenizes, and runs a full forward pass. For RL prompt optimization (hundreds of evaluations), this is the bottleneck.

`eval_core` batches N prompts into a **single forward pass**:
- One KV cache clear
- One `llama_decode` call
- N predictions returned

**Typical speedup: 5–10×** on Apple Metal / CPU.

## API (eval_core.hpp)

```c
// Opaque handle
typedef struct EvalCoreHandle EvalCoreHandle;

// Load model, create context
EvalCoreHandle* eval_core_load_model(const char* model_path,
                                      int n_ctx,           // per-sequence context (e.g., 512)
                                      int n_gpu_layers,    // -1 = all, 0 = CPU
                                      int n_threads);      // CPU threads

// Batched evaluation
// prompts:   array of N instruction strings
// articles:  array of N article strings
// n:         batch size (≤ max_seqs, default 16)
// out_labels: pre-allocated int32_t[N], filled with 0=World, 1=Sports, 2=Business, 3=Sci/Tech
EvalCoreStatus eval_core_score_batch(EvalCoreHandle* handle,
                                      const char* const* prompts,
                                      const char* const* articles,
                                      int n,
                                      int32_t* out_labels);

// Cleanup
void eval_core_free(EvalCoreHandle* handle);
const char* eval_core_last_error(void);
```

## Build

```bash
# From repo root
export LLAMA_PATH=/path/to/llama.cpp   # or set in CMakeLists.txt default
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
# Output: build/libeval_core.dylib (macOS) or .so (Linux)
```

### CMake Options

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_PATH` | `../third_party/llama.cpp` | Path to llama.cpp repo root |
| `CMAKE_BUILD_TYPE` | `Release` | Build type |

## Runtime Tuning (Environment Variables)

Read once at `eval_core_load_model()`:

| Env Var | Default | Description |
|---------|---------|-------------|
| `EVAL_CORE_MAXSEQS` | 16 | Max concurrent sequences per batch |
| `EVAL_CORE_NBATCH` | 4096 | Logical batch size cap (tokens) |
| `EVAL_CORE_NUBATCH` | 4096 | Physical batch size cap (tokens) |
| `EVAL_CORE_CTX_MULT` | 16 | Total context = `n_ctx * CTX_MULT` |
| `EVAL_CORE_KV_QUANT` | 0 | Quantize KV cache to Q8_0 (1=yes) |
| `EVAL_CORE_FLASH_ATTN` | 0 | Force flash attention (1=yes) |
| `EVAL_CORE_TIMING` | 0 | Print per-stage timing to stderr |
| `EVAL_CORE_DEBUG` | 0 | Print top-10 tokens for sequence 0 |

Example:
```bash
EVAL_CORE_MAXSEQS=32 EVAL_CORE_TIMING=1 MODEL=... python3 bench.py
```

## Label Token Strategy

Mistral's SentencePiece tokenizer splits `" World"` → `[▁, World]`. All four labels share the leading space token `▁`. We tokenize each label with a leading space and take the **last token piece**:

```cpp
static const char* kLabels[4] = {" World", " Sports", " Business", " Sci"};
// tokenize → take toks.back() → unique ID per class
```

If predictions collapse to one class, run with `EVAL_CORE_DEBUG=1` to verify the four label token IDs are distinct.

## Python Usage (via ctypes)

```python
import ctypes, pathlib

LIB = str(next(pathlib.Path("build").rglob("libeval_core.dylib")).resolve())
lib = ctypes.CDLL(LIB)

lib.eval_core_last_error.restype = ctypes.c_char_p
lib.eval_core_load_model.restype = ctypes.c_void_p
lib.eval_core_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.eval_core_score_batch.restype = ctypes.c_int
lib.eval_core_score_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32)
]
lib.eval_core_free.argtypes = [ctypes.c_void_p]

# Load
h = lib.eval_core_load_model(b"/path/to/model.gguf", 512, -1, 8)

# Batch evaluate
prompts = (ctypes.c_char_p * n)([p.encode() for p in prompts_list])
articles = (ctypes.c_char_p * n)([a.encode() for a in articles_list])
labels = (ctypes.c_int32 * n)()

rc = lib.eval_core_score_batch(h, prompts, articles, n, labels)
assert rc == 0, lib.eval_core_last_error().decode()

# labels[i] ∈ {0,1,2,3}
lib.eval_core_free(h)
```

See `../benchmarks/smoke_test.py` and `../benchmarks/bench.py` for complete examples.

## Performance Notes

- **Batch size**: Limited by `EVAL_CORE_MAXSEQS` (default 16) and total context (`n_ctx * CTX_MULT`). Increase `CTX_MULT` for larger batches.
- **Metal (macOS)**: First decode allocates Metal buffers; subsequent calls reuse. Warm-up run recommended.
- **KV cache**: Cleared automatically at start of each `score_batch`. No cross-batch contamination.
- **Threading**: `eval_core_score_batch` holds a mutex on the handle. For parallel evaluation, create multiple handles (each loads own model copy).

## Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `dlopen: image not found` | `libllama.dylib` not on RPATH | Rebuild from clean `build/`; ensure `llama.cpp` built as shared |
| All predictions = `World` | Label token collision | Run with `EVAL_CORE_DEBUG=1`; verify 4 distinct token IDs |
| `rc=-4` (decode failed) | Context too small | Increase `n_ctx` or `EVAL_CORE_CTX_MULT` |
| Abort on exit (`GGML_ASSERT`) | Metal teardown bug | Cosmetic; call `eval_core_free` explicitly; use `os._exit(0)` in benchmarks |

## Files

| File | Purpose |
|------|---------|
| `eval_core.hpp` | Public C API |
| `eval_core.cpp` | Implementation (llama.cpp integration, batching, label tokens) |
| `CMakeLists.txt` | Build config (in repo root) |

## Integration with PromptRL

`RewardCalculator` (Python) should call `eval_core_score_batch` instead of looping `llama_cpp_python`. This is the primary optimization target — reduces per-step reward eval from ~20s to ~2s.