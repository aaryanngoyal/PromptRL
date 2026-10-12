# benchmarks — Python harnesses

Three scripts that exercise the `eval_core` shared library.

## Files

| Script          | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| `smoke_test.py` | Correctness check — 4 hand-labeled articles, asserts expected output |
| `bench.py`      | Performance benchmark — times `score_batch` at batch sizes 1/4/8/16  |
| `mlx_bench.py`  | MLX comparison — same task using Apple's native framework            |

## Prerequisites

```bash
# Build the shared library first
cd ..
./build.sh

# Pick a model
export MODEL=$PWD/../models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

## `smoke_test.py` — correctness

Classifies four AG News articles in a single batch of 4 and prints the
predicted label for each.

```bash
MODEL=$MODEL python3 smoke_test.py
```

Expected output:

```
loading /path/to/model.gguf ...
model loaded, handle = 0x...
  World      The UN Security Council met today to discuss the ongoing...
  Sports     The Lakers defeated the Celtics 112-108 in a thrilling...
  Business   Apple reported quarterly earnings that beat analyst...
  Sci/Tech   NASA launched a new satellite to study solar flares in...
done
```

If any label is wrong, the model's tokenizer likely splits the class
strings differently. Add `EVAL_CORE_DEBUG=1` to see the top-10 tokens
at the last position — the winning token should be one of `" World"`,
`" Sports"`, `" Business"`, `" Sci"`.

```bash
EVAL_CORE_DEBUG=1 MODEL=$MODEL python3 smoke_test.py 2>&1 | grep -A12 "top-10"
```

## `bench.py` — performance benchmark

Times `score_batch` at batch sizes 1, 4, 8, 16. Median of 20 runs after
a warm-up call. Prints a table with median ms, ms/sample, speedup
relative to n=1, and accuracy.

```bash
MODEL=$MODEL python3 bench.py
```

Add per-stage timing:

```bash
EVAL_CORE_TIMING=1 MODEL=$MODEL python3 bench.py > /tmp/bench.out 2>&1
grep "\[timing\]" /tmp/bench.out
```

Output format:

```
[timing] n= 1 tokens=   63  tokenize=  0.08ms  build=  4.9ms  decode=  0.5ms  argmax=183.0ms  total= 188.5ms
[timing] n= 4 tokens=  256  tokenize=  0.37ms  build=  4.9ms  decode=953.8ms  argmax=119.3ms  total=1078.5ms
[timing] n= 8 tokens=  512  tokenize=  0.75ms  build=  6.0ms  decode=1834.8ms  argmax=119.3ms  total=1960.6ms
[timing] n=16 tokens= 1024  tokenize=  1.20ms  build=  6.0ms  decode=3583.0ms  argmax=119.3ms  total=3710.5ms
```

### Note on `os._exit(0)`

On macOS, llama.cpp's Metal backend aborts during process teardown
(`ggml_metal_device_free` asserts). The script may exit with code 134.
This happens **after** all output has been written and does not affect
the benchmark numbers. The `backend_acquire`/`backend_release` refcount
in `eval_core.cpp` mitigates it when `eval_core_free` is called
explicitly, but the abort can still fire on abnormal exits.

### Pool cycling

`bench.py` cycles a 4-entry `POOL` list to reach batch sizes larger than 4. At n=16, each article is scored 4 times. If you replace `POOL` with
16 distinct articles, remove the `i % len(POOL)` and use `POOL[:n]`.

## `mlx_bench.py` — MLX comparison

The same task using `mlx-lm`'s `batch_generate`. Measures whether MLX
batches effectively where llama.cpp/Metal does not.

```bash
# Install once
pip install mlx-lm

# Download an equivalent MLX model
hf download mlx-community/Mistral-7B-Instruct-v0.2-4bit \
  --local-dir ~/models/mlx-mistral-7b

# Run
python3 mlx_bench.py
```

Expected output (if MLX batches correctly):

```
  n   median ms   ms/sample
  1      185.00      185.00   speedup 1.00x
  4       60.00       15.00   speedup 12.3x
  8       40.00        5.00   speedup 37.0x
 16       35.00        2.19   speedup 84.5x
```

If the `ms/sample` column drops significantly at higher n, MLX's
batched matmul is working and the C++ path is the wrong choice for
Apple Silicon. If the column stays flat around 180–200 ms, MLX has the
same limitation and the C++ implementation is the right deliverable.

## Common issues

| Symptom                                                     | Cause                                  | Fix                                                                  |
| ----------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| `RuntimeError: dlopen ... image not found`                  | `libllama.dylib` not on RPATH          | Rebuild with `./build.sh` from a clean `build/`                      |
| All predictions = `World`                                   | Wrong label token (shared space token) | Check `EVAL_CORE_DEBUG=1` output; `eval_core.cpp` uses `toks.back()` |
| `score_batch failed: rc=-4`                                 | `llama_decode` returned non-zero       | Often `n_ctx` too small; increase `n_ctx` in load call               |
| Process aborts with `GGML_ASSERT([rsets->data count] == 0)` | Metal teardown bug on abnormal exit    | Cosmetic — output has already been written                           |
| Accuracy 0/N                                                | `POOL` gold labels wrong               | Compare `pred=` and `gold=` in debug output                          |
