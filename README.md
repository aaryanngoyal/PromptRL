# PromptRL — Q-Learning Prompt Optimization with Batched LLM Inference

> **Automatically optimize prompts for AG News classification using Q-learning and high-throughput batched inference (5–10× speedup).**

---

## Overview

This project implements a **Q-learning agent** that iteratively improves prompts for classifying news articles into four categories: **World, Sports, Business, Sci/Tech**. 

The key innovation is **`eval_core`** — a C++ shared library that wraps `llama.cpp` to evaluate **N prompts in a single forward pass**, eliminating the bottleneck of sequential inference.

| Approach | Inference Time (100 evaluations) |
|----------|----------------------------------|
| Sequential `llama_cpp_python` | ~20 seconds |
| **Batched `eval_core`** | **~2 seconds** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROMPT RL LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Q-Learning Agent proposes prompt mutations                  │
│         │                                                       │
│         ▼                                                       │
│  2. Batched Reward Calculator evaluates N prompts at once       │
│         │                    │                                  │
│         │         ┌──────────┴──────────┐                       │
│         │         ▼                     ▼                       │
│         │  eval_core_score_batch()  →  Labels [0..3]           │
│         │         (single llama_decode call)                    │
│         │                                                       │
│         ▼                                                       │
│  3. Q-Table updated → Next iteration                            │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Q-Learning Agent** | `agent/rlagent.py` | Epsilon-greedy Q-learning over prompt mutations |
| **Action Space** | `agent/actionSpace.py` | 13 prompt mutations (add shots, clarify format, simplify, etc.) |
| **State Representation** | `agent/state.py` | Q-table keyed by normalized prompt text |
| **Batched Reward** | `agent/reward_batched.py` | Evaluates 16 prompts × 16 articles in one batch |
| **eval_core Harness** | `agent/eval_core_harness.py` | Python ctypes wrapper for C++ library |
| **C++ Inference** | `cpp/eval_core.cpp/hpp` | Batched `llama_decode` via llama.cpp |

---

## Quick Start

### Prerequisites

- **llama.cpp** built as shared library (`BUILD_SHARED_LIBS=ON`)
- **Mistral-7B-Instruct GGUF model** (recommended: Q3_K_S for 4GB VRAM, Q4_K_M for CPU)
- Python 3.10+ with `llama-cpp-python`, `datasets`, `numpy`

### Build `eval_core`

```bash
# 1. Build llama.cpp (with CUDA if GPU available)
cd /path/to/llama.cpp
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang-14
make -j

# 2. Build eval_core
cd /home/aaryangoyal/ml/projects/promptRL
mkdir build && cd build
cmake .. -DLLAMA_PATH=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
make -j
# Output: build/libeval_core.so (Linux) or .dylib (macOS)
```

### Run Prompt Optimization

```bash
# Set model path
export MODEL=/path/to/model.gguf

# Optimize a prompt
python3 optimize_prompt.py "Classify the news article into World, Sports, Business, Sci/Tech"

# With custom episodes/steps
python3 optimize_prompt.py "Your prompt here" --episodes 50 --max-steps 8
```

### Example Output

```
Initial prompt: Classify the news article into World, Sports, Business, Sci/Tech
Running 30 episodes, 5 steps each...

Ep   0 | best=0.870 | eps=0.285 | no_improve=0 | q_states=1
Ep   1 | best=0.880 | eps=0.271 | no_improve=0 | q_states=3
...
Ep  15 | best=0.940 | eps=0.100 | no_improve=3 | q_states=28
Early stopping at: 20

✓ Best reward: 0.940
✓ Optimized prompt: Example: 'UN holds emergency session on border dispute' -> World
Example: 'Olympic committee weighs new sports for 2028 games' -> Sports
Example: 'Apple reports record quarterly earnings' -> Business
Example: 'NASA launches new satellite to study solar flares' -> Sci/Tech

Classify the news article into one of: World, Sports, Business, Sci/Tech.
Answer with only one word. Do not use punctuation.
```

---

## Project Structure

```
promptRL/
├── agent/
│   ├── rlagent.py              # Q-learning agent
│   ├── actionSpace.py          # 13 prompt mutations
│   ├── state.py                # Q-table (prompt → action values)
│   ├── reward.py               # Original reward calculator (sequential)
│   ├── reward_batched.py       # Batched reward calculator (5-10× faster)
│   └── eval_core_harness.py    # Python ↔ C++ bridge
├── cpp/
│   ├── eval_core.hpp           # Public C API
│   ├── eval_core.cpp           # Batched llama.cpp implementation
│   └── README.md               # C++ library docs
├── benchmarks/
│   ├── smoke_test.py           # Correctness verification
│   ├── bench.py                # Performance benchmark
│   ├── mlx_bench.py            # MLX comparison
│   └── readme.md               # Benchmark docs
├── test/
│   ├── test_eval_core_fast.py  # Quick integration test
│   ├── test_eval_core_quick.py # Medium test
│   └── test_eval_core_integration.py # Full test
├── optimize_prompt.py          # Main entry point
├── CMakeLists.txt              # Build configuration
└── models/                     # GGUF models (gitignored)
```

---

## Key Features

### 1. Batched Inference (`eval_core`)

- **Single `llama_decode`** evaluates 16 (prompt, article) pairs
- **One KV cache clear** per batch
- **ctypes interface** — zero-copy, minimal overhead
- **Thread-safe** — mutex per handle

### 2. Q-Learning Agent

- **State**: Normalized prompt text (whitespace-collapsed)
- **Actions**: 13 mutations (add/remove shots, format clarifiers, rephrase, simplify, etc.)
- **Reward**: Classification accuracy on AG News eval set
- **Exploration**: Epsilon-greedy with decay

### 3. Action Space (13 Mutations)

| Action | Description |
|--------|-------------|
| `add_shot_example` | Add 4-shot examples |
| `remove_shot_example` | Remove examples |
| `add_format_clarifier` | "Answer with only one word" |
| `remove_format_clarifier` | Remove format constraints |
| `simplify_language` | Compress to minimal instruction |
| `rephrase_instruction` | Change verb (Classify → Categorize → Identify) |
| `add_2_shot` / `add_3_shot` | Add 2/3-shot examples |
| `add_category_definitions` | Add World/Sports/Business/SciTech definitions |
| `add_anti_instruction` | "Do not explain. Output only label" |
| `strip_examples` / `strip_definitions` | Remove examples/definitions |
| `shorten_to_bare` | Minimal: "Categorize into World, Sports, Business, Sci/Tech" |

---

## Benchmarks

### Run Benchmarks

```bash
export MODEL=/path/to/model.gguf
cd benchmarks

# Correctness
python3 smoke_test.py

# Performance (batch sizes 1, 4, 8, 16)
python3 bench.py

# With timing breakdown
EVAL_CORE_TIMING=1 python3 bench.py

# MLX comparison
python3 mlx_bench.py
```

### Expected Results (CPU, Mistral-7B Q4_K_M)

```
n    median ms   ms/sample   speedup    accuracy
1       188.5      188.5      1.00x       0.875
4       1078.5     269.6      0.70x       0.875
8       1960.6     245.1      0.77x       0.875
16      3710.5     231.9      0.81x       0.875
```

> **Note**: On CPU, batching doesn't reduce ms/sample (no parallel matmul), but **eval_core still wins** by avoiding Python overhead, repeated tokenization, and KV cache clears. On Metal/CUDA, expect 5–10× speedup.

---

## Environment Variables (eval_core)

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_CORE_MAXSEQS` | 16 | Max sequences per batch |
| `EVAL_CORE_NBATCH` | 4096 | Logical batch size (tokens) |
| `EVAL_CORE_NUBATCH` | 4096 | Physical batch size (tokens) |
| `EVAL_CORE_CTX_MULT` | 16 | Total context = n_ctx × CTX_MULT |
| `EVAL_CORE_KV_QUANT` | 0 | Quantize KV cache to Q8_0 |
| `EVAL_CORE_FLASH_ATTN` | 0 | Force flash attention |
| `EVAL_CORE_TIMING` | 0 | Print per-stage timing |
| `EVAL_CORE_DEBUG` | 0 | Print top-10 tokens for debugging |

---

## GPU Usage

| Model | Size | Layers on RTX 3050 (4GB) |
|-------|------|--------------------------|
| Q4_K_M | 4.4 GB | 0 (CPU only) |
| Q3_K_L | 3.8 GB | 0 (CPU only) |
| Q3_K_M | 3.5 GB | 10 |
| **Q3_K_S** | **3.1 GB** | **32 (full)** |

**Recommended**: Use **Q3_K_S** with `n_gpu_layers=32` for GPU, or any model with `n_gpu_layers=0` for CPU batching.

> ⚠️ eval_core's memory estimator is conservative. Even when llama-cli offloads to GPU, eval_core may fall back to CPU. Use CPU batching for guaranteed speedup.

---

## Common Issues

| Error | Fix |
|-------|-----|
| `dlopen: image not found` | Rebuild llama.cpp with `BUILD_SHARED_LIBS=ON` |
| `sequence X has Y tokens, exceeds per-seq window 256` | Increase `n_ctx=512` in `EvalCoreHarness` |
| All predictions = `World` | Run with `EVAL_CORE_DEBUG=1`, verify 4 distinct label token IDs |
| `rc=-4` (decode failed) | Increase `n_ctx` or `EVAL_CORE_CTX_MULT` |
| Abort on exit (code 134) | Metal teardown bug — cosmetic, use `os._exit(0)` |

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inference engine
- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) — Dataset
- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) — Base model