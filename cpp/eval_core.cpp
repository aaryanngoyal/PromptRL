#include "eval_core.hpp"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

// ============================================================================
//  Environment-variable tuning
// ============================================================================
//
//  All knobs are read once, at load_model time. Change the env var, rerun.
//
//    EVAL_CORE_MAXSEQS    (int, default 16)   max concurrent sequences
//    EVAL_CORE_NBATCH     (int, default 4096) logical batch size cap
//    EVAL_CORE_NUBATCH    (int, default 4096) physical batch size cap
//    EVAL_CORE_CTX_MULT   (int, default 16)   total_ctx = n_ctx * CTX_MULT
//    EVAL_CORE_KV_QUANT   (0/1, default 0)    quantize KV cache to Q8_0
//    EVAL_CORE_FLASH_ATTN (0/1, default 0)    force-enable flash attention
//    EVAL_CORE_TIMING     (0/1, default 0)    print per-stage timing
//    EVAL_CORE_DEBUG      (0/1, default 0)    print top-10 tokens per batch
//
namespace {

int env_int(const char *name, int default_val) {
  const char *s = std::getenv(name);
  if (!s || !*s)
    return default_val;
  return std::atoi(s);
}

bool env_flag(const char *name, bool default_val = false) {
  const char *s = std::getenv(name);
  if (!s || !*s)
    return default_val;
  return std::atoi(s) != 0;
}

// ---------------------------------------------------------------------------
//  Backend lifecycle (refcounted)
// ---------------------------------------------------------------------------
std::mutex g_backend_mtx;
int g_backend_refs = 0;

void backend_acquire() {
  std::lock_guard<std::mutex> lk(g_backend_mtx);
  if (g_backend_refs++ == 0) {
    llama_backend_init();
    std::atexit([]() {
      std::lock_guard<std::mutex> lk2(g_backend_mtx);
      if (g_backend_refs > 0) {
        llama_backend_free();
        g_backend_refs = 0;
      }
    });
  }
}

void backend_release() {
  std::lock_guard<std::mutex> lk(g_backend_mtx);
  if (g_backend_refs > 0 && --g_backend_refs == 0) {
    llama_backend_free();
  }
}

// ---------------------------------------------------------------------------
//  Error buffer
// ---------------------------------------------------------------------------
thread_local char g_last_error[1024] = {0};

void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
  va_end(ap);
}

// ---------------------------------------------------------------------------
//  KV-cache clear shim
// ---------------------------------------------------------------------------
static inline void clear_kv_cache(llama_context *ctx) {
#if defined(EVALCORE_LLAMA_NEW_MEMORY_API)
  llama_memory_clear(llama_get_memory(ctx), /*full=*/true);
#elif defined(EVALCORE_LLAMA_KV_SELF_API)
  llama_kv_self_clear(ctx);
#elif defined(EVALCORE_LLAMA_OLD_KV_CACHE_API)
  llama_kv_cache_clear(ctx);
#else
#error                                                                         \
    "No llama.cpp KV-cache clear API detected. Re-run cmake with a valid LLAMA_PATH."
#endif
}

// ---------------------------------------------------------------------------
//  Timing helper
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

inline double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() /
         1e6;
}

} // namespace

// ============================================================================
//  Opaque handle
// ============================================================================
struct EvalCoreHandle {
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  int n_ctx = -1; // per-sequence context window
  int32_t n_vocab = 0;

  // Tunables captured at load time so score_batch doesn't re-read env.
  int max_seqs = 16;
  bool timing = false;
  bool debug = false;

  // Label token IDs: 0=World 1=Sports 2=Business 3=Sci/Tech.
  // We store the LAST BPE piece of each label because Mistral's SPM
  // tokenizer splits " World" into [▁, World] and every class shares
  // the same leading whitespace marker.
  llama_token label_tokens[4] = {0, 0, 0, 0};

  std::mutex mtx;
};

// ============================================================================
//  Public C API
// ============================================================================

extern "C" const char *eval_core_last_error(void) { return g_last_error; }

extern "C" EvalCoreHandle *eval_core_load_model(const char *model_path,
                                                int n_ctx, int n_gpu_layers,
                                                int n_threads) {
  if (!model_path || n_ctx <= 0 || n_threads <= 0) {
    set_error("invalid arguments to eval_core_load_model");
    return nullptr;
  }

  backend_acquire();

  // ---- Read tunables from env ------------------------------------------
  const int max_seqs = env_int("EVAL_CORE_MAXSEQS", 16);
  const int n_batch = env_int("EVAL_CORE_NBATCH", 4096);
  const int n_ubatch = env_int("EVAL_CORE_NUBATCH", 4096);
  const int ctx_mult = env_int("EVAL_CORE_CTX_MULT", 16);
  const bool kv_quant = env_flag("EVAL_CORE_KV_QUANT", false);
  const bool flash_attn = env_flag("EVAL_CORE_FLASH_ATTN", false);
  const bool timing_env = env_flag("EVAL_CORE_TIMING", false);
  const bool debug_env = env_flag("EVAL_CORE_DEBUG", false);

  // ---- 1. Load model ----------------------------------------------------
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;

  llama_model *model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    set_error("failed to load GGUF model: %s", model_path);
    backend_release();
    return nullptr;
  }

  // ---- 2. Create context ------------------------------------------------
  const int total_ctx = n_ctx * ctx_mult;

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = (uint32_t)total_ctx;
  cparams.n_batch = (uint32_t)n_batch;
  cparams.n_ubatch = (uint32_t)n_ubatch;
  cparams.n_seq_max = (uint32_t)max_seqs;

  if (flash_attn) {
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  }

  if (kv_quant) {
    cparams.type_k = GGML_TYPE_Q8_0;
    cparams.type_v = GGML_TYPE_Q8_0;
  }

  llama_context *ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    set_error("failed to create llama context (total_ctx=%d, n_seq_max=%d)",
              total_ctx, max_seqs);
    llama_model_free(model);
    backend_release();
    return nullptr;
  }

  llama_set_n_threads(ctx, /*n_threads=*/n_threads,
                      /*n_threads_batch=*/n_threads);

  const llama_vocab *vocab = llama_model_get_vocab(model);

  // ---- 3. Allocate handle ----------------------------------------------
  EvalCoreHandle *h = new (std::nothrow) EvalCoreHandle();
  if (!h) {
    set_error("out of memory allocating EvalCoreHandle");
    llama_free(ctx);
    llama_model_free(model);
    backend_release();
    return nullptr;
  }
  h->model = model;
  h->ctx = ctx;
  h->n_ctx = n_ctx;
  h->n_vocab = llama_vocab_n_tokens(vocab);
  h->max_seqs = max_seqs;
  h->timing = timing_env;
  h->debug = debug_env;

  // ---- 4. Resolve the four label token IDs -------------------------------
  static const char *kLabels[4] = {" World", " Sports", " Business", " Sci"};

  for (int i = 0; i < 4; ++i) {
    const char *s = kLabels[i];
    const int len = (int)strlen(s);

    int n_tok = llama_tokenize(vocab, s, len, nullptr, 0,
                               /*add_special=*/false,
                               /*parse_special=*/false);
    if (n_tok < 0)
      n_tok = -n_tok;
    if (n_tok <= 0) {
      set_error("tokenization probe failed for label '%s'", s);
      eval_core_free(h);
      return nullptr;
    }

    std::vector<llama_token> toks(n_tok);
    const int got = llama_tokenize(vocab, s, len, toks.data(), n_tok,
                                   /*add_special=*/false,
                                   /*parse_special=*/false);
    if (got <= 0) {
      set_error("tokenization failed for label '%s'", s);
      eval_core_free(h);
      return nullptr;
    }
    h->label_tokens[i] = toks.back();
  }

  std::fprintf(stderr,
               "[eval_core] loaded: n_ctx/seq=%d total_ctx=%d max_seqs=%d "
               "n_batch=%d n_ubatch=%d kv_quant=%d flash_attn=%d\n",
               n_ctx, total_ctx, max_seqs, n_batch, n_ubatch, (int)kv_quant,
               (int)flash_attn);

  return h;
}

extern "C" EvalCoreStatus eval_core_score_batch(EvalCoreHandle *handle,
                                                const char *const *prompts,
                                                const char *const *articles,
                                                int n, int32_t *out_labels) {
  if (!handle || !prompts || !articles || !out_labels || n <= 0) {
    set_error("invalid arguments to eval_core_score_batch");
    return EVAL_CORE_ERR_INVALID_ARGS;
  }
  if (n > handle->max_seqs) {
    set_error("n=%d exceeds max_seqs=%d", n, handle->max_seqs);
    return EVAL_CORE_ERR_INVALID_ARGS;
  }

  std::lock_guard<std::mutex> lock(handle->mtx);
  const llama_vocab *vocab = llama_model_get_vocab(handle->model);

  auto t_start = Clock::now();
  auto t_after_tk = t_start;
  auto t_after_bd = t_start;
  auto t_after_dc = t_start;

  // ---- 1. Tokenize ------------------------------------------------------
  std::vector<std::vector<llama_token>> seq_tokens(n);
  int64_t total_tokens = 0;

  for (int i = 0; i < n; ++i) {
    std::string text;
    text.reserve(64 + (prompts[i] ? strlen(prompts[i]) : 0) +
                 (articles[i] ? strlen(articles[i]) : 0));
    text += "[INST] ";
    if (prompts[i])
      text += prompts[i];
    text += "\n\nText: ";
    if (articles[i])
      text += articles[i];
    text += " [/INST] ";

    int n_tok = llama_tokenize(vocab, text.data(), (int)text.size(), nullptr, 0,
                               /*add_special=*/true,
                               /*parse_special=*/true);
    if (n_tok < 0)
      n_tok = -n_tok;
    if (n_tok <= 0) {
      set_error("tokenization produced zero tokens for sequence %d", i);
      return EVAL_CORE_ERR_TOKENIZE;
    }

    seq_tokens[i].resize(n_tok);
    const int got = llama_tokenize(vocab, text.data(), (int)text.size(),
                                   seq_tokens[i].data(), n_tok,
                                   /*add_special=*/true,
                                   /*parse_special=*/true);
    if (got <= 0) {
      set_error("tokenization failed for sequence %d", i);
      return EVAL_CORE_ERR_TOKENIZE;
    }
    seq_tokens[i].resize(got);
    total_tokens += got;

    if ((int)seq_tokens[i].size() > handle->n_ctx) {
      set_error("sequence %d has %zu tokens, exceeds per-seq window %d", i,
                seq_tokens[i].size(), handle->n_ctx);
      return EVAL_CORE_ERR_OOM;
    }
  }
  t_after_tk = Clock::now();

  // ---- 2. Reset KV cache ------------------------------------------------
  clear_kv_cache(handle->ctx);

  // ---- 3. Build one batch spanning all n sequences ----------------------
  llama_batch batch = llama_batch_init(/*n_tokens=*/(int32_t)total_tokens,
                                       /*embd=*/0,
                                       /*n_seq_max=*/n);

  std::vector<int32_t> last_batch_idx(n, -1);

  int32_t b = 0;
  for (int i = 0; i < n; ++i) {
    const int L = (int)seq_tokens[i].size();
    for (int j = 0; j < L; ++j) {
      batch.token[b] = seq_tokens[i][j];
      batch.pos[b] = j;
      batch.n_seq_id[b] = 1;
      batch.seq_id[b][0] = i;
      const bool is_last = (j == L - 1);
      batch.logits[b] = is_last ? 1 : 0;
      if (is_last)
        last_batch_idx[i] = b;
      ++b;
    }
  }
  batch.n_tokens = (int32_t)total_tokens;
  t_after_bd = Clock::now();

  // ---- 4. One llama_decode ---------------------------------------------
  const int rc = llama_decode(handle->ctx, batch);
  t_after_dc = Clock::now();

  if (rc != 0) {
    llama_batch_free(batch);
    set_error("llama_decode failed (rc=%d)", rc);
    return EVAL_CORE_ERR_BATCH_DECODE;
  }

  // ---- 5. Debug: top-10 tokens at last position of seq 0 ----------------
  if (handle->debug) {
    const float *logits0 = llama_get_logits_ith(handle->ctx, last_batch_idx[0]);
    if (logits0) {
      std::vector<std::pair<float, int>> top;
      top.reserve(handle->n_vocab);
      for (int v = 0; v < handle->n_vocab; ++v) {
        top.emplace_back(logits0[v], v);
      }
      std::partial_sort(
          top.begin(), top.begin() + 10, top.end(),
          [](const std::pair<float, int> &a, const std::pair<float, int> &c) {
            return a.first > c.first;
          });
      std::fprintf(stderr, "[debug] top-10 at last pos of seq 0:\n");
      for (int k = 0; k < 10; ++k) {
        char buf[128] = {0};
        const int np = llama_token_to_piece(vocab, top[k].second, buf,
                                            sizeof(buf) - 1, 0, false);
        std::fprintf(stderr, "  #%d logit=%9.3f id=%-6d piece=\"%s\"\n", k,
                     top[k].first, top[k].second, np > 0 ? buf : "?");
      }
      std::fprintf(stderr,
                   "[debug] labels: World=%d Sports=%d Business=%d Sci=%d\n",
                   (int)handle->label_tokens[0], (int)handle->label_tokens[1],
                   (int)handle->label_tokens[2], (int)handle->label_tokens[3]);
    }
  }

  // ---- 6. Argmax over the four label tokens ----------------------------
  for (int i = 0; i < n; ++i) {
    const float *logits = llama_get_logits_ith(handle->ctx, last_batch_idx[i]);
    if (!logits) {
      llama_batch_free(batch);
      set_error("llama_get_logits_ith returned NULL for sequence %d", i);
      return EVAL_CORE_ERR_BATCH_DECODE;
    }

    int best = 0;
    float best_v = logits[handle->label_tokens[0]];
    for (int k = 1; k < 4; ++k) {
      const float v = logits[handle->label_tokens[k]];
      if (v > best_v) {
        best_v = v;
        best = k;
      }
    }
    out_labels[i] = best;
  }

  auto t_end = Clock::now();

  if (handle->timing) {
    std::fprintf(stderr,
                 "[timing] n=%2d tokens=%5lld  "
                 "tokenize=%6.2fms  build=%6.2fms  decode=%6.2fms  "
                 "argmax=%6.2fms  total=%6.2fms\n",
                 n, (long long)total_tokens, ms_between(t_start, t_after_tk),
                 ms_between(t_after_tk, t_after_bd),
                 ms_between(t_after_bd, t_after_dc),
                 ms_between(t_after_dc, t_end), ms_between(t_start, t_end));
  }

  llama_batch_free(batch);
  return EVAL_CORE_OK;
}

extern "C" void eval_core_free(EvalCoreHandle *handle) {
  if (!handle)
    return;
  if (handle->ctx)
    llama_free(handle->ctx);
  if (handle->model)
    llama_model_free(handle->model);
  delete handle;
  backend_release();
}