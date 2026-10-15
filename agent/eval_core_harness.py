"""
EvalCoreHarness: Drop-in replacement for PromptHarness using eval_core batched inference.

Provides the same interface as PromptHarness but uses the C++ eval_core library
for 5-10x faster batched evaluation.
"""

import ctypes
import pathlib
import os
from datasets import load_dataset

# Label mapping: AG News uses 0=World, 1=Sports, 2=Business, 3=Sci/Tech
# eval_core returns the same mapping
LABELS = ["World", "Sports", "Business", "Sci/Tech"]


class EvalCoreHarness:
    def __init__(
        self,
        model_path: str = None,
        n_ctx: int = 512,
        n_gpu_layers: int = -1,
        n_threads: int = 8,
        eval_size: int = 150,
        seed: int = 42,
        lib_path: str = None,
    ):
        """
        Initialize EvalCoreHarness.
        
        Args:
            model_path: Path to GGUF model file. If None, reads from MODEL env var.
            n_ctx: Per-sequence context window (default 512)
            n_gpu_layers: GPU layers to offload (-1 = all, 0 = CPU only)
            n_threads: CPU threads for inference
            eval_size: Number of test examples to use
            seed: Random seed for data selection
            lib_path: Path to libeval_core.dylib/.so. If None, auto-detects in build/
        """
        # Resolve model path
        if model_path is None:
            model_path = os.environ.get("MODEL")
        if not model_path:
            raise ValueError("model_path must be provided or MODEL env var set")
        
        # Resolve library path
        if lib_path is None:
            build_dir = pathlib.Path(__file__).parent.parent / "build"
            matches = list(build_dir.rglob("libeval_core*"))
            if not matches:
                raise FileNotFoundError(
                    f"libeval_core not found in {build_dir}. Build it first with: "
                    "LLAMA_PATH=~/llama.cpp cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j"
                )
            lib_path = str(matches[0].resolve())
        
        # Load the shared library
        self.lib = ctypes.CDLL(lib_path)
        self._setup_signatures()
        
        # Load model
        self.handle = self.lib.eval_core_load_model(
            model_path.encode(), n_ctx, n_gpu_layers, n_threads
        )
        if not self.handle:
            err = self.lib.eval_core_last_error().decode()
            raise RuntimeError(f"Failed to load model: {err}")
        
        # Expose max_seqs for reward calculators (default 16 from eval_core)
        self.max_seqs = 16  # Default EVAL_CORE_MAXSEQS
        
        # Load evaluation dataset (same as PromptHarness)
        raw = load_dataset("fancyzhx/ag_news", split="test")
        self.eval_data = raw.select(range(min(eval_size, len(raw))))
        
        # Cache for single evaluations (compatible with PromptHarness interface)
        self._cache = {}
        self.use_cache = True
        
        # Warm-up call (allocates Metal buffers on macOS)
        dummy_prompts = (ctypes.c_char_p * 1)(b"Classify: ")
        dummy_articles = (ctypes.c_char_p * 1)(b"Test article.")
        dummy_labels = (ctypes.c_int32 * 1)()
        self.lib.eval_core_score_batch(self.handle, dummy_prompts, dummy_articles, 1, dummy_labels)
    
    def _setup_signatures(self):
        """Configure ctypes function signatures."""
        self.lib.eval_core_last_error.restype = ctypes.c_char_p
        
        self.lib.eval_core_load_model.restype = ctypes.c_void_p
        self.lib.eval_core_load_model.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        
        self.lib.eval_core_score_batch.restype = ctypes.c_int
        self.lib.eval_core_score_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
        ]
        
        self.lib.eval_core_free.argtypes = [ctypes.c_void_p]
    
    def _format_prompt(self, prompt_template: str, article: str) -> str:
        """Format prompt in Mistral chat template (same as PromptHarness)."""
        user_content = f"{prompt_template}\n\nText: {article}"
        return f" {user_content} 那些"
    
    def evaluate_single(self, prompt_template: str, article: str, true_label: int) -> float:
        """
        Evaluate a prompt on a single article.
        Uses batched inference under the hood (batch size = 1).
        Compatible with PromptHarness.evaluate_single interface.
        """
        key = (prompt_template, article)
        if self.use_cache and key in self._cache:
            predicted = self._cache[key]
        else:
            # Batch size 1
            n = 1
            c_prompts = (ctypes.c_char_p * n)(prompt_template.encode())
            c_articles = (ctypes.c_char_p * n)(article.encode())
            c_labels = (ctypes.c_int32 * n)()
            
            rc = self.lib.eval_core_score_batch(self.handle, c_prompts, c_articles, n, c_labels)
            if rc != 0:
                err = self.lib.eval_core_last_error().decode()
                raise RuntimeError(f"eval_core_score_batch failed: {err}")
            
            predicted = c_labels[0]
            if self.use_cache:
                self._cache[key] = predicted
        
        return 1.0 if predicted == true_label else 0.0
    
    def evaluate_batch(self, prompt_template: str, articles: list[str], labels: list[int]) -> list[float]:
        """
        Evaluate a single prompt on multiple articles in one batched call.
        Returns list of 1.0/0.0 for each article.
        """
        n = len(articles)
        c_prompts = (ctypes.c_char_p * n)(*[prompt_template.encode()] * n)
        c_articles = (ctypes.c_char_p * n)(*[a.encode() for a in articles])
        c_labels = (ctypes.c_int32 * n)()
        
        rc = self.lib.eval_core_score_batch(self.handle, c_prompts, c_articles, n, c_labels)
        if rc != 0:
            err = self.lib.eval_core_last_error().decode()
            raise RuntimeError(f"eval_core_score_batch failed: {err}")
        
        return [1.0 if c_labels[i] == labels[i] else 0.0 for i in range(n)]
    
    def evaluate(self, prompt_template: str) -> float:
        """
        Evaluate prompt on entire eval dataset.
        Uses batched inference for speed.
        """
        articles = [ex["text"] for ex in self.eval_data]
        labels = [ex["label"] for ex in self.eval_data]
        
        # Process in chunks of max_seqs (default 16)
        max_batch = 16  # Default EVAL_CORE_MAXSEQS
        correct = 0
        
        for i in range(0, len(articles), max_batch):
            batch_articles = articles[i:i + max_batch]
            batch_labels = labels[i:i + max_batch]
            results = self.evaluate_batch(prompt_template, batch_articles, batch_labels)
            correct += sum(results)
        
        return correct / len(articles)
    
    def evaluate_prompts_batch(self, prompts: list[str], articles: list[str], labels: list[int]) -> list[float]:
        """
        Evaluate multiple prompts on multiple articles in one batched call.
        This is the key optimization: N prompts x N articles in ONE forward pass.
        
        Args:
            prompts: List of N prompt templates
            articles: List of N articles  
            labels: List of N true labels
            
        Returns:
            List of N rewards (1.0/0.0)
        """
        n = len(prompts)
        assert len(articles) == n and len(labels) == n
        
        c_prompts = (ctypes.c_char_p * n)(*[p.encode() for p in prompts])
        c_articles = (ctypes.c_char_p * n)(*[a.encode() for a in articles])
        c_labels = (ctypes.c_int32 * n)()
        
        rc = self.lib.eval_core_score_batch(self.handle, c_prompts, c_articles, n, c_labels)
        if rc != 0:
            err = self.lib.eval_core_last_error().decode()
            raise RuntimeError(f"eval_core_score_batch failed: {err}")
        
        return [1.0 if c_labels[i] == labels[i] else 0.0 for i in range(n)]
    
    def debug_inspect(self, prompt_template: str, n: int = 10) -> None:
        """Debug: show predictions for first n examples."""
        for i, example in enumerate(self.eval_data):
            if i >= n:
                break
            article = example["text"]
            true_label = example["label"]
            
            pred = self.evaluate_single(prompt_template, article, true_label)
            match = "✓" if pred == 1.0 else "✗"
            print(f"{match} true={LABELS[true_label]:<8} pred={LABELS[int(pred)]:<8} article={article[:60]}...")
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'handle') and self.handle:
            self.lib.eval_core_free(self.handle)
    
    def close(self):
        """Explicit cleanup."""
        if hasattr(self, 'handle') and self.handle:
            self.lib.eval_core_free(self.handle)
            self.handle = None


# Convenience function for direct batched evaluation (bypasses harness interface)
def evaluate_prompts_batched(
    prompts: list[str],
    articles: list[str],
    model_path: str = None,
    n_ctx: int = 512,
    n_gpu_layers: int = -1,
    n_threads: int = 8,
    lib_path: str = None,
) -> list[int]:
    """
    Standalone function to evaluate N prompts on N articles in one batch.
    Returns raw predicted labels (0-3).
    
    This is the fastest way to use eval_core directly.
    """
    if model_path is None:
        model_path = os.environ.get("MODEL")
    if not model_path:
        raise ValueError("model_path or MODEL env var required")
    
    if lib_path is None:
        build_dir = pathlib.Path(__file__).parent.parent / "build"
        matches = list(build_dir.rglob("libeval_core*"))
        if not matches:
            raise FileNotFoundError(f"libeval_core not found in {build_dir}")
        lib_path = str(matches[0].resolve())
    
    lib = ctypes.CDLL(lib_path)
    lib.eval_core_load_model.restype = ctypes.c_void_p
    lib.eval_core_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.eval_core_score_batch.restype = ctypes.c_int
    lib.eval_core_score_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.eval_core_free.argtypes = [ctypes.c_void_p]
    lib.eval_core_last_error.restype = ctypes.c_char_p
    
    handle = lib.eval_core_load_model(model_path.encode(), n_ctx, n_gpu_layers, n_threads)
    if not handle:
        raise RuntimeError(f"Load failed: {lib.eval_core_last_error().decode()}")
    
    n = len(prompts)
    c_prompts = (ctypes.c_char_p * n)(*[p.encode() for p in prompts])
    c_articles = (ctypes.c_char_p * n)(*[a.encode() for a in articles])
    c_labels = (ctypes.c_int32 * n)()
    
    rc = lib.eval_core_score_batch(handle, c_prompts, c_articles, n, c_labels)
    if rc != 0:
        lib.eval_core_free(handle)
        raise RuntimeError(f"Batch eval failed: {lib.eval_core_last_error().decode()}")
    
    results = list(c_labels)
    lib.eval_core_free(handle)
    return results