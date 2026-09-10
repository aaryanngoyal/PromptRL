"""
PromptHarness: evaluates a prompt template against a piece of AG News
using a local Mistral-7B-Instruct model (via llama-cpp-python).

"""

from llama_cpp import Llama
from datasets import load_dataset

LABELS = ["World", "Sports", "Business", "Sci/Tech"]    


class PromptHarness:
    def __init__(
        self,
        model_path: str = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_ctx: int = 2048,
        n_gpu_layers: int = 20, # 0 for CPU and >0 for GPU acceleration
        eval_size: int = 150, #kept at 150 for testing to manage time due to cpu constraints
        n_threads: int = 6,  
        seed: int = 42,
    ):
        # --- Load the LLM ---
        # CPU-only inference: no CUDA GPU available, so n_gpu_layers=0.
        # n_threads controls CPU parallelism for each call.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=seed,
            verbose=False,
        )

        self.temperature = 0.1
        self.max_tokens = 20  

        raw = load_dataset("fancyzhx/ag_news", split="test")
        self.eval_data = raw.select(range(eval_size))

    def _format_prompt(self, prompt_template: str, article: str) -> str:
        """
        Wraps the instruction and article into Mistral's chat format.

        Mistral-Instruct needs <s>[INST] {instruction} [/INST] .
    
        We keep the two separate so the RL agent only ever edits the
        instruction, never the data.
        """
        user_content = f"{prompt_template}\n\nText: {article}"
        return f"[INST] {user_content} [/INST]"

    def _parse_response(self, text: str) -> int:
        """
        Maps the model's raw text output to an integer label in {0,1,2,3},
        matching AG News's own label scheme (LABELS above).
        """
        cleaned = text.strip().lower()
        keyword_to_label = {
            "sci/tech": 3,
            "science": 3,
            "technology": 3,
            "tech": 3,
            "business": 2,
            "sports": 1,
            "sport": 1,
            "world": 0,
        }

        for keyword, label in keyword_to_label.items():
            if keyword in cleaned:
                return label

        # Unparseable output — fall back to a random guess rather than silently miscounting it as correct or crashing similar to the usual LLM interpretations
        return int(__import__("random").randint(0, 3))

    def evaluate_single(self, prompt_template: str, article: str, true_label: int) -> float:

        """
        Evaluate a prompt template on a single (article, true label) pair.
        Return 1.0 if correct else 0.0
        """

        full_prompt = self._format_prompt(prompt_template, article)
        
        output = self.llm(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "[INST]"],
        )

        response_text = output["choices"][0]["text"]
        predicted_label = self._parse_response(response_text)
        return 1.0 if predicted_label == true_label else 0.0

    def evaluate(self, prompt_template: str) -> float:

        """
        Runs prompt_template over the whole eval slice and returns accuracy
        as a float in [0, 1]. This is the reward signal the RL agent uses.
        """

        correct = 0
        for example in self.eval_data:
            correct += self.evaluate_single(
                prompt_template, example["text"], example["label"]
            )

        return correct / len(self.eval_data)

    def debug_inspect(self, prompt_template: str, n: int = 10) -> None:
        """
        Debugging helper. NOT part of the reward pipeline. Prints the raw model output, alongside the true label and what the  parser mapped it to. Use this to see why a prompt is scoring the way it does, instead of just the final accuracy number.
        """
        for i, example in enumerate(self.eval_data):
            if i >= n:
                break
            article = example["text"]
            true_label = example["label"]

            full_prompt = self._format_prompt(prompt_template, article)
            output = self.llm(
                full_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "[INST]"],
            )
            raw_text = output["choices"][0]["text"]
            parsed = self._parse_response(raw_text)

            match = "✓" if parsed == true_label else "✗"
            print(
                f"{match} true={LABELS[true_label]:<8} "
                f"parsed={LABELS[parsed]:<8} "
                f"raw_output={raw_text.strip()!r}"
            )

   