"""
Optimized RewardCalculator using eval_core batched evaluation.

Replaces sequential evaluate_single calls with single batched eval_core_score_batch.
"""

import random
from collections import defaultdict
from typing import List


class BatchedRewardCalculator:
    """
    Reward calculator that uses eval_core's batched evaluation.
    
    Instead of calling evaluate_single N times (sequential), it evaluates
    all batch examples in ONE forward pass.
    
    Expected speedup: 5-10x over RewardCalculator + PromptHarness.
    """
    
    def __init__(
        self,
        eval_core_harness,
        num_evals: int = 1,
        seed: int = 42,
        batch_size: int = None,  # None = use all eval_data
    ):
        """
        Args:
            eval_core_harness: Instance of EvalCoreHarness
            num_evals: Number of evaluations per example (for noise reduction)
            seed: Random seed for batch selection
            batch_size: Number of examples per reward evaluation. 
                       If None, uses all eval_data.
        """
        self.harness = eval_core_harness
        self.num_evals = num_evals
        # Use harness's max_seqs if available, else default 16
        self.max_batch_size = getattr(eval_core_harness, 'max_seqs', 16)
        
        # Select balanced batch from eval_data (same logic as RewardCalculator, fixed)
        rng = random.Random(seed)
        by_class = defaultdict(list)
        for i, ex in enumerate(eval_core_harness.eval_data):
            by_class[ex["label"]].append(i)
        
        batch_indices = []
        for _, idxs in by_class.items():
            # Fixed bug: min(len(idxs), ...) not min(by_class, ...)
            if batch_size:
                sample_size = min(len(idxs), max(1, batch_size // len(by_class)))
            else:
                sample_size = len(idxs)
            batch_indices += rng.sample(idxs, sample_size)
        
        self.batch = eval_core_harness.eval_data.select(batch_indices)
        self._batch_articles = [ex["text"] for ex in self.batch]
        self._batch_labels = [ex["label"] for ex in self.batch]
        
        print(f"[BatchedRewardCalculator] Batch size: {len(self.batch)} examples, max_batch: {self.max_batch_size}")
    
    def _evaluate_batch_chunked(self, prompt: str, articles: list, labels: list) -> list:
        """Evaluate in chunks of max_batch_size."""
        n = len(articles)
        if n <= self.max_batch_size:
            return self.harness.evaluate_batch(prompt, articles, labels)
        
        # Chunk into max_batch_size
        all_results = []
        for i in range(0, n, self.max_batch_size):
            chunk_articles = articles[i:i + self.max_batch_size]
            chunk_labels = labels[i:i + self.max_batch_size]
            results = self.harness.evaluate_batch(prompt, chunk_articles, chunk_labels)
            all_results.extend(results)
        return all_results
    
    def get_stable_reward(self, prompt: str) -> float:
        """
        Evaluate prompt on batch using batched calls (chunked if needed).
        
        For num_evals > 1, repeats the batch evaluation (still batched each time).
        """
        n = len(self._batch_articles)
        total_correct = 0
        
        for _ in range(self.num_evals):
            rewards = self._evaluate_batch_chunked(prompt, self._batch_articles, self._batch_labels)
            total_correct += sum(rewards)
        
        return total_correct / (n * self.num_evals)
    
    def get_rewards_for_prompts(self, prompts: List[str]) -> List[float]:
        """
        Evaluate MULTIPLE prompts on the SAME batch.
        
        For each prompt, evaluates on all examples (chunked if needed).
        This is still much faster than sequential single evaluations.
        
        Args:
            prompts: List of prompt strings to evaluate
            
        Returns:
            List of accuracy scores (one per prompt)
        """
        rewards = []
        for prompt in prompts:
            results = self._evaluate_batch_chunked(prompt, self._batch_articles, self._batch_labels)
            rewards.append(sum(results) / len(results))
        return rewards


class QLearningBatchedRewardCalculator(BatchedRewardCalculator):
    """
    Specialized for Q-learning agent.
    
    Evaluates all possible next prompts (current prompt + each action)
    returning rewards for all actions.
    """
    
    def get_action_rewards(self, current_prompt: str, action_space) -> List[float]:
        """
        Get rewards for all actions from current state.
        
        Args:
            current_prompt: Current prompt string
            action_space: ActionSpace instance with .actions list
            
        Returns:
            List of rewards, one per action (same order as action_space.actions)
        """
        # Generate all next prompts
        next_prompts = []
        for action_func in action_space.actions:
            new_prompt = action_func(current_prompt)
            # Validate (same as RLAgent.step)
            REQUIRED = ["World", "Sports", "Business", "Sci/Tech"]
            if not all(cat in new_prompt for cat in REQUIRED):
                next_prompts.append(current_prompt)  # Invalid action -> stay
            elif new_prompt == current_prompt:
                next_prompts.append(current_prompt)  # No change -> stay
            else:
                next_prompts.append(new_prompt)
        
        # Evaluate all prompts
        return self.get_rewards_for_prompts(next_prompts)