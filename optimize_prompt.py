"""
Optimize a user prompt using Q-learning + batched eval_core inference.
Usage: python3 optimize_prompt.py "Classify the article into World, Sports, Business, Sci/Tech"
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from agent.eval_core_harness import EvalCoreHarness
from agent.reward_batched import BatchedRewardCalculator
from agent.rlagent import RLAgent
from agent.actionSpace import ActionSpace
from agent.state import StateRepresentation


def optimize_user_prompt(user_prompt: str, episodes: int = 30, max_steps: int = 5) -> str:
    """Run Q-learning to optimize the given prompt."""
    
    # 1. Setup batched inference (CPU, fast)
    model_path = os.environ.get("MODEL")
    if not model_path:
        raise ValueError("Set MODEL env var to your .gguf file")
    
    harness = EvalCoreHarness(
        model_path=model_path,
        eval_size=100,      # Evaluation dataset size
        n_gpu_layers=0,     # CPU (GPU needs smaller model)
        n_ctx=512,          # Context per sequence
        n_threads=8
    )
    
    # 2. Batched reward calculator (5-10x speedup)
    reward_calc = BatchedRewardCalculator(
        harness, 
        num_evals=1,        # Evaluations per prompt (1 = fast)
        batch_size=16       # Max examples per batch
    )
    
    # 3. Q-learning agent
    action_space = ActionSpace()
    state_rep = StateRepresentation(action_space)
    agent = RLAgent(action_space, state_rep, reward_calc)
    
    # 4. Ensure prompt has required categories
    REQUIRED = ["World", "Sports", "Business", "Sci/Tech"]
    for cat in REQUIRED:
        if cat not in user_prompt:
            user_prompt += f" {cat}"
    
    print(f"Initial prompt: {user_prompt}")
    print(f"Running {episodes} episodes, {max_steps} steps each...")
    
    # 5. Train
    best_prompt, best_reward = agent.train(
        user_prompt, 
        episodes=episodes, 
        max_step=max_steps,
        patience=10
    )
    
    harness.close()
    
    print(f"\n✓ Best reward: {best_reward:.3f}")
    print(f"✓ Optimized prompt: {best_prompt}")
    
    return best_prompt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 optimize_prompt.py \"your prompt here\"")
        print('Example: python3 optimize_prompt.py "Classify news: World, Sports, Business, Sci/Tech"')
        sys.exit(1)
    
    user_prompt = " ".join(sys.argv[1:])
    optimize_user_prompt(user_prompt)