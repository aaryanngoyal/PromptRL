#!/usr/bin/env python3
"""
Quick integration test: EvalCoreHarness + RewardCalculator + RLAgent

Reuses model across tests for speed.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.eval_core_harness import EvalCoreHarness
from agent.actionSpace import ActionSpace
from agent.state import StateRepresentation
from agent.reward import RewardCalculator
from agent.reward_batched import BatchedRewardCalculator, QLearningBatchedRewardCalculator
from agent.rlagent import RLAgent


def run_tests():
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    print("=" * 60)
    print("Loading EvalCoreHarness (one-time model load)...")
    print("=" * 60)
    
    t0 = time.time()
    harness = EvalCoreHarness(model_path=model_path, eval_size=16)
    t1 = time.time()
    print(f"✓ Harness loaded in {t1-t0:.1f}s, eval_data: {len(harness.eval_data)}, max_seqs: {harness.max_seqs}")
    
    # Test 1: Basic evaluation
    print("\n" + "=" * 60)
    print("TEST 1: Basic evaluation")
    print("=" * 60)
    prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
    article = harness.eval_data[0]["text"]
    label = harness.eval_data[0]["label"]
    
    reward = harness.evaluate_single(prompt, article, label)
    print(f"✓ evaluate_single: reward={reward}")
    
    accuracy = harness.evaluate(prompt)
    print(f"✓ evaluate (full): accuracy={accuracy:.3f}")
    
    # Test 2: Original RewardCalculator compatibility
    print("\n" + "=" * 60)
    print("TEST 2: RewardCalculator compatibility")
    print("=" * 60)
    reward_calc = RewardCalculator(harness, num_evals=1)
    print(f"✓ RewardCalculator initialized, batch size: {len(reward_calc.batch)}")
    
    reward = reward_calc.get_stable_reward(prompt)
    print(f"✓ get_stable_reward: reward={reward:.3f}")
    
    # Test 3: BatchedRewardCalculator
    print("\n" + "=" * 60)
    print("TEST 3: BatchedRewardCalculator")
    print("=" * 60)
    batched_reward = BatchedRewardCalculator(harness, num_evals=1, batch_size=16)
    reward = batched_reward.get_stable_reward(prompt)
    print(f"✓ get_stable_reward: reward={reward:.3f}")
    
    # Test 4: Multi-prompt evaluation
    print("\n" + "=" * 60)
    print("TEST 4: Multi-prompt evaluation")
    print("=" * 60)
    prompts = [
        "Classify the news article into one of: World, Sports, Business, Sci/Tech.",
        "Categorize this article: World, Sports, Business, or Sci/Tech.",
        "Identify the topic: World, Sports, Business, Sci/Tech.",
    ]
    rewards = batched_reward.get_rewards_for_prompts(prompts)
    for p, r in zip(prompts, rewards):
        print(f"  Prompt: {p[:50]}... -> reward={r:.3f}")
    
    # Test 5: QLearningBatchedRewardCalculator
    print("\n" + "=" * 60)
    print("TEST 5: QLearningBatchedRewardCalculator")
    print("=" * 60)
    action_space = ActionSpace()
    ql_reward = QLearningBatchedRewardCalculator(harness, num_evals=1, batch_size=16)
    rewards = ql_reward.get_action_rewards(prompt, action_space)
    print(f"✓ Evaluated {len(rewards)} actions:")
    for i, (action, r) in enumerate(zip(action_space.actions, rewards)):
        print(f"  Action {i} ({action.__name__}): reward={r:.3f}")
    
    # Test 6: Full agent with original RewardCalculator
    print("\n" + "=" * 60)
    print("TEST 6: RLAgent with RewardCalculator (2 episodes)")
    print("=" * 60)
    reward_calc2 = RewardCalculator(harness, num_evals=1)
    action_space2 = ActionSpace()
    state_rep = StateRepresentation(action_space2)
    agent = RLAgent(action_space2, state_rep, reward_calc2)
    
    best_prompt, best_reward = agent.train(prompt, episodes=2, max_step=2, patience=10)
    print(f"✓ Best reward: {best_reward:.3f}")
    print(f"  Best prompt: {best_prompt[:80]}...")
    
    # Test 7: Full agent with BatchedRewardCalculator
    print("\n" + "=" * 60)
    print("TEST 7: RLAgent with BatchedRewardCalculator (2 episodes)")
    print("=" * 60)
    batched_reward2 = BatchedRewardCalculator(harness, num_evals=1, batch_size=16)
    action_space3 = ActionSpace()
    state_rep2 = StateRepresentation(action_space3)
    agent2 = RLAgent(action_space3, state_rep2, batched_reward2)
    
    best_prompt2, best_reward2 = agent2.train(prompt, episodes=2, max_step=2, patience=10)
    print(f"✓ Best reward: {best_reward2:.3f}")
    print(f"  Best prompt: {best_prompt2[:80]}...")
    
    harness.close()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if not os.environ.get("MODEL"):
        print("Set MODEL environment variable:")
        print("  export MODEL=/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        sys.exit(1)
    
    success = run_tests()
    sys.exit(0 if success else 1)