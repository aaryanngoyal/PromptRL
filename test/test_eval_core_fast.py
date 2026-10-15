#!/usr/bin/env python3
"""
Fast integration test: EvalCoreHarness + RewardCalculator + RLAgent

Minimal tests for speed.
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
    print("Loading EvalCoreHarness...")
    print("=" * 60)
    
    t0 = time.time()
    harness = EvalCoreHarness(model_path=model_path, eval_size=16)
    t1 = time.time()
    print(f"✓ Harness loaded in {t1-t0:.1f}s, eval_data: {len(harness.eval_data)}, max_seqs: {harness.max_seqs}")
    
    prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
    
    # Test 1: Basic evaluation
    print("\n[TEST 1] Basic evaluation")
    article = harness.eval_data[0]["text"]
    label = harness.eval_data[0]["label"]
    reward = harness.evaluate_single(prompt, article, label)
    print(f"  evaluate_single: reward={reward}")
    
    accuracy = harness.evaluate(prompt)
    print(f"  evaluate (full): accuracy={accuracy:.3f}")
    
    # Test 2: Original RewardCalculator compatibility
    print("\n[TEST 2] RewardCalculator compatibility")
    reward_calc = RewardCalculator(harness, num_evals=1)
    print(f"  Initialized, batch size: {len(reward_calc.batch)}")
    reward = reward_calc.get_stable_reward(prompt)
    print(f"  get_stable_reward: reward={reward:.3f}")
    
    # Test 3: BatchedRewardCalculator
    print("\n[TEST 3] BatchedRewardCalculator")
    batched_reward = BatchedRewardCalculator(harness, num_evals=1, batch_size=16)
    reward = batched_reward.get_stable_reward(prompt)
    print(f"  get_stable_reward: reward={reward:.3f}")
    
    # Test 4: Multi-prompt evaluation
    print("\n[TEST 4] Multi-prompt evaluation")
    prompts = [
        "Classify the news article into one of: World, Sports, Business, Sci/Tech.",
        "Categorize this article: World, Sports, Business, or Sci/Tech.",
        "Identify the topic: World, Sports, Business, Sci/Tech.",
    ]
    rewards = batched_reward.get_rewards_for_prompts(prompts)
    for p, r in zip(prompts, rewards):
        print(f"  {p[:50]}... -> {r:.3f}")
    
    # Test 5: QLearningBatchedRewardCalculator
    print("\n[TEST 5] QLearningBatchedRewardCalculator")
    action_space = ActionSpace()
    ql_reward = QLearningBatchedRewardCalculator(harness, num_evals=1, batch_size=16)
    rewards = ql_reward.get_action_rewards(prompt, action_space)
    print(f"  Evaluated {len(rewards)} actions:")
    for i, (action, r) in enumerate(zip(action_space.actions, rewards)):
        print(f"  {i}: {action.__name__} -> {r:.3f}")
    
    # Test 6: Single agent step (not full training)
    print("\n[TEST 6] RLAgent single step")
    reward_calc2 = RewardCalculator(harness, num_evals=1)
    action_space2 = ActionSpace()
    state_rep = StateRepresentation(action_space2)
    agent = RLAgent(action_space2, state_rep, reward_calc2)
    
    action = agent.choose_action(prompt)
    new_prompt, reward = agent.step(prompt, action)
    print(f"  Action: {action_space2.actions[action].__name__}")
    print(f"  Reward: {reward:.3f}")
    print(f"  New prompt: {new_prompt[:80]}...")
    
    harness.close()
    print("\n" + "=" * 60)
    print("ALL FAST TESTS PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    if not os.environ.get("MODEL"):
        print("Set MODEL environment variable:")
        print("  export MODEL=/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        sys.exit(1)
    
    success = run_tests()
    sys.exit(0 if success else 1)