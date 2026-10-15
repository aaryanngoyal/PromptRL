#!/usr/bin/env python3
"""
Integration test: EvalCoreHarness + RewardCalculator + RLAgent

Tests that the new harness works with existing agent components.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.eval_core_harness import EvalCoreHarness
from agent.actionSpace import ActionSpace
from agent.state import StateRepresentation
from agent.reward import RewardCalculator
from agent.reward_batched import BatchedRewardCalculator, QLearningBatchedRewardCalculator
from agent.rlagent import RLAgent


def test_harness_basic():
    """Test EvalCoreHarness basic functionality."""
    print("=" * 60)
    print("TEST 1: EvalCoreHarness basic evaluation")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=10)
        print(f"✓ Harness loaded, eval_data size: {len(harness.eval_data)}")
        
        # Test single evaluation
        prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        article = "The Lakers defeated the Celtics 112-108 in overtime."
        label = 1  # Sports
        
        reward = harness.evaluate_single(prompt, article, label)
        print(f"✓ evaluate_single works: reward={reward}")
        
        # Test batch evaluation
        accuracy = harness.evaluate(prompt)
        print(f"✓ evaluate (full batch) works: accuracy={accuracy:.3f}")
        
        # Test debug
        print("\nDebug inspect:")
        harness.debug_inspect(prompt, n=3)
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_calculator_compatibility():
    """Test that RewardCalculator works with EvalCoreHarness."""
    print("\n" + "=" * 60)
    print("TEST 2: RewardCalculator compatibility")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=20)
        
        # Original RewardCalculator (should work unchanged)
        reward_calc = RewardCalculator(harness, num_evals=1)
        print(f"✓ RewardCalculator initialized, batch size: {len(reward_calc.batch)}")
        
        prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        reward = reward_calc.get_stable_reward(prompt)
        print(f"✓ get_stable_reward works: reward={reward:.3f}")
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_reward_calculator():
    """Test the new optimized BatchedRewardCalculator."""
    print("\n" + "=" * 60)
    print("TEST 3: BatchedRewardCalculator (optimized)")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=20)
        
        # New batched reward calculator
        batched_reward = BatchedRewardCalculator(harness, num_evals=1)
        print(f"✓ BatchedRewardCalculator initialized")
        
        prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        reward = batched_reward.get_stable_reward(prompt)
        print(f"✓ get_stable_reward works: reward={reward:.3f}")
        
        # Test multi-prompt evaluation
        prompts = [
            "Classify the news article into one of: World, Sports, Business, Sci/Tech.",
            "Categorize this article: World, Sports, Business, or Sci/Tech.",
            "Identify the topic: World, Sports, Business, Sci/Tech.",
        ]
        rewards = batched_reward.get_rewards_for_prompts(prompts)
        for p, r in zip(prompts, rewards):
            print(f"  Prompt: {p[:50]}... -> reward={r:.3f}")
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qlearning_batched_reward():
    """Test QLearningBatchedRewardCalculator with ActionSpace."""
    print("\n" + "=" * 60)
    print("TEST 4: QLearningBatchedRewardCalculator + ActionSpace")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=20)
        action_space = ActionSpace()
        ql_reward = QLearningBatchedRewardCalculator(harness)
        
        prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        rewards = ql_reward.get_action_rewards(prompt, action_space)
        
        print(f"✓ Evaluated {len(rewards)} actions in one batch:")
        for i, (action, reward) in enumerate(zip(action_space.actions, rewards)):
            print(f"  Action {i} ({action.__name__}): reward={reward:.3f}")
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_agent():
    """Test full RLAgent with EvalCoreHarness."""
    print("\n" + "=" * 60)
    print("TEST 5: Full RLAgent with EvalCoreHarness")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=20)
        
        # Use original RewardCalculator (works with EvalCoreHarness)
        reward_calc = RewardCalculator(harness, num_evals=1)
        action_space = ActionSpace()
        state_rep = StateRepresentation(action_space)
        
        agent = RLAgent(action_space, state_rep, reward_calc)
        
        initial_prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        
        print("Running 3 episodes (short test)...")
        best_prompt, best_reward = agent.train(
            initial_prompt, episodes=3, max_step=3, patience=10
        )
        
        print(f"\n✓ Training complete!")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Best prompt: {best_prompt[:80]}...")
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_agent_batched_rewards():
    """Test full RLAgent with BatchedRewardCalculator (optimized)."""
    print("\n" + "=" * 60)
    print("TEST 6: RLAgent with BatchedRewardCalculator (optimized)")
    print("=" * 60)
    
    model_path = os.environ.get("MODEL")
    if not model_path:
        print("SKIP: MODEL env var not set")
        return False
    
    try:
        harness = EvalCoreHarness(model_path=model_path, eval_size=20)
        
        # Use optimized batched reward calculator
        reward_calc = BatchedRewardCalculator(harness, num_evals=1)
        action_space = ActionSpace()
        state_rep = StateRepresentation(action_space)
        
        agent = RLAgent(action_space, state_rep, reward_calc)
        
        initial_prompt = "Classify the news article into one of: World, Sports, Business, Sci/Tech."
        
        print("Running 3 episodes with batched rewards...")
        best_prompt, best_reward = agent.train(
            initial_prompt, episodes=3, max_step=3, patience=10
        )
        
        print(f"\n✓ Training complete!")
        print(f"Best reward: {best_reward:.3f}")
        print(f"Best prompt: {best_prompt[:80]}...")
        
        harness.close()
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("EvalCore Integration Test Suite")
    print("=" * 60)
    
    # Check model
    if not os.environ.get("MODEL"):
        print("\n⚠️  Set MODEL environment variable to run tests:")
        print("   export MODEL=/path/to/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        print("\nRunning without model will test imports only.\n")
    
    results = []
    
    # Test imports first
    try:
        from agent.eval_core_harness import EvalCoreHarness
        from agent.reward_batched import BatchedRewardCalculator, QLearningBatchedRewardCalculator
        print("✓ All imports successful")
        results.append(("Imports", True))
    except Exception as e:
        print(f"✗ Import failed: {e}")
        results.append(("Imports", False))
    
    if os.environ.get("MODEL"):
        results.append(("Basic harness", test_harness_basic()))
        results.append(("RewardCalculator compat", test_reward_calculator_compatibility()))
        results.append(("BatchedRewardCalculator", test_batched_reward_calculator()))
        results.append(("QLearningBatchedReward", test_qlearning_batched_reward()))
        results.append(("Full agent (original reward)", test_full_agent()))
        results.append(("Full agent (batched reward)", test_full_agent_batched_rewards()))
    else:
        print("Skipping model-dependent tests (MODEL not set)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(p for _, p in results)
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)