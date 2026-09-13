from harness import PromptHarness
from agent.actionSpace import ActionSpace
from agent.reward import RewardCalculator
from agent.state import StateRepresentation
from agent.rlagent import RLAgent

agentharness = PromptHarness()  
action_space = ActionSpace()
state_rep = StateRepresentation(action_space)
reward_calc = RewardCalculator(agentharness, num_evals=1)

# Create agent
agent = RLAgent(action_space, state_rep, reward_calc)

# Initial prompt
initial_prompt = "Read the news article and select its topic. Options: World, Sports, Business, Sci/Tech. Output only the topic."

# Train the agent
best_prompt, best_reward = agent.train(initial_prompt, episodes=30, max_step=4)

print(f"\nBest prompt found: {best_prompt}")
print(f"Best accuracy: {best_reward:.3f}")

full_accuracy = agentharness.evaluate(best_prompt)
print(f"Batch accuracy  (training):  {best_reward:.4f}")
print(f"Full accuracy   (eval set):  {full_accuracy:.4f}")

# Also evaluate the ORIGINAL prompt for comparison
baseline_accuracy = agentharness.evaluate(initial_prompt)
print(f"Baseline (initial prompt):   {baseline_accuracy:.4f}")
print(f"Improvement: {full_accuracy - baseline_accuracy:+.4f}")