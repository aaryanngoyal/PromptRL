import random

from harness import PromptHarness
from agent.actionSpace import ActionSpace
from agent.reward import RewardCalculator
from agent.state import StateRepresentation

class RLAgent:

    def __init__(self, action_space, state_rep, reward_calc):

        self.action_space = action_space
        self.state_rep = state_rep
        self.reward_calc = reward_calc

        self.learning_rate = 0.1
        self.discount_order = 0.9
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995

        self.num_actions = len(action_space.actions)
        self.episode_history = []

    def choose_action(self, prompt):

        state_key = self.state_rep.state_key(prompt)
        q_value = self.state_rep.get_q_value(prompt)

        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)

        base_action = max(q_value, key=q_value.get)
        return base_action

    def step(self, prompt, action_index):

        action_func = self.action_space.actions[action_index]
        new_prompt = action_func(prompt)

        reward = self.reward_calc.get_stable_reward(new_prompt)

        self.update_q_table(prompt, action_index, reward, new_prompt)

        self.exploration_rate *= self.exploration_decay

        return new_prompt, reward

    def update_q_table(self, prompt, action_index, reward, new_prompt):

        state_key = self.state_rep.state_key(prompt)
        q_value = self.state_rep.get_q_value(prompt)

        current_q = q_value[action_index]

        next_q_value = self.state_rep.get_q_value(new_prompt)
        max_future_q = max(next_q_value.values() if next_q_value else 0)

        new_q = current_q + self.learning_rate * (
            reward + self.discount_order * max_future_q - current_q
        )

        q_value[action_index] = new_q
        self.state_rep.q_table[state_key] = q_value

    def train(self, initial_prompt, episodes=100):

        best_prompt = initial_prompt
        best_reward = -float("inf")

        for episode in range(episodes):
            current_prompt = initial_prompt

            for step in range(5):
                action = self.choose_action(current_prompt)
                current_prompt, reward = self.step(current_prompt, action)

                if reward > best_reward:
                    best_reward = reward
                    best_prompt = current_prompt

                self.episode_history.append({
                    'episode': episode,
                    'step': step,
                    'action': action,
                    'reward': reward,
                    'prompt': current_prompt
                })

            print(f"Episode {episode}: Best Reward = {best_reward:.3f}")

        return best_prompt, best_reward


agentharness = PromptHarness()  
action_space = ActionSpace()
state_rep = StateRepresentation(action_space)
reward_calc = RewardCalculator(agentharness, evals=3)

# Create agent
agent = RLAgent(action_space, state_rep, reward_calc)

# Initial prompt
initial_prompt = "Apple Inc. shares surged 5% after strong earnings report... Classify the following news article into a category:"

# Train the agent
best_prompt, best_reward = agent.train(initial_prompt, episodes=50)

print(f"\nBest prompt found: {best_prompt}")
print(f"Best accuracy: {best_reward:.3f}")