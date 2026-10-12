"""
RLAgent: Q-Learning as Guided Search for Prompt Optimization on AG News Classification
"""

import random

REQUIRED = ["World", "Sports", "Business", "Sci/Tech"]

class RLAgent:

    def __init__(self, action_space, state_rep, reward_calc):

        self.action_space = action_space
        self.state_rep = state_rep
        self.reward_calc = reward_calc

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_order = 0.9
        self.exploration_rate = 0.3
        self.exploration_decay = 0.995

        self.num_actions = len(action_space.actions)

    def choose_action(self, prompt):

        """
        Action Selection from Action Space
        """

        q_value = self.state_rep.get_q_value(prompt)

        # Choose random action
        if random.random() < self.exploration_rate:
            return random.randint(0, self.num_actions - 1)

        # Choose best action
        if not q_value:
            return random.randint(0, self.num_actions - 1)
        base_action = max(q_value, key=q_value.get)
        return base_action


    def step(self, prompt, action_index):

        """
        Execute action
        Get Reward
        Update Q-Table
        """

        # Execute action
        action_func = self.action_space.actions[action_index]
        new_prompt = action_func(prompt)

        if not isinstance(new_prompt, str):
            raise TypeError(
                f"Action '{action_func.__name__}' returned "
                f"{type(new_prompt).__name__}, expected str. "
                f"Value: {new_prompt!r}"
            )

        if not all(cat in new_prompt for cat in REQUIRED):
            return prompt, -0.05 

        if new_prompt == prompt:
            return prompt, -0.01

        # Reward calculation
        reward = self.reward_calc.get_stable_reward(new_prompt)

        # Update Q table
        self.update_q_table(prompt, action_index, reward, new_prompt)

        self.exploration_rate *= self.exploration_decay

        return new_prompt, reward

    def update_q_table(self, prompt, action_index, reward, new_prompt):

        """
        Q-learning update
        """

        state_key = self.state_rep.state_key(prompt)
        q_value = self.state_rep.get_q_value(prompt)

        # Current Q value
        current_q = q_value[action_index]

        # Future Q value
        next_q_value = self.state_rep.get_q_value(new_prompt) or {}
        max_future_q = max(next_q_value.values(), default=0.0)

        # Q-learning formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_order * max_future_q - current_q
        )

        # Update Q-table
        q_value[action_index] = new_q
        self.state_rep.q_table[state_key] = q_value

    def train(self, initial_prompt, episodes=100, max_step=5, patience=15):

        """
        Train the agent for multiple episodes
        """

        best_prompt = initial_prompt
        best_reward = -float("inf")
        no_improve = 0

        for episode in range(episodes):
            current_prompt = initial_prompt
            episode_best_reward = -float("inf")
            episode_best_prompt = current_prompt

            for _ in range(max_step):
                action = self.choose_action(current_prompt)
                current_prompt, reward = self.step(current_prompt, action)
                if reward > episode_best_reward:
                    episode_best_reward = reward
                    episode_best_prompt = current_prompt


            if episode_best_reward > best_reward + 1e-6:
                best_reward = episode_best_reward
                best_prompt = episode_best_prompt
                no_improve = 0

            else:
                no_improve += 1

            self.exploration_rate = max(0.05, self.exploration_rate * 0.95)

            print(f"Ep {episode:>3} | best={best_reward:.3f} | "
                  f"eps={self.exploration_rate:.3f} | no_improve={no_improve} | "
                  f"q_states={len(self.state_rep.q_table)}")

            if no_improve >= patience:
                print(f"Early stopping at: {episode}")
                break

        return best_prompt, best_reward
