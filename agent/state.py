class StateRepresentation:

    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space

    def state_key(self, prompt):

        """
        Using prompt text as state key
        """

        return prompt

    def get_q_value(self, prompt):

        """
        Get Q-value for all actions for given state
        """

        state_key = self.state_key(prompt)
        if state_key not in self.q_table:

            # Initialize all actions with 0.0
            self.q_table[state_key] = {action: 0.0 for action in range(len(self.action_space.actions))}

        return self.q_table[state_key]

    def update_q_value(self, prompt, action_index, new_q_value):

        """
        Update Q-value for specific action in a state
        """

        state_key = self.state_key(prompt)
        self.q_table[state_key][action_index] = new_q_value