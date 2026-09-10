class RewardCalculator:

    """
    Reward Calculation
    """

    def __init__(self, prompt_harness, batch_size=5, num_evals=3):
        self.prompt_harness = prompt_harness
        self.num_evals = num_evals
        self.batch = prompt_harness.eval_data.select(range(batch_size))

    def get_stable_reward(self, prompt):

        """
        Evalute prompt multiple times then return average accuracy
        Reduce noise
        """

        total = 0.0

        for example in self.batch:
            for _ in range(self.num_evals):
                total += self.prompt_harness.evaluate_single(
                    prompt, example["text"], example["label"]
                )

        return total / (len(self.batch) * self.num_evals)