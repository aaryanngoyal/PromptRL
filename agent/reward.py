class RewardCalculator:

    """
    Reward Calculation
    """

    def __init__(self, prompt_harness, num_evals=1):
        self.prompt_harness = prompt_harness
        self.num_evals = num_evals

        import random
        rng = random.Random(42)
        by_class = {0: [], 1: [], 2: [], 3: []}
        for i, ex in enumerate(prompt_harness.eval_data):
            by_class[ex["label"]].append(i)

        batch_indices = []
        for _, idxs in by_class.items():
            batch_indices += rng.sample(idxs, 8)

        self.batch = prompt_harness.eval_data.select(batch_indices)

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