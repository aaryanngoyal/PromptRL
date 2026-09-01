import random
from harness import PromptHarness

prompt_harness = PromptHarness()

class RewardCalculator:

    """
    Reward Calculation
    """

    def __init__(self, prompt_harness, evals=3):
        self.prompt_harness = prompt_harness
        self.evals = evals

    def get_stable_reward(self, prompt):

        """
        Evalute prompt multiple times then return average accuracy
        Reduce noise
        """

        accuracies = []

        for _ in range(self.evals):

            accuracy = self.prompt_harness.evaluate(prompt)
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        return avg_accuracy
