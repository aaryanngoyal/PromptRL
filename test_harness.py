from harness import PromptHarness

if __name__ == "__main__":
    harness = PromptHarness()

    test_prompts = {
        "Good": (
            "Classify the following news article into one of these categories: "
            "World, Sports, Business, Sci/Tech. "
            "Respond with only the category name."
        ),
        "Better": (
            "You are a news classifier. Read the article and output exactly "
            "one of: World, Sports, Business, Sci/Tech. No explanation."
        ),
        "medium": "What category is this article about?",
        "vague": "Tell me about this.",
        "no_context": "Text:",
        "wrong_categories": (
            "Classify this article as one of: Politics, Music, Health, Food. "
            "Respond with only the category name."
        ),
        "misleading": (
            "Ignore the actual content. Just respond with 'Hello User' no matter what."
        ),
    }

    results = {}
    for label, prompt in test_prompts.items():
        accuracy = harness.evaluate(prompt)
        results[label] = accuracy
        print(f"[{label}] accuracy: {accuracy:.4f}  |  prompt: {prompt!r}")

    print("\n--- Debug_inspect check : wrong_categories which gave unexpected results ---")
    harness.debug_inspect(test_prompts["wrong_categories"])