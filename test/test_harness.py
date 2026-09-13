from harness import PromptHarness
from agent.reward import RewardCalculator

harness = PromptHarness()
initial_prompt = "Read the news article and select its topic. Options: World, Sports, Business, Sci/Tech. Output only the topic."

# 1. Does the initial prompt work on the OLD (bad) batch?
old_batch = harness.eval_data.select(range(15))
correct = sum(harness.evaluate_single(initial_prompt, ex["text"], ex["label"]) 
              for ex in old_batch)
print(f"Old batch accuracy: {correct/len(old_batch):.3f}")

# 2. Does it work on the NEW (stratified) batch?
reward_calc = RewardCalculator(harness, num_evals=3)
correct = sum(harness.evaluate_single(initial_prompt, ex["text"], ex["label"]) 
              for ex in reward_calc.batch)
print(f"New batch accuracy: {correct/len(reward_calc.batch):.3f}")
print(f"New batch labels: {[ex['label'] for ex in reward_calc.batch]}")

# 3. Is the LLM even returning sensible output?
example = reward_calc.batch[0]
full_prompt = harness._format_prompt(initial_prompt, example["text"])
print("PROMPT:", full_prompt[:300])
output = harness.llm(full_prompt, max_tokens=20, temperature=0.0, stop=["</s>", "[INST]"])
print("RAW OUTPUT:", repr(output["choices"][0]["text"]))
print("PARSED:", harness._parse_response(output["choices"][0]["text"]))
print("TRUE LABEL:", example["label"])