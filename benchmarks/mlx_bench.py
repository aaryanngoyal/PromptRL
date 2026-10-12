import time, statistics
from mlx_lm import load, batch_generate
from mlx_lm.sample_utils import make_sampler

MODEL = "/Users/aravsrivastava/miniProject/PromptRL/models/Mistral-7B-v0.2-4bit"

print("loading...")
model, tokenizer, *_ = load(MODEL)
print("loaded.")

INSTRUCTION = ("Classify the following news article into exactly one of: "
               "World, Sports, Business, Sci/Tech. "
               "Answer with one word only, no explanation, no punctuation.")

ARTICLES = [
    "The UN Security Council met today to discuss the ongoing crisis in the region.",
    "The Lakers defeated the Celtics 112-108 in a thrilling overtime finish.",
    "Apple reported quarterly earnings that beat analyst expectations.",
    "NASA launched a new satellite to study solar flares in unprecedented detail.",
]

def make_token_lists(n):
    """Return a list of token-ID lists, one per prompt."""
    arts = [ARTICLES[i % len(ARTICLES)] for i in range(n)]
    return [
        tokenizer.encode(f"[INST] {INSTRUCTION}\n\nText: {a} [/INST] ")
        for a in arts
    ]

def run(n, repeats=20):
    prompts = make_token_lists(n)
    sampler = make_sampler(temp=0.0)

    # warm-up
    batch_generate(model, tokenizer, prompts,
                   max_tokens=1, sampler=sampler, verbose=False)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        batch_generate(model, tokenizer, prompts,
                       max_tokens=1, sampler=sampler, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return statistics.median(times)

print(f"{'n':>3}  {'median ms':>10}  {'ms/sample':>10}")
base = None
for n in (1, 4, 8, 16):
    med = run(n)
    per = med / n
    if base is None: base = per
    print(f"{n:>3}  {med:>10.2f}  {per:>10.2f}   speedup {base/per:.2f}x")