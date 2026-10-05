import ctypes, pathlib, os, sys, time, statistics

LIB = str(next(pathlib.Path("build").rglob("libeval_core.dylib")).resolve())
lib = ctypes.CDLL(LIB)
lib.eval_core_last_error.restype  = ctypes.c_char_p
lib.eval_core_load_model.restype  = ctypes.c_void_p
lib.eval_core_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.eval_core_score_batch.restype  = ctypes.c_int
lib.eval_core_score_batch.argtypes = [ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
lib.eval_core_free.argtypes = [ctypes.c_void_p]

MODEL = os.environ["MODEL"]

# A pool of 32 AG News-style articles. Labels are irrelevant here; we're
# only timing. Use whatever ground truth you have.
POOL = [
    ("The UN Security Council met today to discuss the ongoing crisis in the region.", 0),
    ("The Lakers defeated the Celtics 112-108 in a thrilling overtime finish.",       1),
    ("Apple reported quarterly earnings that beat analyst expectations.",             2),
    ("NASA launched a new satellite to study solar flares in unprecedented detail.", 3),
]
print(f"POOL has {len(POOL)} entries")

PROMPT = ("Classify the following news article into exactly one of these "
        "categories: World, Sports, Business, Sci/Tech. "
        "Respond with only the category name and nothing else.")
NAMES = ["World", "Sports", "Business", "Sci/Tech"]

h = lib.eval_core_load_model(MODEL.encode(), 512, -1, 8)
if not h:
    sys.exit(lib.eval_core_last_error().decode())

def run(n, repeats=20):
    # Cycle the pool so n can exceed len(POOL) without IndexError.
    # If POOL has 4 entries and n=16, each entry is used 4 times.
    items    = [POOL[i % len(POOL)] for i in range(n)]
    articles = [a for a, _ in items]
    golds    = [g for _, g in items]

    prompts = (ctypes.c_char_p * n)(*[PROMPT.encode()] * n)
    arts    = (ctypes.c_char_p * n)(*[a.encode() for a in articles])
    labels  = (ctypes.c_int32 * n)()

    # warm-up (fills labels for the debug print below)
    assert lib.eval_core_score_batch(h, prompts, arts, n, labels) == 0

    # --- debug: one pred/gold line per sequence ---
    NAMES = ["World", "Sports", "Business", "Sci/Tech"]
    for i in range(n):
        a = articles[i]
        print(f"  pred={NAMES[labels[i]]:<9} gold={NAMES[golds[i]]:<9} {a[:50]}")

    # --- timing loop ---
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rc = lib.eval_core_score_batch(h, prompts, arts, n, labels)
        t1 = time.perf_counter()
        assert rc == 0, lib.eval_core_last_error().decode()
        times.append((t1 - t0) * 1000.0)

    correct = sum(1 for i in range(n) if labels[i] == golds[i])
    return statistics.median(times), correct, n

print(f"{'n':>3}  {'median ms':>10}  {'ms/sample':>10}  {'speedup':>8}  {'accuracy':>9}")
baseline_ms_per_sample = None
for n in (1, 4, 8, 16):
    med, correct, _ = run(n)
    per = med / n
    if baseline_ms_per_sample is None:
        baseline_ms_per_sample = per
    speedup = baseline_ms_per_sample / per
    print(f"{n:>3}  {med:>10.2f}  {per:>10.2f}  {speedup:>7.2f}x  {correct:>3}/{n}")

lib.eval_core_free(h)
