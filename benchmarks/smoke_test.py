import ctypes, pathlib, sys, os

LIB = str(next(pathlib.Path("build").rglob("libeval_core.so")).resolve())
lib = ctypes.CDLL(LIB)

lib.eval_core_last_error.restype = ctypes.c_char_p

lib.eval_core_load_model.restype  = ctypes.c_void_p
lib.eval_core_load_model.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
]

lib.eval_core_score_batch.restype  = ctypes.c_int
lib.eval_core_score_batch.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int32),
]

lib.eval_core_free.argtypes = [ctypes.c_void_p]

MODEL = os.environ.get("MODEL") or sys.exit("set MODEL=/path/to/model.gguf")

print(f"loading {MODEL} ...")
h = lib.eval_core_load_model(MODEL.encode(), 2048, 0, 8)
if not h:
    print("load failed:", lib.eval_core_last_error().decode())
    sys.exit(1)
print("model loaded, handle =", hex(h))

PROMPT  = "Classify the following news article into one of: World, Sports, Business, Sci/Tech."
articles = [
    "The UN Security Council met today to discuss the ongoing crisis in the region.",
    "The Lakers defeated the Celtics 112-108 in a thrilling overtime finish.",
    "Apple reported quarterly earnings that beat analyst expectations.",
    "NASA launched a new satellite to study solar flares in unprecedented detail.",
    "India won against New Zealand in cricket",
    "Elon musk is the richest person on earth"
]

n = len(articles)
prompts  = (ctypes.c_char_p * n)(*[PROMPT.encode()] * n)
arts     = (ctypes.c_char_p * n)(*[a.encode() for a in articles])
labels   = (ctypes.c_int32 * n)()

rc = lib.eval_core_score_batch(h, prompts, arts, n, labels)
if rc != 0:
    print(f"score_batch failed: rc={rc} err={lib.eval_core_last_error().decode()}")
        # Correctness check: three consecutive calls must give identical results.
    print("\nrepeat-call check:")
    first = list(labels)
    for run in range(2):
        rc = lib.eval_core_score_batch(h, prompts, arts, n, labels)
        assert rc == 0, lib.eval_core_last_error().decode()
        assert list(labels) == first, (
            f"run {run+2} disagrees: {list(labels)} vs {first} — KV not reset?"
        )
    print("  three consecutive calls agree:", first)
    lib.eval_core_free(h)
    sys.exit(1)

NAMES = ["World", "Sports", "Business", "Sci/Tech"]
for i, a in enumerate(articles):
    print(f"  {NAMES[labels[i]]:<9}  {a[:60]}")

lib.eval_core_free(h)
print("done")