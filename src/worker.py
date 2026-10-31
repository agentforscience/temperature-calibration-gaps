"""Long-lived in-process vLLM worker for a shared GPU (adapted from earlier workspaces on this host).

Holds GPU memory as 512 MB placeholder tensors and runs models from a queue file one after another IN-PROCESS.
Placeholders are handed to vLLM through PyTorch's caching allocator (empty_cache disabled), so memory never goes
back to the driver between models -> no startup races with other tenants.

Usage: CUDA_VISIBLE_DEVICES=<g> [TAKEOVER_PID=<holder pid> TAKEOVER_GB=40] python src/worker.py QUEUE_FILE MAX_GB
Each model writes results/gens/<model>.jsonl: one line per (prompt, temperature) with n samples.
"""
import os, sys, json, time, gc, fcntl, random, subprocess
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS

SEED = 42
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
N_SAMPLES = 4                      # samples per (prompt, T); T=0 is greedy -> 1 sample
MAX_TOKENS = {"story": 256, "qa": 24, "code": 448}
N_PROMPTS = 564
CHUNK = 512 << 20
LOGS, GENS = ROOT / "logs", ROOT / "results" / "gens"
GENS.mkdir(parents=True, exist_ok=True)
PLACE = []


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if os.environ.get("TAKEOVER_PID"):       # hand-over: wait (imports done) until the previous holder exits
    open(LOGS / f"ready_{os.getpid()}", "w").write("1")
    while os.path.exists(f"/proc/{os.environ['TAKEOVER_PID']}"):
        time.sleep(0.002)
while True:                              # CUDA context needs ~0.5 GB; retry while the GPU is full
    try:
        torch.zeros(1, device="cuda"); break
    except Exception:
        time.sleep(0.5)


def held():
    return len(PLACE) * CHUNK / 2**30


def grab(gb, timeout):
    """Try to hold `gb` GB in total, polling for up to `timeout` s."""
    t = time.time()
    while held() < gb and time.time() - t < timeout:
        try:
            PLACE.append(torch.empty(CHUNK, dtype=torch.uint8, device="cuda"))
        except Exception:
            time.sleep(0.02)
    return held() >= gb


if os.environ.get("TAKEOVER_PID"):
    grab(float(os.environ.get("TAKEOVER_GB", "4.5")), timeout=20)
    log(f"took over {held():.1f} GB")

# ---- hand-off to vLLM: placeholders are held between models; at vLLM's memory measurement enough of them are freed
# into PyTorch's cache for the weights, and the rest is released to the driver only right before the KV-cache
# allocation (see _init_kv). Releasing everything at measurement time lost the memory to other tenants during
# weight loading; handing *all* memory over through the cache failed for the multi-GB KV allocation.
_real_empty_cache = torch.cuda.empty_cache
torch.cuda.empty_cache = lambda: None
try:
    torch.accelerator.empty_cache = lambda: None
except Exception:
    pass
import vllm.v1.worker.gpu_worker as gw
RELEASE = {}


class _Snap(gw.MemorySnapshot):
    """At vLLM's startup measurement, move enough placeholders into PyTorch's cache for the weights and report the
    cached memory as free (so the startup check passes without returning anything to the driver)."""
    def measure(self):
        n = int(RELEASE.get("weights_gb", 0) * 2**30 / CHUNK) + 2
        for _ in range(min(n, len(PLACE))):
            PLACE.pop()
        RELEASE["weights_gb"] = 0
        super().measure()
        self.free_memory += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()


gw.MemorySnapshot = _Snap
try:
    import vllm.utils.mem_utils as mu
    mu.MemorySnapshot = _Snap
except Exception:
    pass
_orig_init_kv = gw.Worker.initialize_from_config      # covers both model-runner implementations (v1 and gpu/ v2)


def _init_kv(self, *a, **k):
    """Release the remaining placeholders to the driver only right before the (large) KV allocation, then refill
    PyTorch's cache with an activation margin. Race window with other tenants: milliseconds instead of minutes."""
    PLACE.clear(); gc.collect(); _real_empty_cache()
    out = _orig_init_kv(self, *a, **k)
    grab(held() + RELEASE.get("margin_gb", 1.5), timeout=5)
    PLACE.clear()                      # freed into PyTorch's cache (empty_cache is disabled) -> activations reuse it
    return out


gw.Worker.initialize_from_config = _init_kv
from vllm import LLM, SamplingParams


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def sampling(kind, t, seed):
    """Pure temperature sampling: top_p=1, top_k off, min_p=0, no repetition penalty."""
    return SamplingParams(temperature=t, top_p=1.0, top_k=-1, min_p=0.0, repetition_penalty=1.0,
                          n=1 if t == 0 else N_SAMPLES, max_tokens=MAX_TOKENS[kind], seed=seed, logprobs=0)


def run_model(key):
    cfg = MODELS[key]
    out = GENS / f"{key}.jsonl"
    prompts = read_jsonl(ROOT / "data/prompts.jsonl")
    pidx = {p["pid"]: i for i, p in enumerate(prompts)}
    done = {(r["pid"], r["T"]) for r in read_jsonl(out)} if out.exists() else set()
    jobs = [(p, ti, t) for p in prompts for ti, t in enumerate(TEMPS) if (p["pid"], t) not in done]
    random.Random(SEED).shuffle(jobs)            # temperatures interleaved in batches; partial runs are uniform
    order = {"qa": 0, "story": 1, "code": 2}
    jobs.sort(key=lambda j: order[j[0]["kind"]])  # stable: shuffled within task
    log(f"{key}: {len(done)} done, {len(jobs)} to do, holding {held():.1f} GB")
    if not jobs:
        return
    total = held()
    margin = cfg.get("margin", 1.5)
    kv = min(6.0, total - cfg["w"] - margin - cfg.get("misc", 0.8))   # misc: CUDA context / non-torch memory
    RELEASE.update(weights_gb=cfg["w"], margin_gb=margin)
    llm = LLM(model=cfg["hf"], dtype="auto", max_model_len=1536, gpu_memory_utilization=min(0.9, (cfg["w"] + 0.3) / 47.4),
              kv_cache_memory_bytes=int(kv * 2**30), enforce_eager=True, max_num_seqs=256,
              max_num_batched_tokens=4096, seed=SEED, trust_remote_code=False, enable_prefix_caching=True,
              generation_config="vllm")
    t0 = time.time()
    with open(out, "a") as f:
        for i in range(0, len(jobs), 600):
            ch = jobs[i:i + 600]
            sps = [sampling(p["kind"], t, cfg.get("seed", SEED) + 1000 * pidx[p["pid"]] + 10 * ti) for p, ti, t in ch]
            outs = llm.chat([[{"role": "user", "content": p["prompt"]}] for p, _, _ in ch], sps, use_tqdm=False)
            for (p, ti, t), o in zip(ch, outs):
                samples = []
                for c in o.outputs:
                    lps = [next(iter(d.values())).logprob for d in (c.logprobs or [])]
                    samples.append(dict(text=c.text, finish=c.finish_reason, n_out=len(c.token_ids),
                                        sum_lp=float(sum(lps)), mean_lp=float(sum(lps) / max(1, len(lps)))))
                f.write(json.dumps(dict(pid=p["pid"], kind=p["kind"], T=t, n_prompt=len(o.prompt_token_ids),
                                        samples=samples), ensure_ascii=False) + "\n")
            f.flush()
            log(f"{key} {i + len(ch)}/{len(jobs)} {time.time() - t0:.0f}s")
    del llm
    gc.collect()


def complete(key):
    """All (prompt, T) rows present? Cheap line count; the output is only ever appended whole lines."""
    f = GENS / f"{key}.jsonl"
    if not f.exists():
        return False
    with open(f, "rb") as h:
        return sum(1 for _ in h) >= N_PROMPTS * len(TEMPS)


def n_fails(key):
    f = LOGS / f"fails_{key}"
    return int(f.read_text()) if f.exists() else 0


def handoff(qfile, max_gb, gb):
    """A second engine in the same process dies with 'CUDA error: invalid argument', so each process runs one model.
    Spawn the successor, wait until its imports are done, then exit; it grabs our memory the moment we are gone."""
    env = dict(os.environ, TAKEOVER_PID=str(os.getpid()), TAKEOVER_GB=str(gb))
    p = subprocess.Popen([sys.executable, sys.argv[0], qfile, str(max_gb)], env=env, cwd=ROOT,
                         stdout=sys.stdout, stderr=sys.stderr)
    t = time.time()
    while not (LOGS / f"ready_{p.pid}").exists() and time.time() - t < 900:
        time.sleep(0.5)
    log(f"handing {gb:.1f} GB to successor {p.pid}")
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def main():
    qfile, max_gb = sys.argv[1], float(sys.argv[2])
    while not (LOGS / "STOP_workers").exists():
        queue = [l.strip() for l in open(qfile) if l.strip() and not l.startswith("#")]
        todo = [k for k in queue if MODELS[k]["need"] <= max_gb and n_fails(k) < 2 and not complete(k)]
        if not todo:
            log("nothing left for this worker; exiting (memory goes back to the host)")
            return
        for key in todo:
            lock = open(GENS / f"{key}.lock", "w")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock.close(); continue                     # another worker has it
            if not grab(MODELS[key]["need"], timeout=30):
                lock.close(); continue
            grab(min(max_gb, MODELS[key]["need"] + 4), timeout=5)   # a bit of extra KV if it's free
            before = held()
            ok = False
            try:
                run_model(key)
                ok = True
            except Exception as e:
                import traceback; traceback.print_exc(); log(f"FAILED {key}: {e}")
                (LOGS / f"fails_{key}").write_text(str(n_fails(key) + 1))
            lock.close()
            log(f"{'done' if ok else 'failed'} {key}; handing over")
            handoff(qfile, max_gb, before)
        time.sleep(20)                                     # everything locked or too big for now
    log("STOP seen; exiting")


if __name__ == "__main__":
    main()
