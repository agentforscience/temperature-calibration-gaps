"""LLM-judge coherence/fluency of stories with the host's Qwen3-8B vLLM server (:8011).

Score = probability-weighted digit 1-9 from the first output token's top-20 logprobs (thinking disabled).
The server has been seen to return corrupted logprobs, so every judgment is validated (sampled token == argmax of
top logprobs, digit mass >= 0.5); after 3 failed tries we fall back to the greedy text digit.
Judged samples: sidx 0 at T=0 and sidx 0,1 at T>0 (2 of the 4 samples).
Usage: python src/judge.py MODEL [MODEL ...]  -> results/judge/<model>.jsonl
"""
import json, math, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/judge"
OUT.mkdir(parents=True, exist_ok=True)
URL = "http://127.0.0.1:8011/v1/chat/completions"
PROMPTS = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
THREADS = 48

RUBRIC = """You are evaluating a short story written in response to a writing prompt.

Writing prompt: {prompt}

Story:
<<<
{story}
>>>

Rate the story's COHERENCE and FLUENCY on a 1-9 scale:
9 = fluent, grammatical, logically coherent story that clearly responds to the prompt
7 = mostly fluent and coherent, minor slips
5 = understandable but with noticeable errors, odd word choices, or lapses in logic
3 = largely broken: many ungrammatical or nonsensical passages
1 = incoherent or gibberish (e.g. random word salad, mixed languages, broken tokens)
Judge the whole text, including how it degrades. Do not judge creativity or style. The story may be cut off at the end because of a length limit; do not penalize
the cut-off itself. Reply with a single digit (1-9) and nothing else."""


def judge_one(prompt, story):
    msg = RUBRIC.format(prompt=prompt, story=story[:2500])
    body = dict(model="Qwen3-8B", messages=[{"role": "user", "content": msg}], max_tokens=1, temperature=0,
                logprobs=True, top_logprobs=20, chat_template_kwargs={"enable_thinking": False})
    for attempt in range(3):
        try:
            r = requests.post(URL, json=body, timeout=120).json()
            c = r["choices"][0]
            tok = c["message"]["content"]
            top = c["logprobs"]["content"][0]["top_logprobs"]
            best = max(top, key=lambda x: x["logprob"])
            probs = {t["token"].strip(): math.exp(t["logprob"]) for t in top}
            digits = {int(k): v for k, v in probs.items() if k in [str(i) for i in range(1, 10)]}
            mass = sum(digits.values())
            if best["token"] == tok and mass >= 0.5:
                return dict(score=sum(k * v for k, v in digits.items()) / mass, argmax=int(max(digits, key=digits.get)),
                            mass=mass, method="logprob")
        except Exception as e:
            time.sleep(2 + attempt * 3)
    # fallback: greedy text
    try:
        body2 = dict(body, max_tokens=4); body2.pop("logprobs"); body2.pop("top_logprobs")
        t = requests.post(URL, json=body2, timeout=120).json()["choices"][0]["message"]["content"]
        d = next(int(ch) for ch in t if ch in "123456789")
        return dict(score=float(d), argmax=d, mass=None, method="text")
    except Exception:
        return dict(score=None, argmax=None, mass=None, method="failed")


def judge_model(model):
    out = OUT / f"{model}.jsonl"
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r["score"] is not None:
                done.add((r["pid"], r["T"], r["sidx"]))
    items = []
    for l in open(ROOT / f"results/gens/{model}.jsonl"):
        r = json.loads(l)
        if r["kind"] != "story":
            continue
        for si, s in enumerate(r["samples"][:2]):
            if (r["pid"], r["T"], si) not in done:
                items.append((r["pid"], r["T"], si, s["text"]))
    print(f"{model}: {len(done)} judged, {len(items)} to judge", flush=True)
    lock = threading.Lock()
    t0 = time.time()
    with open(out, "a") as f, ThreadPoolExecutor(THREADS) as ex:
        def work(it):
            pid, T, si, text = it
            res = judge_one(PROMPTS[pid]["source"], text)
            with lock:
                f.write(json.dumps(dict(pid=pid, T=T, sidx=si, **res)) + "\n"); f.flush()
        for i, _ in enumerate(ex.map(work, items)):
            if i % 500 == 499:
                print(f"{model}: {i + 1}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    print(f"{model}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        judge_model(m)
