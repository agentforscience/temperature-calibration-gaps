"""Automatic scoring of generations (everything except the LLM judge).

For each model's results/gens/<model>.jsonl writes results/scores/<model>.samples.parquet (one row per sample) and
results/scores/<model>.groups.parquet (one row per (prompt, T): diversity metrics over the n samples).

  qa    : correct (TriviaQA normalized-alias match), normalized answer string
  code  : HumanEval execution pass/fail (sandboxed subprocess), syntax validity, normalized-AST hash
  story : lexical validity (share of word tokens found in a WritingPrompts-train vocabulary), distinct-2,
          mpnet embeddings -> mean pairwise cosine distance among the samples of a (prompt, T) group

Usage: python src/score.py MODEL [MODEL ...]   (embeddings use the GPU in CUDA_VISIBLE_DEVICES if any, else CPU)
"""
import ast, hashlib, json, os, re, resource, string, subprocess, sys, tempfile, itertools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GENS, OUT = ROOT / "results/gens", ROOT / "results/scores"
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
EXPECTED_ROWS = len(PROMPTS) * 11
EXEC_CACHE = ROOT / "results/scores/exec_cache.jsonl"


# ----------------------------------------------------------------------------- QA
def norm_answer(s):
    """TriviaQA-style normalization: lowercase, drop punctuation, articles and extra whitespace."""
    s = s.lower().replace("_", " ")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_correct(pred, aliases):
    p = norm_answer(pred.strip().split("\n")[0])
    if not p:
        return False
    for a in aliases:
        if not a:
            continue
        if p == a:
            return True
        # alias contained as whole words in a short answer (e.g. "the beatles band" vs "beatles")
        if re.search(rf"(^| ){re.escape(a)}( |$)", p) and len(p.split()) <= len(a.split()) + 4:
            return True
    return False


# ----------------------------------------------------------------------------- code
def extract_code(text, item):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m:
        code = max(m, key=len)
    else:
        m = re.search(r"```(?:python|py)?\s*\n(.*)", text, re.S)       # unterminated block (hit max_tokens)
        code = m.group(1) if m else text
    if f"def {item['entry_point']}" not in code:                      # body-only answer -> prepend prompt
        code = item["code_prompt"] + code
    return code


def ast_key(code):
    try:
        return hashlib.sha1(ast.dump(ast.parse(code)).encode()).hexdigest()[:16], True
    except Exception:
        return hashlib.sha1(code.encode()).hexdigest()[:16], False


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def run_program(code, item):
    prog = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "prog.py"
        f.write_text(prog)
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=d, capture_output=True, timeout=15,
                               preexec_fn=_limits, env={"PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def load_exec_cache():
    c = {}
    if EXEC_CACHE.exists():
        for l in open(EXEC_CACHE):
            r = json.loads(l); c[r["k"]] = r["ok"]
    return c


# ----------------------------------------------------------------------------- stories
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_VOCAB = None


def vocab():
    """Words seen >=3 times in 20k WritingPrompts *train* stories (disjoint from the test prompts we use)."""
    global _VOCAB
    if _VOCAB is None:
        vf = ROOT / "results/scores/wp_vocab.json"
        if vf.exists():
            _VOCAB = set(json.load(open(vf)))
        else:
            import glob
            fs = sorted(glob.glob("/data/transformers/hub/datasets--euclaise--writingprompts/snapshots/*/data/train-*"))
            df = pd.read_parquet(fs[0], columns=["story"]).head(20000)
            c = Counter(w.lower() for s in df["story"] for w in WORD.findall(s))
            _VOCAB = {w for w, k in c.items() if k >= 3}
            json.dump(sorted(_VOCAB), open(vf, "w"))
    return _VOCAB


def lexical_validity(text):
    ws = WORD.findall(text)
    toks = text.split()
    if not toks:
        return 0.0
    v = vocab()
    return sum(w.lower() in v for w in ws) / len(toks)     # non-word tokens (garbage, CJK) count as invalid


def ngrams(text, n=2):
    t = text.lower().split()
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


def distinct_n(texts, n=2):
    g = [x for t in texts for x in ngrams(t, n)]
    return len(set(g)) / len(g) if g else 0.0


_EMB = None


def embed(texts):
    global _EMB
    if _EMB is None:
        os.environ.setdefault("HF_HOME", "/data/transformers"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                   device="cuda" if torch.cuda.is_available() else "cpu")
    return _EMB.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


# ----------------------------------------------------------------------------- main
def score_model(model):
    rows = [json.loads(l) for l in open(GENS / f"{model}.jsonl")]
    keys = {(r["pid"], r["T"]) for r in rows}
    if len(keys) < EXPECTED_ROWS:
        print(f"{model}: only {len(keys)}/{EXPECTED_ROWS} (pid,T) rows - scoring partial file")
    seen, uniq = set(), []
    for r in rows:                                   # dedupe in case of resumed writes
        if (r["pid"], r["T"]) not in seen:
            seen.add((r["pid"], r["T"])); uniq.append(r)
    samples = []
    for r in uniq:
        for si, s in enumerate(r["samples"]):
            samples.append(dict(model=model, pid=r["pid"], kind=r["kind"], T=r["T"], sidx=si, text=s["text"],
                                finish=s["finish"], n_out=s["n_out"], mean_lp=s["mean_lp"]))
    df = pd.DataFrame(samples)

    # QA
    qa = df.kind == "qa"
    df.loc[qa, "correct"] = [qa_correct(t, PROMPTS[p]["answers"]) for t, p in zip(df.text[qa], df.pid[qa])]
    df.loc[qa, "norm"] = [norm_answer(t.strip().split("\n")[0]) for t in df.text[qa]]

    # code
    cd = df.kind == "code"
    codes = [extract_code(t, PROMPTS[p]) for t, p in zip(df.text[cd], df.pid[cd])]
    ak = [ast_key(c) for c in codes]
    df.loc[cd, "ast"] = [a for a, _ in ak]
    df.loc[cd, "syntax_ok"] = [ok for _, ok in ak]
    cache = load_exec_cache()
    todo = {}
    for c, p, (a, ok) in zip(codes, df.pid[cd], ak):
        k = hashlib.sha1((p + "\0" + c).encode()).hexdigest()
        if k not in cache and k not in todo:
            todo[k] = (c, p, ok)
    print(f"{model}: executing {len(todo)} unique programs ({len(cache)} cached)", flush=True)
    with ThreadPoolExecutor(12) as ex, open(EXEC_CACHE, "a") as fc:
        fut = {k: ex.submit(run_program, c, PROMPTS[p]) if ok else None for k, (c, p, ok) in todo.items()}
        for k, f in fut.items():
            ok = bool(f.result()) if f is not None else False
            cache[k] = ok
            fc.write(json.dumps(dict(k=k, ok=ok)) + "\n")
    df.loc[cd, "passed"] = [cache[hashlib.sha1((p + "\0" + c).encode()).hexdigest()] for c, p in zip(codes, df.pid[cd])]

    # stories
    st = df.kind == "story"
    df.loc[st, "lex_valid"] = [lexical_validity(t) for t in df.text[st]]
    df.loc[st, "n_words"] = [len(t.split()) for t in df.text[st]]
    embs = embed(df.text[st].tolist())
    emb_map = dict(zip(df.index[st], embs))

    # per-(pid,T) group diversity
    groups = []
    for (pid, kind, T), g in df.groupby(["pid", "kind", "T"], sort=False):
        d = dict(model=model, pid=pid, kind=kind, T=T, n=len(g))
        if kind == "story":
            E = np.stack([emb_map[i] for i in g.index])
            if len(g) > 1:
                sims = [float(E[i] @ E[j]) for i, j in itertools.combinations(range(len(g)), 2)]
                d["sem_div"] = 1 - float(np.mean(sims))
            else:
                d["sem_div"] = 0.0
            d["distinct2"] = distinct_n(g.text.tolist(), 2)
            d["distinct2_within"] = float(np.mean([distinct_n([t], 2) for t in g.text]))
        elif kind == "qa":
            d["acc"] = float(g.correct.mean())
            d["cover"] = float(g.correct.any())
            d["n_distinct"] = g.norm.nunique()
        else:
            d["pass1"] = float(g.passed.mean())
            d["passk"] = float(g.passed.any())
            d["n_distinct"] = g.ast.nunique()
            d["syntax_ok"] = float(g.syntax_ok.mean())
        groups.append(d)
    df.drop(columns=[]).to_parquet(OUT / f"{model}.samples.parquet")
    pd.DataFrame(groups).to_parquet(OUT / f"{model}.groups.parquet")
    print(f"{model}: scored {len(df)} samples, {len(groups)} groups", flush=True)


if __name__ == "__main__":
    for m in sys.argv[1:]:
        score_model(m)
