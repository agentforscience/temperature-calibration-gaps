"""Build data/prompts.jsonl: 100 WritingPrompts (story), 300 TriviaQA (qa), 164 HumanEval (code). Seed 42."""
import json, re, random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "datasets"
rng = random.Random(42)
rows = []

# --- open-ended: WritingPrompts test split; strip "[ WP ]"-style tags, keep short single-sentence-ish prompts
wp = pd.read_parquet(D / "writingprompts_test.parquet")
col = "prompt" if "prompt" in wp.columns else wp.columns[0]
cands = []
for p in wp[col].astype(str).unique():
    p = re.sub(r"^\s*(\[\s*\w+\s*\]\s*)+", "", p).replace("<newline>", " ").strip()
    if 60 <= len(p) <= 250 and not re.search(r"\[|\]|http", p):
        cands.append(p)
cands = sorted(set(cands))
for i, p in enumerate(rng.sample(cands, 100)):
    rows.append(dict(pid=f"story{i:03d}", kind="story", source=p,
                     prompt=f"Write a short story (about 120 words) based on this prompt:\n\n{p}\n\n"
                            "Output only the story, with no title or commentary."))

# --- factual QA: TriviaQA rc.nocontext validation (closed-book)
tq = pd.read_parquet(D / "triviaqa_rc_nocontext_validation.parquet")
idx = sorted(rng.sample(range(len(tq)), 300))
for i, j in enumerate(idx):
    r = tq.iloc[j]
    ans = r["answer"]
    aliases = sorted(set(list(ans["normalized_aliases"]) + [ans["normalized_value"]]))
    rows.append(dict(pid=f"qa{i:03d}", kind="qa", source=r["question"], answers=aliases,
                     prompt="Answer the following trivia question with only the answer (a few words), "
                            f"no explanation.\n\nQuestion: {r['question']}\nAnswer:"))

# --- code: HumanEval (all 164)
he = pd.read_parquet(D / "humaneval_test.parquet")
for i, r in he.iterrows():
    rows.append(dict(pid=f"code{i:03d}", kind="code", task_id=r["task_id"], entry_point=r["entry_point"],
                     code_prompt=r["prompt"], test=r["test"],
                     prompt="Complete the following Python function. Return the complete function (including the "
                            "imports and signature) in a single ```python code block, with no explanation.\n\n"
                            f"```python\n{r['prompt']}```"))

(ROOT / "data").mkdir(exist_ok=True)
with open(ROOT / "data/prompts.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print({k: sum(r["kind"] == k for r in rows) for k in ("story", "qa", "code")})
for k in ("story", "qa", "code"):
    print(next(r for r in rows if r["kind"] == k)["prompt"][:400], "\n---")
