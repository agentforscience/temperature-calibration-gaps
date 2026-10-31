"""Failure-mode breakdown by temperature (pooled over main models) -> results/analysis/error_analysis.csv
  code : share of samples that are passing / syntax errors / valid-but-failing / truncated at max_tokens
  qa   : share correct / wrong-short / wrong-long (rambling >8 words) / empty, non-ASCII share
  story: non-ASCII character share, truncated at max_tokens, code-like tokens (underscores, braces)
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS

rows = []
for m, cfg in MODELS.items():
    f = ROOT / f"results/scores/{m}.samples.parquet"
    if cfg.get("control") or not f.exists():
        continue
    s = pd.read_parquet(f)
    s["nonascii"] = [sum(ord(c) > 127 for c in t) / max(1, len(t)) for t in s.text]
    s["trunc"] = s.finish == "length"
    s["nwords"] = s.text.str.split().str.len()
    for T, g in s.groupby("T"):
        c, q, st = g[g.kind == "code"], g[g.kind == "qa"], g[g.kind == "story"]
        rows.append(dict(model=m, T=T,
                         code_pass=c.passed.astype(float).mean(), code_syntax_err=1 - c.syntax_ok.astype(float).mean(),
                         code_valid_fail=(c.syntax_ok.astype(bool) & ~c.passed.astype(bool)).mean(), code_trunc=c.trunc.mean(),
                         qa_correct=q.correct.astype(float).mean(),
                         qa_wrong_short=((~q.correct.astype(bool)) & (q.nwords <= 8)).mean(),
                         qa_wrong_long=((~q.correct.astype(bool)) & (q.nwords > 8)).mean(), qa_nonascii=q.nonascii.mean(),
                         story_nonascii=st.nonascii.mean(), story_trunc=st.trunc.mean(),
                         story_codechars=np.mean([len(re.findall(r"[_{}\[\]<>=/\\]", t)) / max(1, len(t)) for t in st.text])))
df = pd.DataFrame(rows); df.to_csv(ROOT / "results/analysis/error_analysis.csv", index=False)
print(df.drop(columns="model").groupby("T").mean().round(3).to_string())
