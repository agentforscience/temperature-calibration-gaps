"""Analysis of the temperature grid: curves, useful ranges, hypothesis tests, figures.

Inputs : results/scores/<model>.{groups,samples}.parquet, results/judge/<model>.jsonl
Outputs: results/analysis/*.csv|json|md, figures/*.png
Usage  : python src/analyze.py
"""
import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models import MODELS

SC, JD = ROOT / "results/scores", ROOT / "results/judge"
OUT, FIG = ROOT / "results/analysis", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
TEMPS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]
B = 2000
RNG = np.random.default_rng(42)
MAIN = [m for m in MODELS if not MODELS[m].get("control")]
SIZE_LADDERS = {"qwen2.5": ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b"],
                "llama3": ["llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]}
MATCHED = {"7-9B (AWQ)": ["qwen2.5-7b", "llama3.1-8b", "mistral-7b", "gemma2-9b"],
           "1-1.5B (bf16)": ["qwen2.5-1.5b", "llama3.2-1b"], "3B (AWQ)": ["qwen2.5-3b", "llama3.2-3b"]}


def load():
    G, S, J = [], [], []
    for m in MODELS:
        if not (SC / f"{m}.groups.parquet").exists():
            continue
        g = pd.read_parquet(SC / f"{m}.groups.parquet")
        if len(g) < 564 * 11:
            print(f"skip {m}: incomplete ({len(g)} groups)"); continue
        G.append(g); S.append(pd.read_parquet(SC / f"{m}.samples.parquet").drop(columns=["text"]))
        if (JD / f"{m}.jsonl").exists():
            j = pd.DataFrame([json.loads(l) for l in open(JD / f"{m}.jsonl")])
            j = j.dropna(subset=["score"]).drop_duplicates(["pid", "T", "sidx"], keep="last")
            j["model"] = m; J.append(j)
    G, S = pd.concat(G, ignore_index=True), pd.concat(S, ignore_index=True)
    J = pd.concat(J, ignore_index=True) if J else pd.DataFrame(columns=["model", "pid", "T", "sidx", "score"])
    return G, S, J


def item_matrix(df, value, model):
    """items x temperatures matrix of an item-level metric for one model (rows aligned across temperatures)."""
    d = df[df.model == model].pivot_table(index="pid", columns="T", values=value, aggfunc="mean")
    return d.reindex(columns=TEMPS)


def boot_curves(M):
    """Bootstrap (over items) of column means: returns B x T array."""
    X = M.values
    idx = RNG.integers(0, len(X), size=(B, len(X)))
    return np.nanmean(X[idx], axis=1)


def useful_upper(curve, frac=0.9):
    """Largest T (linear interpolation, searching upward from the argmax) with quality >= frac * max."""
    c = np.asarray(curve, float); T = np.array(TEMPS)
    k = int(np.nanargmax(c)); thr = frac * c[k]
    for i in range(k, len(c) - 1):
        if c[i + 1] < thr:
            return T[i] + (T[i + 1] - T[i]) * (c[i] - thr) / (c[i] - c[i + 1])
    return T[-1]


def useful_lower(curve, frac=0.9):
    c = np.asarray(curve, float); T = np.array(TEMPS)
    k = int(np.nanargmax(c)); thr = frac * c[k]
    for i in range(k, 0, -1):
        if c[i - 1] < thr:
            return T[i - 1] + (T[i] - T[i - 1]) * (thr - c[i - 1]) / (c[i] - c[i - 1])
    return T[0]


def ci(a):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


def signflip_p(d, n=10000):
    """Two-sided sign-flip permutation test of mean(d) = 0 for paired item-level differences."""
    d = np.asarray(d, float); d = d[~np.isnan(d)]
    if len(d) == 0 or np.all(d == 0):
        return 1.0
    obs = abs(d.mean())
    s = RNG.choice([-1, 1], size=(n, len(d)))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); adj = np.empty_like(ps, dtype=float); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(ps) - r) * ps[i]); adj[i] = min(1, run)
    return adj


def main():
    G, S, J = load()
    models = [m for m in MODELS if m in set(G.model)]
    print("models:", models)
    story = G[G.kind == "story"].copy(); qa = G[G.kind == "qa"].copy(); code = G[G.kind == "code"].copy()
    J["coherent"] = (J.score >= 5).astype(float)
    jg = J.groupby(["model", "pid", "T"]).agg(judge=("score", "mean"), coherent=("coherent", "mean")).reset_index()
    st_s = S[S.kind == "story"].groupby(["model", "pid", "T"]).agg(lex_valid=("lex_valid", "mean"),
                                                                     mean_lp=("mean_lp", "mean"),
                                                                     hit_max=("finish", lambda x: (x == "length").mean())
                                                                     ).reset_index()
    story = story.merge(jg, on=["model", "pid", "T"], how="left").merge(st_s, on=["model", "pid", "T"], how="left")

    # distinct-2 pools the samples of a group; at T=0 there is only one sample, so the value is not comparable
    story.loc[story["T"] == 0, ["distinct2"]] = np.nan

    # ------------------------------------------------------------------ 1. curves table (mean + 95% CI)
    metrics = {"story": (story, ["sem_div", "distinct2", "judge", "coherent", "lex_valid", "mean_lp", "hit_max"]),
               "qa": (qa, ["acc", "cover", "n_distinct"]), "code": (code, ["pass1", "passk", "n_distinct", "syntax_ok"])}
    rows, boots = [], {}
    for kind, (df, cols) in metrics.items():
        for m in models:
            for c in cols:
                M = item_matrix(df, c, m)
                bc = boot_curves(M); boots[(kind, m, c)] = bc
                mean = M.mean().values
                for ti, T in enumerate(TEMPS):
                    lo, hi = ci(bc[:, ti])
                    rows.append(dict(task=kind, model=m, metric=c, T=T, mean=mean[ti], lo=lo, hi=hi))
    curves = pd.DataFrame(rows)
    curves.to_csv(OUT / "curves.csv", index=False)

    def cv(kind, m, c):
        d = curves[(curves.task == kind) & (curves.model == m) & (curves.metric == c)]
        return d.set_index("T")["mean"].reindex(TEMPS).values

    # ------------------------------------------------------------------ 2. H1: open-ended plateau + coherence drop
    h1 = []
    iT = {T: i for i, T in enumerate(TEMPS)}
    for m in models:
        r = dict(model=m)
        for c in ["sem_div", "distinct2"]:
            bc = boots[("story", m, c)]
            lo_s = (bc[:, iT[1.2]] - bc[:, iT[0.2]]) / 1.0
            hi_s = (bc[:, iT[2.0]] - bc[:, iT[1.2]]) / 0.8
            ratio = hi_s / lo_s
            cvv = cv("story", m, c)
            r[f"{c}_slope_0.2-1.2"] = (cvv[iT[1.2]] - cvv[iT[0.2]]) / 1.0
            r[f"{c}_slope_1.2-2.0"] = (cvv[iT[2.0]] - cvv[iT[1.2]]) / 0.8
            r[f"{c}_ratio"] = r[f"{c}_slope_1.2-2.0"] / r[f"{c}_slope_0.2-1.2"]
            r[f"{c}_ratio_ci"] = ci(ratio)
            r[f"{c}_argmax_T"] = TEMPS[int(np.nanargmax(cvv))]
        j = cv("story", m, "judge"); co = cv("story", m, "coherent"); lv = cv("story", m, "lex_valid")
        k = int(np.nanargmax(j)); r["judge_peak_T"] = TEMPS[k]; r["judge_peak"] = j[k]
        for T in [1.2, 1.4, 1.6, 2.0]:
            bj = boots[("story", m, "judge")]
            drop = 1 - bj[:, iT[T]] / bj[:, k]
            r[f"judge_drop@{T}"] = 1 - j[iT[T]] / j[k]; r[f"judge_drop@{T}_ci"] = ci(drop)
            r[f"coherent@{T}"] = co[iT[T]]
            bco = boots[("story", m, "coherent")]
            r[f"coherent_drop@{T}"] = 1 - co[iT[T]] / np.nanmax(co)
            r[f"coherent_drop@{T}_ci"] = ci(1 - bco[:, iT[T]] / np.nanmax(bco, axis=1))
        r["coherent_peak"] = np.nanmax(co); r["lexvalid@2.0"] = lv[iT[2.0]]
        h1.append(r)
    h1 = pd.DataFrame(h1); h1.to_csv(OUT / "h1_openended.csv", index=False)

    # ------------------------------------------------------------------ 3. H2: QA
    h2 = []
    for m in models:
        A = item_matrix(qa, "acc", m); C = item_matrix(qa, "cover", m)
        base = A[0.0]
        ps, rr = [], []
        for T in TEMPS[1:]:
            d = (A[T] - base).values
            ps.append(signflip_p(d)); rr.append((T, float(np.nanmean(d))))
        adj = holm(ps)
        r = {"model": m, "acc_greedy": float(base.mean()),
             "acc_T<=0.7": float(A[[0.0, 0.2, 0.4, 0.6, 0.7]].values.mean()),
             "acc_T0.8-1.0": float(A[[0.8, 1.0]].values.mean()),
             "acc_T>=1.2": float(A[[1.2, 1.4, 1.6, 2.0]].values.mean())}
        for (T, dm), p, pa in zip(rr, ps, adj):
            r[f"dacc@{T}"] = dm; r[f"p_holm@{T}"] = pa
        # paired test: mean acc over T in {0.8,1.0} vs mean over T<=0.7 (item level)
        d = (A[[0.8, 1.0]].mean(1) - A[[0.0, 0.2, 0.4, 0.6, 0.7]].mean(1)).values
        r["d_hi_vs_lo"] = float(d.mean()); r["p_hi_vs_lo"] = signflip_p(d)
        for T in [0.7, 1.0, 1.2]:
            r[f"cover@{T}"] = float(C[T].mean()); r[f"cover_gain@{T}"] = float((C[T] - base).mean())
        r["T_u"] = useful_upper(A.mean().values)
        h2.append(r)
    h2 = pd.DataFrame(h2); h2["p_hi_vs_lo_holm"] = holm(h2.p_hi_vs_lo.values)
    h2.to_csv(OUT / "h2_qa.csv", index=False)

    # ------------------------------------------------------------------ 4. H3: code
    h3 = []
    for m in models:
        P = item_matrix(code, "pass1", m); K = item_matrix(code, "passk", m)
        c = P.mean().values
        r = dict(model=m, pass1_greedy=c[0], pass1_max=np.nanmax(c), argmax_T=TEMPS[int(np.nanargmax(c))],
                 T_lo=useful_lower(c), T_u=useful_upper(c), T50=useful_upper(c, 0.5))
        for T in [0.2, 0.6, 0.8, 1.0, 1.2]:
            r[f"pass1@{T}"] = c[iT[T]]
        d = (P[1.0] - P[0.6]).values
        r["rel_drop_0.6to1.0"] = 1 - c[iT[1.0]] / c[iT[0.6]] if c[iT[0.6]] > 0 else np.nan
        r["p_0.6_vs_1.0"] = signflip_p(d)
        d2 = (P[0.8] - P[0.6]).values
        r["rel_drop_0.6to0.8"] = 1 - c[iT[0.8]] / c[iT[0.6]] if c[iT[0.6]] > 0 else np.nan
        r["p_0.6_vs_0.8"] = signflip_p(d2)
        for T in [0.6, 0.8, 1.0]:
            r[f"passk@{T}"] = float(K[T].mean())
        h3.append(r)
    h3 = pd.DataFrame(h3)
    for c in ["p_0.6_vs_1.0", "p_0.6_vs_0.8"]:
        h3[c + "_holm"] = holm(h3[c].values)
    h3.to_csv(OUT / "h3_code.csv", index=False)

    # ------------------------------------------------------------------ 5. H4: useful ranges vs size (bootstrap T_u)
    qual = {"story": ("coherent", story),   # judge rarely scores <3 -> use share of coherent (>=5) outputs
            "qa": ("acc", qa), "code": ("pass1", code)}
    tu_rows = []
    for kind, (c, _) in qual.items():
        for m in models:
            bc = boots[(kind, m, c)]
            tus = np.array([useful_upper(b) for b in bc[:500]])
            t50 = np.array([useful_upper(b, 0.5) for b in bc[:500]])
            tl = np.array([useful_lower(b) for b in bc[:500]])
            cvv = cv(kind, m, c)
            tu_rows.append(dict(task=kind, model=m, family=MODELS[m]["family"], params=MODELS[m]["params"],
                                T_lo=useful_lower(cvv), T_u=useful_upper(cvv), T_u_ci=ci(tus),
                                width=useful_upper(cvv) - useful_lower(cvv), width_ci=ci(tus - tl),
                                T50=useful_upper(cvv, 0.5), T50_ci=ci(t50), peak=float(np.nanmax(cvv))))
    tu = pd.DataFrame(tu_rows); tu.to_csv(OUT / "useful_ranges.csv", index=False)
    h4 = []
    for kind in qual:
        d = tu[(tu.task == kind) & tu.model.isin(MAIN)]
        for name, sel in [("all", d)] + [(f, d[d.model.isin(l)]) for f, l in SIZE_LADDERS.items()]:
            if len(sel) >= 3:
                for col in ["T_u", "width", "T50"]:
                    rho, p = stats.spearmanr(np.log(sel.params), sel[col])
                    h4.append(dict(task=kind, subset=name, metric=col, n=len(sel), rho=rho, p=p))
    h4 = pd.DataFrame(h4); h4.to_csv(OUT / "h4_size.csv", index=False)
    # largest vs smallest model of each ladder: bootstrap CI of the difference in T_u (more direct than rho, n small)
    h4b = []
    for kind, (c, _) in qual.items():
        for fam, l in SIZE_LADDERS.items():
            l = [m for m in l if m in models]
            if len(l) < 2:
                continue
            a, b = l[0], l[-1]
            d = np.array([useful_upper(x) - useful_upper(y) for x, y in zip(boots[(kind, b, c)][:500], boots[(kind, a, c)][:500])])
            h4b.append(dict(task=kind, family=fam, small=a, large=b, dT_u=useful_upper(cv(kind, b, c)) - useful_upper(cv(kind, a, c)),
                            dT_u_ci=ci(d)))
    h4b = pd.DataFrame(h4b); h4b.to_csv(OUT / "h4_ladder_contrast.csv", index=False)

    # ------------------------------------------------------------------ 6. H5: family differences at matched scale
    h5 = []
    floors = {"noise floor (seed)": ["qwen2.5-0.5b", "qwen2.5-0.5b-seed2"], "noise floor (precision)": ["qwen2.5-7b", "qwen2.5-7b-bf16"]}
    for grp, ms in {**MATCHED, **floors}.items():
        ms = [m for m in ms if m in models]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                for kind, c in [("story", "sem_div"), ("story", "distinct2"), ("story", "judge"), ("qa", "acc"),
                                ("qa", "n_distinct"), ("code", "pass1"), ("code", "n_distinct")]:
                    ba, bb = boots[(kind, a, c)], boots[(kind, b, c)]
                    # shape: curve normalised to its own max (removes level differences)
                    na = ba / np.nanmax(ba, axis=1, keepdims=True); nb = bb / np.nanmax(bb, axis=1, keepdims=True)
                    dshape = np.nanmean(np.abs(na - nb), axis=1)              # mean abs gap between shapes
                    # null for shape gap: bootstrap both curves around the pooled shape is complex; we instead
                    # report the T at which the normalised curves differ most + per-T CI excluding 0
                    diff = na - nb
                    lo = np.nanpercentile(diff, 2.5, axis=0); hi = np.nanpercentile(diff, 97.5, axis=0)
                    sig_T = [T for T, l, h in zip(TEMPS, lo, hi) if l > 0 or h < 0]
                    ta = np.array([useful_upper(x, 0.5) for x in ba[:500]]); tb = np.array([useful_upper(x, 0.5) for x in bb[:500]])
                    pa = np.array([TEMPS[int(np.nanargmax(x))] for x in ba[:500]])
                    pb = np.array([TEMPS[int(np.nanargmax(x))] for x in bb[:500]])
                    h5.append(dict(group=grp, a=a, b=b, task=kind, metric=c, shape_gap=float(np.nanmean(dshape)),
                                   shape_gap_ci=ci(dshape), sig_T=sig_T, n_sig_T=len(sig_T),
                                   dT50=float(np.mean(ta - tb)), dT50_ci=ci(ta - tb),
                                   argmaxT_a=TEMPS[int(np.nanargmax(cv(kind, a, c)))],
                                   argmaxT_b=TEMPS[int(np.nanargmax(cv(kind, b, c)))],
                                   d_argmax_ci=ci(pa - pb)))
    h5 = pd.DataFrame(h5); h5.to_csv(OUT / "h5_family.csv", index=False)

    # ------------------------------------------------------------------ 7. precision control (bf16 vs AWQ)
    if "qwen2.5-7b-bf16" in models and "qwen2.5-7b" in models:
        ctl = curves[curves.model.isin(["qwen2.5-7b", "qwen2.5-7b-bf16"]) &
                     curves.metric.isin(["sem_div", "judge", "acc", "pass1"])]
        ctl.pivot_table(index=["task", "metric", "T"], columns="model", values="mean").to_csv(OUT / "precision_control.csv")

    # ------------------------------------------------------------------ 8. diversity restricted to coherent outputs
    # pairwise distance between samples 0 and 1 is not stored; use groups where BOTH judged samples are coherent
    both = J[J.sidx.isin([0, 1])].pivot_table(index=["model", "pid", "T"], columns="sidx", values="coherent")
    both = both.dropna().reset_index(); both["both_coh"] = (both[0] + both[1]) == 2
    sc = story.merge(both[["model", "pid", "T", "both_coh"]], on=["model", "pid", "T"], how="left")
    cd = sc[sc.both_coh == True].groupby(["model", "T"]).sem_div.mean().unstack()
    cd.to_csv(OUT / "sem_div_coherent_groups.csv")

    make_figures(curves, models, tu)
    print(h1.round(3).to_string()); print(h2.round(3).to_string()); print(h3.round(3).to_string())
    print(h4.round(3).to_string()); print(h4b.round(3).to_string()); print(tu.round(2).to_string())


def make_figures(curves, models, tu):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fam_col = {"qwen2.5": "tab:blue", "llama3": "tab:orange", "mistral": "tab:green", "gemma2": "tab:red"}
    ls = {}
    for f, l in SIZE_LADDERS.items():
        for k, m in enumerate(l):
            ls[m] = 0.35 + 0.65 * (k + 1) / len(l)
    panels = [("story", "sem_div", "Stories: semantic diversity\n(mean pairwise cosine distance)"),
              ("story", "distinct2", "Stories: distinct-2 (pooled samples)"),
              ("story", "judge", "Stories: coherence (judge 1-9)"),
              ("story", "lex_valid", "Stories: lexical validity"),
              ("qa", "acc", "TriviaQA: accuracy per sample"),
              ("qa", "n_distinct", "TriviaQA: distinct answers (of 4)"),
              ("code", "pass1", "HumanEval: pass@1"),
              ("code", "n_distinct", "HumanEval: distinct programs (of 4)")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (kind, c, title) in zip(axes.flat, panels):
        for m in models:
            d = curves[(curves.task == kind) & (curves.model == m) & (curves.metric == c)].sort_values("T")
            if d.empty:
                continue
            ctl = MODELS[m].get("control")
            ax.plot(d["T"], d["mean"], marker="o", ms=3, color="gray" if ctl else fam_col[MODELS[m]["family"]],
                    alpha=ls.get(m, 1.0), ls=":" if ctl else "-", label=m)
            ax.fill_between(d["T"], d["lo"], d["hi"], alpha=0.08, color=fam_col[MODELS[m]["family"]])
        ax.set_title(title, fontsize=10); ax.set_xlabel("temperature"); ax.grid(alpha=0.3)
        for x in (0.7, 1.2):
            ax.axvline(x, color="k", lw=0.6, ls="--", alpha=0.4)
    axes[0, 0].legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / "curves_all.png", dpi=130); plt.close()

    # diversity-quality frontier for stories
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for m in models:
        if MODELS[m].get("control"):
            continue
        a = curves[(curves.task == "story") & (curves.model == m)]
        x = a[a.metric == "sem_div"].sort_values("T"); y = a[a.metric == "judge"].sort_values("T")
        ax[0].plot(x["mean"].values, y["mean"].values, marker="o", ms=3, color=fam_col[MODELS[m]["family"]],
                   alpha=ls.get(m, 1.0), label=m)
        for T in (0.7, 1.2, 2.0):
            i = TEMPS.index(T); ax[0].annotate(str(T), (x["mean"].values[i], y["mean"].values[i]), fontsize=6)
        x2 = a[a.metric == "distinct2"].sort_values("T")
        ax[1].plot(x2["mean"].values, y["mean"].values, marker="o", ms=3, color=fam_col[MODELS[m]["family"]],
                   alpha=ls.get(m, 1.0))
    ax[0].set_xlabel("semantic diversity"); ax[0].set_ylabel("coherence (judge)"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("distinct-2"); ax[1].set_ylabel("coherence (judge)")
    ax[0].set_title("Stories: diversity vs coherence as T rises (labels = T)"); ax[1].set_title("lexical diversity vs coherence")
    for a_ in ax: a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "story_frontier.png", dpi=130); plt.close()

    # useful ranges vs size
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, kind in zip(axes, ["story", "qa", "code"]):
        d = tu[(tu.task == kind) & tu.model.isin(MAIN)].sort_values("params")
        for _, r in d.iterrows():
            col = fam_col[r.family]
            ax.plot([r.params, r.params], [r.T_lo, r.T_u], color=col, lw=4, alpha=0.6)
            ax.errorbar(r.params, r.T_u, yerr=[[r.T_u - r.T_u_ci[0]], [r.T_u_ci[1] - r.T_u]], fmt="o", color=col,
                        ms=4, capsize=2)
            ax.annotate(r.model, (r.params, r.T_u), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("parameters (B)"); ax.set_ylabel("temperature")
        ax.set_title(f"{kind}: useful range (quality >= 90% of peak)\nbar = [T_lo, T_u], dot = T_u with 95% CI", fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(FIG / "useful_ranges.png", dpi=130); plt.close()

    # matched-scale normalised diversity curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (kind, c, title) in zip(axes, [("story", "sem_div", "story semantic diversity"),
                                           ("qa", "n_distinct", "QA distinct answers"),
                                           ("code", "n_distinct", "code distinct programs")]):
        for m in MATCHED["7-9B (AWQ)"]:
            d = curves[(curves.task == kind) & (curves.model == m) & (curves.metric == c)].sort_values("T")
            if d.empty:
                continue
            ax.plot(d["T"], d["mean"], marker="o", ms=3, color=fam_col[MODELS[m]["family"]], label=m)
            ax.fill_between(d["T"], d["lo"], d["hi"], alpha=0.15, color=fam_col[MODELS[m]["family"]])
        ax.set_title(f"7-9B AWQ models: {title}"); ax.set_xlabel("temperature"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "family_matched_diversity.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
