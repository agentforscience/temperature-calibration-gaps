# Planning: Temperature Calibration Gaps

## Motivation & Novelty Assessment

### Why This Research Matters
Softmax temperature is the single most widely exposed decoding knob, and practitioners pick it by folklore
("0.7 for chat, 0.2 for code, >1 for creativity"). If temperature's effect on *useful* diversity saturates while
quality keeps falling, or if the useful range depends on model size/family, then a single default is miscalibrated
for most deployments. A task × model × temperature map tells practitioners where the knob still does something.

### Gap in Existing Work
No resource-finder output was available for this workspace (no `literature_review.md`, `papers/`, `code/`), so the
gap is stated from prior knowledge: Renze & Guven (2024, "The Effect of Sampling Temperature on Problem Solving in
LLMs") found accuracy on MCQA flat over T∈[0,1] for several models; Peeperkorn et al. (2024, "Is Temperature the
Creativity Parameter of LLMs?") found only weak novelty gains and coherence loss with temperature for one model on
story writing; Holtzman et al. (2020) motivated nucleus sampling by degeneration at high T; Chen et al. (2021,
Codex) reported best pass@1 at low T and best pass@100 at ~0.8. These studies each look at one task type, few models,
and rarely above T=1.0, and none jointly measures diversity *and* quality curves across a size ladder and several
families with matched prompts and decoding settings.

### Our Novel Contribution
One controlled grid: 10 open-weight instruct models (two size ladders: Qwen2.5 0.5–14B, Llama-3 1–8B; plus Mistral-7B
and Gemma-2-9B at ~7–9B) × 11 temperatures (0–2.0, pure temperature sampling, top-p=1) × 3 task types (open-ended
stories, closed-book factual QA, code with execution). For each cell we measure diversity (semantic + lexical) and
quality (LLM-judge coherence + model-independent proxies, QA accuracy, pass@1), and derive per-model "useful
temperature ranges" and curve-shape parameters to test each clause of the hypothesis.

### Experiment Justification
- Exp 1 (stories, WritingPrompts): tests the "diversity plateaus above 1.2 while coherence drops 20–40%" clause.
- Exp 2 (TriviaQA closed-book): tests "above 0.7 accuracy degrades without meaningful diversity gains"; we measure
  both per-sample accuracy and coverage@4 (the only diversity that is *useful* for factual QA).
- Exp 3 (HumanEval with execution): tests "useful range 0.2–0.6, sharp collapse above 0.8".
- Exp 4 (cross-model analysis on Exps 1–3): useful-range width vs. size (two ladders) → "smaller models have narrower
  useful ranges"; curve-shape comparison at fixed scale (7–9B: Qwen/Llama/Mistral/Gemma; 1–1.5B and 3B: Qwen vs Llama)
  → "curve shape differs across families". A Qwen2.5-7B AWQ run controls for the bf16/AWQ precision confound.

## Research Question
How do output diversity and output quality vary with softmax temperature (0–2.0) across open-ended generation,
factual QA and code generation, and do the useful temperature ranges depend on model size and family?

## Hypothesis Decomposition
- H1 (open-ended): diversity gain per unit T is much smaller above T=1.2 than below it (plateau), while coherence at
  T≥1.4 is 20–40% below its peak.
- H2 (factual QA): accuracy at T∈{0.8..2.0} is significantly lower than at T≤0.7, while coverage@4 (any of 4 samples
  correct) gains little over greedy accuracy.
- H3 (code): pass@1 ≥90% of max only for T in ~[0.2,0.6] (actually: we test whether the useful range upper bound is
  ≈0.6–0.8) and drops sharply (e.g. >30% relative) from T=0.6 to T=1.0.
- H4 (size): the useful-range upper bound T_u (largest T with quality ≥ 90% of the model's best) increases with model
  size within each family.
- H5 (family): at matched scale, temperature–diversity curves (and T_u / T50) differ significantly across families.

Independent variables: temperature, model (size, family), task. Dependent: diversity and quality metrics below.

## Proposed Methodology

### Approach
Real model generations with vLLM 0.32 on local GPUs (no API key is available on this host; the hosted
Qwen3-8B vLLM server on :8011 serves as the LLM judge). Pure temperature sampling (top_p=1, top_k off, min_p=0,
repetition_penalty=1, `generation_config="vllm"` so checkpoint defaults like Qwen's top_p=0.8 are NOT applied).
Temperatures {0, .2, .4, .6, .7, .8, 1.0, 1.2, 1.4, 1.6, 2.0}; 4 samples per (prompt, T>0), 1 greedy sample at T=0.

### Experimental Steps
1. `prep_data.py`: 100 WritingPrompts, 300 TriviaQA (rc.nocontext val), 164 HumanEval, seed 42.
2. `worker.py`: generate for every model (stories ≤256 tokens, QA ≤24, code ≤448); store text, finish reason and the
   raw (T=1) log-prob of each sampled token.
3. `score.py`: QA accuracy (normalized alias match), HumanEval execution (sandboxed subprocess, 10 s timeout),
   story diversity (mpnet embedding pairwise distance, distinct-2), lexical-validity and self-NLL proxies.
4. `judge.py`: Qwen3-8B judges story coherence/fluency 1–9 (probability-weighted digit, validated logprobs) on 2 of
   the 4 samples per (prompt, T).
5. `analyze.py`: curves, useful ranges, bootstrap CIs, hypothesis tests, figures.

### Baselines
Greedy decoding (T=0) and the conventional T=0.7/1.0 defaults are the reference points; each model is its own
baseline (quality is normalized to the model's best temperature).

### Evaluation Metrics
- Stories: semantic diversity = mean pairwise cosine distance of the 4 samples (all-mpnet-base-v2); lexical diversity
  = distinct-2 over the pooled samples; coherence = judge score 1–9; proxies = fraction of tokens that are dictionary
  words (vocabulary built from the WritingPrompts train split) and mean raw log-prob under the generating model.
- QA: per-sample accuracy, coverage@4, number of distinct normalized answers / 4.
- Code: pass@1 (mean over samples), pass@4, syntactic validity, distinct-program ratio.
- Derived: T_u (largest T with quality ≥ 0.9×best, linear interpolation), T50 (quality = 0.5×best).

### Statistical Analysis Plan
Item-level bootstrap (2,000 resamples) for 95% CIs; paired sign-flip permutation tests on item-level differences
between temperatures (α=0.05, Holm correction within each hypothesis family); Spearman correlation of T_u with
log(params); family differences in T_u/T50 via bootstrap CIs of pairwise differences. Plateau test: ratio of
diversity slope over [1.2,2.0] to slope over [0.2,1.2] with bootstrap CI.

## Expected Outcomes
Support: diversity slope ratio ≪1 above 1.2 with coherence drop 20–40%; QA accuracy drop above 0.7 with coverage@4
≈ greedy accuracy; code T_u≈0.6–0.8; T_u increasing with size; non-overlapping CIs for family curve parameters.
Refute: diversity keeps rising roughly linearly to T=2 (no plateau); coherence collapse much larger (>40%) or later;
QA flat up to 1.0; code flat to 1.0; no size trend.

## Timeline and Milestones
Setup+data 30 min; generation ~3–4 h (shared GPUs, run largest models first on the free GPU); scoring and judging in
parallel as model files complete; analysis 45 min; report 45 min.

## Potential Challenges
- GPU contention (shared host): hold GPU0 memory from the start, run models in one long-lived worker.
- Judge bias (Qwen3 judging Qwen2.5 outputs) → judge-free proxies reported alongside.
- Degenerate high-T outputs inflate lexical/semantic "diversity" → report diversity restricted to coherent samples.
- Precision confound (AWQ for Mistral/Gemma) → Qwen2.5-7B AWQ vs bf16 control.

## Success Criteria
All 10 models × 11 temperatures × 3 tasks generated and scored; each sub-hypothesis gets a supported / refuted /
partially-supported verdict with CIs.
