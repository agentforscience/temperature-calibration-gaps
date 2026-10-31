# Temperature Calibration Gaps

How softmax temperature (0–2.0) affects diversity and quality across 10 open-weight models on stories (WritingPrompts),
factual QA (TriviaQA) and code (HumanEval). All models ran locally because no API key was available, so hosted frontier
models were not tested.

## Key findings
- **Quality drops off a cliff, and the location depends on the model.** Story T_u (coherence ≥90% of peak) runs from
  0.84 for Qwen2.5-0.5B to 1.31 for Qwen2.5-14B.
- **Open-ended:** diversity stops improving, but much of the apparent plateau is measurement (metric ceiling, word
  salad embedding alike). Coherence loss usually exceeds the predicted 20–40%: 48–63% at T = 1.4 in 9/10 models.
- **Factual QA:** T = 0.8–1.0 is worse than T ≤ 0.7 in 9/10 models, but by only 0.7–5.2 points, and coverage@4 gains
  nothing further above 0.7.
- **Code:** the "useful range 0.2–0.6, collapse above 0.8" claim is refuted. No model drops significantly from 0.6 to
  0.8, and five models peak at greedy. Qwen ≥3B and Gemma stay near peak to T ≈ 1.1–1.4.
- **Size and family:** larger Qwen models tolerate higher T on all tasks, but Llama only on QA. Llama's story cliff stays
  at T ≈ 1.0 from 1B to 8B. Family differences at matched size are 4–7× the seed noise floor.

Full details, caveats and tables are in [REPORT.md](REPORT.md).

## Reproduce
```bash
UV_CACHE_DIR=$PWD/.uv_cache uv sync
./run_all.sh     # needs a GPU, local HF cache, and a Qwen3-8B OpenAI-compatible server on :8011
```
Re-run the analysis only: `NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python src/analyze.py`

## Files
- `planning.md`: motivation and pre-experiment plan
- `src/`: `prep_data.py`, `worker.py` (generation), `score.py`, `judge.py`, `analyze.py`, `error_analysis.py`
- `results/gens/`: raw outputs; `results/scores/`, `results/judge/`: metrics; `results/analysis/`: tables
- `results/judge_audit.md`: manual check of the judge
- `figures/`: plots used in the report
