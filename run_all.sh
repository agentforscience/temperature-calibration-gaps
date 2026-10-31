#!/bin/bash
# Full pipeline (single-GPU version; on the shared host several workers were run in parallel, see REPORT.md).
set -e
export NUMPY_MADVISE_HUGEPAGE=0 HF_HOME=/data/transformers HF_HUB_OFFLINE=1
source .venv/bin/activate
python src/prep_data.py                                   # data/prompts.jsonl (564 items)
printf "%s\n" qwen2.5-0.5b qwen2.5-1.5b qwen2.5-3b qwen2.5-7b qwen2.5-14b llama3.2-1b llama3.2-3b llama3.1-8b \
  mistral-7b gemma2-9b qwen2.5-0.5b-seed2 qwen2.5-7b-bf16 > logs/queue_all.txt
# generation: one long-lived worker; stop it with `touch logs/STOP_workers` when results/gens has all 11 files
CUDA_VISIBLE_DEVICES=0 TAKEOVER_PID=999999999 TAKEOVER_GB=24 python src/worker.py logs/queue_all.txt 24 &
# scoring (HumanEval execution, embeddings) + LLM judge (needs an OpenAI-compatible Qwen3-8B server on :8011)
for m in $(cat logs/queue_all.txt); do
  until grep -q "^[0-9:]* done $m;" logs/worker*.log 2>/dev/null; do sleep 60; done
  python src/score.py $m && python src/judge.py $m
done
touch logs/STOP_workers
python src/analyze.py
