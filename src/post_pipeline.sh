#!/bin/bash
# Score + judge each model as soon as its generation worker logs "done <model>". Run from workspace root.
export NUMPY_MADVISE_HUGEPAGE=0
while true; do
  for m in $(cat logs/queue_gpu0c.txt logs/queue_extra.txt 2>/dev/null); do
    [ -f results/judge/$m.done ] && continue
    cat logs/worker3_*.log 2>/dev/null | grep -q "^[0-9:]* done $m;" || continue
    if [ ! -f results/scores/$m.groups.parquet ]; then
      CUDA_VISIBLE_DEVICES=${EMB_GPU:-2} .venv/bin/python src/score.py $m >> logs/score_$m.log 2>&1 || CUDA_VISIBLE_DEVICES= .venv/bin/python src/score.py $m >> logs/score_$m.log 2>&1
    fi
    .venv/bin/python src/judge.py $m >> logs/judge_$m.log 2>&1 && touch results/judge/$m.done
  done
  [ -f logs/STOP_post ] && exit 0
  sleep 30
done
