# Datasets (symlinks into the local HF cache; not committed)
| file | HF source | used |
|---|---|---|
| writingprompts_test.parquet | `euclaise/writingprompts` (test) | 100 prompts, open-ended story generation |
| triviaqa_rc_nocontext_validation.parquet | `mandarjoshi/trivia_qa`, config `rc.nocontext`, validation | 300 questions, closed-book factual QA |
| humaneval_test.parquet | `openai/openai_humaneval` (test) | all 164 problems, code generation |

Re-download: `huggingface-cli download --repo-type dataset <repo>` or `datasets.load_dataset(...)`.
`src/prep_data.py` samples items with seed 42 and writes `data/prompts.jsonl`.
