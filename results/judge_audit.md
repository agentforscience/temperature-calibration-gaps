# Judge audit (manual, by the researcher-agent)
16 randomly drawn Qwen2.5-0.5B stories at T in {1.0, 1.2} (the coherence boundary region), sample 0;
texts in `results/judge_audit_sample.json`. Manual label = "readable, grammatical story" (yes/no), made by reading
each text before comparing with the judge score.

| pid | T | judge | manual coherent | judge>=5 |
|---|---|---|---|---|
| story095 | 1.2 | 3.01 | no (word salad) | no |
| story082 | 1.0 | 4.17 | no (incoherent sentences) | no |
| story073 | 1.0 | 5.19 | yes | yes |
| story039 | 1.0 | 5.05 | yes | yes |
| story038 | 1.0 | 4.98 | yes | no |
| story022 | 1.0 | 5.09 | yes | yes |
| story005 | 1.0 | 7.20 | yes | yes |
| story043 | 1.2 | 3.04 | no (mixed-language garbage) | no |
| story017 | 1.0 | 4.73 | no (code-switches to Chinese mid-sentence) | no |
| story028 | 1.0 | 4.94 | yes | no |
| story066 | 1.2 | 4.33 | no (blanks, broken) | no |
| story025 | 1.2 | 3.33 | no | no |
| story034 | 1.2 | 3.71 | no | no |
| story069 | 1.0 | 4.47 | yes (odd but readable) | no |
| story056 | 1.2 | 3.73 | no | no |
| story057 | 1.2 | 5.22 | no (not a story: asks for the prompt) | yes |

Agreement 12/16 (75%). Errors are at the threshold: three readable stories scored 4.47–4.98 (judge slightly
conservative) and one non-story scored 5.22. The judge never assigns <3 even to pure word salad, so the 1–9 scale
is effectively 3–9; relative drops on the raw scale therefore understate degradation, which is why the share of
samples with score >=5 ("coherent rate") is used for useful-range estimates.
