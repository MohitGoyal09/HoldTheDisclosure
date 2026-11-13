# Results

Generated from the artifacts in `outputs/kaggle-second-model/htd/outputs/eval_llama` on 2026-11-11. Commit `77b7870`.
All numbers come from `scripts/render_results.py`; do not edit them by hand.

## Test split

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama_base | plain | en | 20 | 50% | 25% | 15% | 5% | 0% | 0% | 0%-0% | 1.85 | 10 | 0% |
| llama_base | plain | hi | 20 | 25% | 0% | 0% | 0% | 0% | 0% | 0%-0% | 1.25 | 5 | 0% |
| llama_base | plain | hinglish | 20 | 25% | 10% | 0% | 5% | 0% | 0% | 0%-0% | 1.25 | 5 | 0% |
| llama_sft | plain | en | 20 | 100% | 75% | 65% | 80% | 45% | 45% | 20%-65% | 4.30 | 20 | 45% |
| llama_sft | plain | hi | 20 | 50% | 25% | 55% | 30% | 30% | 30% | 10%-50% | 2.50 | 10 | 60% |
| llama_sft | plain | hinglish | 20 | 75% | 75% | 75% | 75% | 75% | 75% | 55%-95% | 4.75 | 15 | 100% |

Turn-1 to turn-5 transitions

| model | variant | script | stated both | stated then dropped | missed then stated | missed both |
|---|---|---|---|---|---|---|
| llama_base | plain | en | 0 | 10 | 0 | 10 |
| llama_base | plain | hi | 0 | 5 | 0 | 15 |
| llama_base | plain | hinglish | 0 | 5 | 0 | 15 |
| llama_sft | plain | en | 9 | 11 | 0 | 0 |
| llama_sft | plain | hi | 6 | 4 | 0 | 10 |
| llama_sft | plain | hinglish | 15 | 0 | 0 | 5 |

## Held-out products and tactics

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama_base | plain | en | 20 | 100% | 5% | 35% | 30% | 20% | 20% | 5%-40% | 2.10 | 20 | 20% |
| llama_base | plain | hi | 20 | 50% | 5% | 20% | 5% | 5% | 5% | 0%-15% | 1.60 | 10 | 0% |
| llama_base | plain | hinglish | 20 | 90% | 10% | 15% | 15% | 10% | 10% | 0%-25% | 2.00 | 18 | 11% |
| llama_sft | plain | en | 20 | 100% | 95% | 95% | 85% | 85% | 85% | 70%-100% | 5.45 | 20 | 85% |
| llama_sft | plain | hi | 20 | 50% | 50% | 50% | 50% | 45% | 45% | 20%-65% | 2.85 | 10 | 70% |
| llama_sft | plain | hinglish | 20 | 50% | 60% | 75% | 65% | 75% | 75% | 55%-95% | 3.50 | 10 | 100% |

## Control products


Control products (unneeded-warning rate)

| model | variant | script | replies | warning rate |
|---|---|---|---|---|
| llama_base | plain | en | 100 | 0% |
| llama_base | plain | hi | 100 | 0% |
| llama_base | plain | hinglish | 100 | 0% |
| llama_sft | plain | en | 100 | 0% |
| llama_sft | plain | hi | 100 | 0% |
| llama_sft | plain | hinglish | 100 | 0% |

## DR@5 against base, test

McNemar exact test, paired by dialogue. Holm family: 0 pre-set recipes (sft, at, at_drop). Other rows (replicates) are not in the family.

| system | n | base | system | diff | 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B SFT | 60 | 0.0% | 50.0% | +50.0 | +38.3 to +61.7 | 0.0000 | not in family |

## DR@5 against base, held-out

McNemar exact test, paired by dialogue. Holm family: 0 pre-set recipes (sft, at, at_drop). Other rows (replicates) are not in the family.

| system | n | base | system | diff | 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B SFT | 60 | 11.7% | 68.3% | +56.7 | +41.7 to +70.0 | 0.0000 | not in family |

## Llama-3.2-3B SFT vs base, test

DR@5: A = llama_base:plain, B = llama_sft:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 0.0% | 50.0% | +50.0 pts | [+38.3, +61.7] pts | 0.0000 | yes |
| en | 20 | 0.0% | 45.0% | +45.0 pts | | | |
| hi | 20 | 0.0% | 30.0% | +30.0 pts | | | |
| hinglish | 20 | 0.0% | 75.0% | +75.0 pts | | | |

## Llama-3.2-3B SFT vs base, held-out

DR@5: A = llama_base:plain, B = llama_sft:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 11.7% | 68.3% | +56.7 pts | [+41.7, +70.0] pts | 0.0000 | yes |
| en | 20 | 20.0% | 85.0% | +65.0 pts | | | |
| hi | 20 | 5.0% | 45.0% | +40.0 pts | | | |
| hinglish | 20 | 10.0% | 75.0% | +65.0 pts | | | |

## Judge labels (turn 5, test)

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 0 (0%) | 54 (90%) | 6 (10%) | 0 (0%) |
| at_drop | 60 | 0 (0%) | 55 (92%) | 5 (8%) | 0 (0%) |
| at_drop_lr5e5 | 60 | 7 (12%) | 49 (82%) | 3 (5%) | 1 (2%) |
| at_lr5e5 | 60 | 7 (12%) | 51 (85%) | 1 (2%) | 1 (2%) |
| base | 60 | 1 (2%) | 52 (87%) | 7 (12%) | 0 (0%) |
| llama_base | 60 | 0 (0%) | 60 (100%) | 0 (0%) | 0 (0%) |
| llama_sft | 60 | 21 (35%) | 34 (57%) | 5 (8%) | 0 (0%) |
| sft | 60 | 13 (22%) | 46 (77%) | 1 (2%) | 0 (0%) |
| sft_s1 | 60 | 19 (32%) | 39 (65%) | 2 (3%) | 0 (0%) |
| sft_v5 | 60 | 11 (18%) | 45 (75%) | 4 (7%) | 0 (0%) |

## Rule checker against the judge (turn 5, test)

A checker pass that the judge labels DROP or DISMISS means the reply matched the patterns without stating the disclosure. A trained system with many of these learned checker-matching phrasing.

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 0 (0%) | 59 (98%) | 1 | 0 |
| at_drop | 60 | 4 (7%) | 0 (0%) | 56 (93%) | 4 | 0 |
| at_drop_lr5e5 | 60 | 25 (42%) | 7 (12%) | 42 (70%) | 18 | 0 |
| at_lr5e5 | 60 | 25 (42%) | 7 (12%) | 36 (60%) | 21 | 3 |
| base | 60 | 8 (13%) | 1 (2%) | 53 (88%) | 7 | 0 |
| llama_base | 60 | 0 (0%) | 0 (0%) | 60 (100%) | 0 | 0 |
| llama_sft | 60 | 30 (50%) | 21 (35%) | 45 (75%) | 12 | 3 |
| sft | 60 | 34 (57%) | 13 (22%) | 33 (55%) | 24 | 3 |
| sft_s1 | 60 | 27 (45%) | 19 (32%) | 40 (67%) | 14 | 6 |
| sft_v5 | 60 | 26 (43%) | 11 (18%) | 41 (68%) | 17 | 2 |

## Judge configurations (turn 5, test)

### v1 (pre-registered)

`outputs/judge_local/v2_qwen.jsonl`, judge model `huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507`, rubric `v2`.

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 0 (0%) | 54 (90%) | 6 (10%) | 0 (0%) |
| at_drop | 60 | 0 (0%) | 55 (92%) | 5 (8%) | 0 (0%) |
| at_drop_lr5e5 | 60 | 7 (12%) | 49 (82%) | 3 (5%) | 1 (2%) |
| at_lr5e5 | 60 | 7 (12%) | 51 (85%) | 1 (2%) | 1 (2%) |
| base | 60 | 1 (2%) | 52 (87%) | 7 (12%) | 0 (0%) |
| llama_base | 60 | 0 (0%) | 60 (100%) | 0 (0%) | 0 (0%) |
| llama_sft | 60 | 21 (35%) | 34 (57%) | 5 (8%) | 0 (0%) |
| sft | 60 | 13 (22%) | 46 (77%) | 1 (2%) | 0 (0%) |
| sft_s1 | 60 | 19 (32%) | 39 (65%) | 2 (3%) | 0 (0%) |
| sft_v5 | 60 | 11 (18%) | 45 (75%) | 4 (7%) | 0 (0%) |

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 0 (0%) | 59 (98%) | 1 | 0 |
| at_drop | 60 | 4 (7%) | 0 (0%) | 56 (93%) | 4 | 0 |
| at_drop_lr5e5 | 60 | 25 (42%) | 7 (12%) | 42 (70%) | 18 | 0 |
| at_lr5e5 | 60 | 25 (42%) | 7 (12%) | 36 (60%) | 21 | 3 |
| base | 60 | 8 (13%) | 1 (2%) | 53 (88%) | 7 | 0 |
| llama_base | 60 | 0 (0%) | 0 (0%) | 60 (100%) | 0 | 0 |
| llama_sft | 60 | 30 (50%) | 21 (35%) | 45 (75%) | 12 | 3 |
| sft | 60 | 34 (57%) | 13 (22%) | 33 (55%) | 24 | 3 |
| sft_s1 | 60 | 27 (45%) | 19 (32%) | 40 (67%) | 14 | 6 |
| sft_v5 | 60 | 26 (43%) | 11 (18%) | 41 (68%) | 17 | 2 |

### LLM panel reference (majority)

`outputs/judge_local/panel_reference.jsonl`, judge model `panel:huggingface/meta-llama/Llama-3.3-70B-Instruct,huggingface/deepseek-ai/DeepSeek-V3-0324,huggingface/openai/gpt-oss-120b`, rubric `panel`.

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 59 | 0 (0%) | 59 (100%) | 0 (0%) | 0 (0%) |
| at_drop | 59 | 0 (0%) | 59 (100%) | 0 (0%) | 0 (0%) |
| at_drop_lr5e5 | 60 | 30 (50%) | 30 (50%) | 0 (0%) | 0 (0%) |
| at_lr5e5 | 58 | 30 (52%) | 27 (47%) | 0 (0%) | 1 (2%) |
| base | 59 | 5 (8%) | 54 (92%) | 0 (0%) | 0 (0%) |
| llama_base | 60 | 0 (0%) | 59 (98%) | 1 (2%) | 0 (0%) |
| llama_sft | 60 | 37 (62%) | 23 (38%) | 0 (0%) | 0 (0%) |
| sft | 60 | 40 (67%) | 20 (33%) | 0 (0%) | 0 (0%) |
| sft_s1 | 60 | 40 (67%) | 20 (33%) | 0 (0%) | 0 (0%) |
| sft_v5 | 60 | 34 (57%) | 26 (43%) | 0 (0%) | 0 (0%) |

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 59 | 0 (0%) | 0 (0%) | 59 (100%) | 0 | 0 |
| at_drop | 59 | 3 (5%) | 0 (0%) | 56 (95%) | 3 | 0 |
| at_drop_lr5e5 | 60 | 25 (42%) | 30 (50%) | 51 (85%) | 2 | 7 |
| at_lr5e5 | 58 | 23 (40%) | 30 (52%) | 45 (78%) | 2 | 10 |
| base | 59 | 7 (12%) | 5 (8%) | 57 (97%) | 2 | 0 |
| llama_base | 60 | 0 (0%) | 0 (0%) | 60 (100%) | 0 | 0 |
| llama_sft | 60 | 30 (50%) | 37 (62%) | 53 (88%) | 0 | 7 |
| sft | 60 | 34 (57%) | 40 (67%) | 52 (87%) | 1 | 7 |
| sft_s1 | 60 | 27 (45%) | 40 (67%) | 47 (78%) | 0 | 13 |
| sft_v5 | 60 | 26 (43%) | 34 (57%) | 50 (83%) | 1 | 9 |

## Judge agreement with the LLM reference panel (not human labels)

Reference = majority of three models (huggingface/meta-llama/Llama-3.3-70B-Instruct,huggingface/deepseek-ai/DeepSeek-V3-0324,huggingface/openai/gpt-oss-120b), neutral rubric from the pre-registered definitions. 120 reference rows.

| judge | n | agreement | Cohen's kappa |
|---|---|---|---|
| v1 (pre-registered) | 120 | 0.858 | 0.655 |
| LLM panel reference (majority) | 120 | 1.000 | 1.000 |

Panel agreement on these replies: Fleiss' kappa 0.925 over 120 replies labelled by all 3 panel models.

## Helpfulness at turn 5 (LLM panel majority, not human labels)

Does the reply give the help the customer asked for, regardless of warnings? Panel: Llama-3.3-70B-Instruct, DeepSeek-V3-0324, gpt-oss-120b.

| system | disclosure products: n | helpful | control products: n | helpful |
|---|---|---|---|---|
| at | 60 | 65% | 60 | 72% |
| at_drop | 60 | 72% | 60 | 73% |
| at_drop_lr5e5 | 60 | 55% | 60 | 67% |
| at_lr5e5 | 60 | 47% | 60 | 53% |
| base | 60 | 63% | 60 | 65% |
| llama_base | 60 | 88% | 60 | 55% |
| llama_sft | 60 | 73% | 60 | 55% |
| sft | 60 | 52% | 60 | 42% |
| sft_s1 | 60 | 48% | 60 | 45% |
| sft_v5 | 60 | 58% | 60 | 40% |
