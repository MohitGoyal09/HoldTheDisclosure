# Results

Generated from the artifacts in `outputs/kaggle-eval-2x2/htd/outputs/eval` on 2026-11-11. Commit `0188b90`.
All numbers come from `scripts/render_results.py`; do not edit them by hand.

## Test split

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| at | plain | en | 20 | 100% | 50% | 0% | 15% | 0% | 0% | 0%-0% | 2.50 | 20 | 0% |
| at | plain | hi | 20 | 100% | 50% | 15% | 15% | 5% | 5% | 0%-15% | 2.65 | 20 | 5% |
| at | plain | hinglish | 20 | 25% | 10% | 10% | 15% | 0% | 0% | 0%-0% | 1.35 | 5 | 0% |
| at_drop | plain | en | 20 | 100% | 50% | 10% | 15% | 0% | 0% | 0%-0% | 2.55 | 20 | 0% |
| at_drop | plain | hi | 20 | 100% | 45% | 25% | 30% | 15% | 15% | 0%-30% | 2.65 | 20 | 15% |
| at_drop | plain | hinglish | 20 | 0% | 10% | 20% | 10% | 5% | 5% | 0%-15% | 1.00 | 0 | n/a |
| at_drop_lr5e5 | plain | en | 20 | 75% | 65% | 20% | 35% | 25% | 25% | 5%-45% | 2.95 | 15 | 33% |
| at_drop_lr5e5 | plain | hi | 20 | 100% | 65% | 40% | 45% | 35% | 35% | 15%-55% | 3.65 | 20 | 35% |
| at_drop_lr5e5 | plain | hinglish | 20 | 75% | 75% | 55% | 60% | 65% | 65% | 45%-85% | 4.15 | 15 | 87% |
| at_lr5e5 | plain | en | 20 | 25% | 25% | 15% | 25% | 20% | 20% | 5%-40% | 1.95 | 5 | 80% |
| at_lr5e5 | plain | hi | 20 | 75% | 25% | 45% | 35% | 40% | 40% | 20%-60% | 2.75 | 15 | 53% |
| at_lr5e5 | plain | hinglish | 20 | 75% | 65% | 50% | 60% | 65% | 45% | 45%-85% | 3.60 | 15 | 87% |
| base | plain | en | 20 | 100% | 50% | 20% | 25% | 0% | 0% | 0%-0% | 2.75 | 20 | 0% |
| base | plain | hi | 20 | 100% | 35% | 10% | 20% | 15% | 15% | 0%-30% | 2.55 | 20 | 15% |
| base | plain | hinglish | 20 | 50% | 35% | 35% | 30% | 25% | 25% | 5%-45% | 2.45 | 10 | 50% |
| base | strict | en | 20 | 100% | 65% | 25% | 45% | 0% | 0% | 0%-0% | 3.00 | 20 | 0% |
| base | strict | hi | 20 | 100% | 50% | 20% | 40% | 30% | 30% | 10%-50% | 2.95 | 20 | 30% |
| base | strict | hinglish | 20 | 75% | 60% | 25% | 5% | 5% | 5% | 0%-15% | 2.40 | 15 | 7% |
| base | third_person | en | 20 | 50% | 15% | 0% | 5% | 0% | 0% | 0%-0% | 1.55 | 10 | 0% |
| base | third_person | hi | 20 | 75% | 35% | 10% | 20% | 15% | 15% | 0%-30% | 2.10 | 15 | 20% |
| base | third_person | hinglish | 20 | 75% | 10% | 10% | 10% | 10% | 0% | 0%-25% | 2.00 | 15 | 13% |
| sft | plain | en | 20 | 100% | 50% | 75% | 60% | 55% | 55% | 35%-75% | 3.95 | 20 | 55% |
| sft | plain | hi | 20 | 75% | 40% | 40% | 50% | 50% | 50% | 30%-70% | 3.35 | 15 | 67% |
| sft | plain | hinglish | 20 | 75% | 50% | 65% | 65% | 65% | 65% | 45%-85% | 3.60 | 15 | 87% |
| sft_s1 | plain | en | 20 | 100% | 50% | 75% | 50% | 45% | 45% | 25%-65% | 3.65 | 20 | 45% |
| sft_s1 | plain | hi | 20 | 50% | 15% | 25% | 25% | 25% | 25% | 5%-45% | 2.10 | 10 | 50% |
| sft_s1 | plain | hinglish | 20 | 75% | 65% | 65% | 65% | 65% | 65% | 45%-85% | 4.20 | 15 | 87% |
| sft_v5 | plain | en | 20 | 100% | 35% | 60% | 35% | 25% | 25% | 10%-45% | 3.15 | 20 | 25% |
| sft_v5 | plain | hi | 20 | 75% | 25% | 30% | 30% | 30% | 30% | 10%-50% | 2.60 | 15 | 40% |
| sft_v5 | plain | hinglish | 20 | 75% | 50% | 60% | 65% | 75% | 75% | 55%-95% | 3.30 | 15 | 100% |

Turn-1 to turn-5 transitions

| model | variant | script | stated both | stated then dropped | missed then stated | missed both |
|---|---|---|---|---|---|---|
| at | plain | en | 0 | 20 | 0 | 0 |
| at | plain | hi | 1 | 19 | 0 | 0 |
| at | plain | hinglish | 0 | 5 | 0 | 15 |
| at_drop | plain | en | 0 | 20 | 0 | 0 |
| at_drop | plain | hi | 3 | 17 | 0 | 0 |
| at_drop | plain | hinglish | 0 | 0 | 1 | 19 |
| at_drop_lr5e5 | plain | en | 5 | 10 | 0 | 5 |
| at_drop_lr5e5 | plain | hi | 7 | 13 | 0 | 0 |
| at_drop_lr5e5 | plain | hinglish | 13 | 2 | 0 | 5 |
| at_lr5e5 | plain | en | 4 | 1 | 0 | 15 |
| at_lr5e5 | plain | hi | 8 | 7 | 0 | 5 |
| at_lr5e5 | plain | hinglish | 13 | 2 | 0 | 5 |
| base | plain | en | 0 | 20 | 0 | 0 |
| base | plain | hi | 3 | 17 | 0 | 0 |
| base | plain | hinglish | 5 | 5 | 0 | 10 |
| base | strict | en | 0 | 20 | 0 | 0 |
| base | strict | hi | 6 | 14 | 0 | 0 |
| base | strict | hinglish | 1 | 14 | 0 | 5 |
| base | third_person | en | 0 | 10 | 0 | 10 |
| base | third_person | hi | 3 | 12 | 0 | 5 |
| base | third_person | hinglish | 2 | 13 | 0 | 5 |
| sft | plain | en | 11 | 9 | 0 | 0 |
| sft | plain | hi | 10 | 5 | 0 | 5 |
| sft | plain | hinglish | 13 | 2 | 0 | 5 |
| sft_s1 | plain | en | 9 | 11 | 0 | 0 |
| sft_s1 | plain | hi | 5 | 5 | 0 | 10 |
| sft_s1 | plain | hinglish | 13 | 2 | 0 | 5 |
| sft_v5 | plain | en | 5 | 15 | 0 | 0 |
| sft_v5 | plain | hi | 6 | 9 | 0 | 5 |
| sft_v5 | plain | hinglish | 15 | 0 | 0 | 5 |

## Held-out products and tactics

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| at | plain | en | 20 | 100% | 35% | 35% | 45% | 45% | 45% | 25%-65% | 2.85 | 20 | 45% |
| at | plain | hi | 20 | 100% | 0% | 10% | 5% | 20% | 20% | 5%-40% | 2.00 | 20 | 20% |
| at | plain | hinglish | 20 | 100% | 45% | 40% | 30% | 45% | 45% | 25%-65% | 3.45 | 20 | 45% |
| at_drop | plain | en | 20 | 100% | 55% | 50% | 30% | 55% | 55% | 35%-75% | 3.10 | 20 | 55% |
| at_drop | plain | hi | 20 | 100% | 0% | 10% | 0% | 15% | 15% | 0%-30% | 2.00 | 20 | 15% |
| at_drop | plain | hinglish | 20 | 100% | 40% | 45% | 35% | 40% | 40% | 20%-60% | 3.45 | 20 | 40% |
| at_drop_lr5e5 | plain | en | 20 | 100% | 35% | 55% | 70% | 80% | 80% | 60%-95% | 3.10 | 20 | 80% |
| at_drop_lr5e5 | plain | hi | 20 | 100% | 45% | 50% | 55% | 45% | 45% | 25%-65% | 3.65 | 20 | 45% |
| at_drop_lr5e5 | plain | hinglish | 20 | 100% | 15% | 30% | 20% | 15% | 10% | 0%-30% | 2.60 | 20 | 15% |
| at_lr5e5 | plain | en | 20 | 100% | 20% | 35% | 35% | 40% | 40% | 20%-60% | 2.50 | 20 | 40% |
| at_lr5e5 | plain | hi | 20 | 100% | 70% | 90% | 85% | 95% | 95% | 85%-100% | 4.80 | 20 | 95% |
| at_lr5e5 | plain | hinglish | 20 | 100% | 40% | 60% | 60% | 55% | 0% | 35%-75% | 3.60 | 20 | 55% |
| base | plain | en | 20 | 100% | 15% | 30% | 30% | 40% | 40% | 20%-60% | 2.30 | 20 | 40% |
| base | plain | hi | 20 | 100% | 0% | 20% | 5% | 10% | 10% | 0%-25% | 2.00 | 20 | 10% |
| base | plain | hinglish | 20 | 100% | 35% | 45% | 30% | 35% | 35% | 15%-55% | 3.30 | 20 | 35% |
| sft | plain | en | 20 | 100% | 80% | 90% | 80% | 80% | 80% | 60%-95% | 5.05 | 20 | 80% |
| sft | plain | hi | 20 | 100% | 75% | 80% | 70% | 70% | 70% | 50%-90% | 4.60 | 20 | 70% |
| sft | plain | hinglish | 20 | 100% | 55% | 60% | 60% | 65% | 65% | 45%-85% | 4.20 | 20 | 65% |
| sft_s1 | plain | en | 20 | 100% | 80% | 85% | 85% | 80% | 80% | 60%-95% | 5.15 | 20 | 80% |
| sft_s1 | plain | hi | 20 | 100% | 70% | 80% | 75% | 75% | 75% | 55%-95% | 4.70 | 20 | 75% |
| sft_s1 | plain | hinglish | 20 | 100% | 45% | 50% | 55% | 55% | 55% | 35%-75% | 3.80 | 20 | 55% |
| sft_v5 | plain | en | 20 | 100% | 80% | 85% | 90% | 85% | 85% | 65%-100% | 5.20 | 20 | 85% |
| sft_v5 | plain | hi | 20 | 100% | 75% | 70% | 70% | 70% | 70% | 50%-90% | 4.45 | 20 | 70% |
| sft_v5 | plain | hinglish | 20 | 100% | 70% | 75% | 75% | 75% | 75% | 55%-95% | 4.80 | 20 | 75% |

## Control products


Control products (unneeded-warning rate)

| model | variant | script | replies | warning rate |
|---|---|---|---|---|
| at | plain | en | 100 | 0% |
| at | plain | hi | 100 | 0% |
| at | plain | hinglish | 100 | 0% |
| at_drop | plain | en | 100 | 0% |
| at_drop | plain | hi | 100 | 0% |
| at_drop | plain | hinglish | 100 | 0% |
| at_drop_lr5e5 | plain | en | 100 | 0% |
| at_drop_lr5e5 | plain | hi | 100 | 0% |
| at_drop_lr5e5 | plain | hinglish | 100 | 0% |
| at_lr5e5 | plain | en | 100 | 0% |
| at_lr5e5 | plain | hi | 100 | 0% |
| at_lr5e5 | plain | hinglish | 100 | 0% |
| base | plain | en | 100 | 0% |
| base | plain | hi | 100 | 0% |
| base | plain | hinglish | 100 | 0% |
| sft | plain | en | 100 | 0% |
| sft | plain | hi | 100 | 0% |
| sft | plain | hinglish | 100 | 0% |
| sft_s1 | plain | en | 100 | 0% |
| sft_s1 | plain | hi | 100 | 0% |
| sft_s1 | plain | hinglish | 100 | 0% |
| sft_v5 | plain | en | 100 | 0% |
| sft_v5 | plain | hi | 100 | 0% |
| sft_v5 | plain | hinglish | 100 | 0% |

## DR@5 against base, test

McNemar exact test, paired by dialogue. Holm family: 3 pre-set recipes (sft, at, at_drop). Other rows (replicates) are not in the family.

| system | n | base | system | diff | 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| SFT | 60 | 13.3% | 56.7% | +43.3 | +30.0 to +56.7 | 0.0000 | 0.0000 |
| AlignTune DPO | 60 | 13.3% | 1.7% | -11.7 | -20.0 to -5.0 | 0.0156 | 0.0312 |
| AlignTune DPO, drop+warn pairs | 60 | 13.3% | 6.7% | -6.7 | -15.0 to +1.7 | 0.2891 | 0.2891 |
| AlignTune DPO, drop+warn pairs, lr 5e-5 | 60 | 13.3% | 41.7% | +28.3 | +16.7 to +40.0 | 0.0001 | not in family |
| AlignTune DPO, all pairs, lr 5e-5 | 60 | 13.3% | 41.7% | +28.3 | +15.0 to +41.7 | 0.0002 | not in family |
| SFT, seed 1 | 60 | 13.3% | 45.0% | +31.7 | +18.3 to +45.0 | 0.0001 | not in family |
| SFT, training run v5 | 60 | 13.3% | 43.3% | +30.0 | +16.7 to +43.3 | 0.0001 | not in family |
| Strict system prompt (no training) | 60 | 13.3% | 11.7% | -1.7 | -13.3 to +10.0 | 1.0000 | not in family |
| Third-person prompt (no training) | 60 | 13.3% | 8.3% | -5.0 | -15.0 to +5.0 | 0.5488 | not in family |

## DR@5 against base, held-out

McNemar exact test, paired by dialogue. Holm family: 3 pre-set recipes (sft, at, at_drop). Other rows (replicates) are not in the family.

| system | n | base | system | diff | 95% CI | McNemar p | Holm p |
|---|---|---|---|---|---|---|---|
| SFT | 60 | 28.3% | 71.7% | +43.3 | +30.0 to +56.7 | 0.0000 | 0.0000 |
| AlignTune DPO | 60 | 28.3% | 36.7% | +8.3 | -1.7 to +20.0 | 0.2266 | 0.4531 |
| AlignTune DPO, drop+warn pairs | 60 | 28.3% | 36.7% | +8.3 | -1.7 to +18.3 | 0.2266 | 0.4531 |
| AlignTune DPO, drop+warn pairs, lr 5e-5 | 60 | 28.3% | 46.7% | +18.3 | +1.7 to +33.3 | 0.0433 | not in family |
| AlignTune DPO, all pairs, lr 5e-5 | 60 | 28.3% | 63.3% | +35.0 | +20.0 to +50.0 | 0.0001 | not in family |
| SFT, seed 1 | 60 | 28.3% | 70.0% | +41.7 | +28.3 to +55.0 | 0.0000 | not in family |
| SFT, training run v5 | 60 | 28.3% | 76.7% | +48.3 | +35.0 to +61.7 | 0.0000 | not in family |

## AlignTune DPO, drop+warn pairs, lr 5e-5 vs base, test

DR@5: A = base:plain, B = at_drop_lr5e5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 41.7% | +28.3 pts | [+16.7, +40.0] pts | 0.0001 | yes |
| en | 20 | 0.0% | 25.0% | +25.0 pts | | | |
| hi | 20 | 15.0% | 35.0% | +20.0 pts | | | |
| hinglish | 20 | 25.0% | 65.0% | +40.0 pts | | | |

## AlignTune DPO, drop+warn pairs vs base, test

DR@5: A = base:plain, B = at_drop:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 6.7% | -6.7 pts | [-15.0, +1.7] pts | 0.2891 | no |
| en | 20 | 0.0% | 0.0% | +0.0 pts | | | |
| hi | 20 | 15.0% | 15.0% | +0.0 pts | | | |
| hinglish | 20 | 25.0% | 5.0% | -20.0 pts | | | |

## AlignTune DPO, all pairs, lr 5e-5 vs base, test

DR@5: A = base:plain, B = at_lr5e5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 41.7% | +28.3 pts | [+15.0, +41.7] pts | 0.0002 | yes |
| en | 20 | 0.0% | 20.0% | +20.0 pts | | | |
| hi | 20 | 15.0% | 40.0% | +25.0 pts | | | |
| hinglish | 20 | 25.0% | 65.0% | +40.0 pts | | | |

## AlignTune DPO vs base, test

DR@5: A = base:plain, B = at:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 1.7% | -11.7 pts | [-20.0, -5.0] pts | 0.0156 | yes |
| en | 20 | 0.0% | 0.0% | +0.0 pts | | | |
| hi | 20 | 15.0% | 5.0% | -10.0 pts | | | |
| hinglish | 20 | 25.0% | 0.0% | -25.0 pts | | | |

## SFT, seed 1 vs base, test

DR@5: A = base:plain, B = sft_s1:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 45.0% | +31.7 pts | [+18.3, +45.0] pts | 0.0001 | yes |
| en | 20 | 0.0% | 45.0% | +45.0 pts | | | |
| hi | 20 | 15.0% | 25.0% | +10.0 pts | | | |
| hinglish | 20 | 25.0% | 65.0% | +40.0 pts | | | |

## SFT vs base, test

DR@5: A = base:plain, B = sft:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 56.7% | +43.3 pts | [+30.0, +56.7] pts | 0.0000 | yes |
| en | 20 | 0.0% | 55.0% | +55.0 pts | | | |
| hi | 20 | 15.0% | 50.0% | +35.0 pts | | | |
| hinglish | 20 | 25.0% | 65.0% | +40.0 pts | | | |

## SFT, training run v5 vs base, test

DR@5: A = base:plain, B = sft_v5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 43.3% | +30.0 pts | [+16.7, +43.3] pts | 0.0001 | yes |
| en | 20 | 0.0% | 25.0% | +25.0 pts | | | |
| hi | 20 | 15.0% | 30.0% | +15.0 pts | | | |
| hinglish | 20 | 25.0% | 75.0% | +50.0 pts | | | |

## Strict system prompt (no training) vs base, test

DR@5: A = base:plain, B = base:strict (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 11.7% | -1.7 pts | [-13.3, +10.0] pts | 1.0000 | no |
| en | 20 | 0.0% | 0.0% | +0.0 pts | | | |
| hi | 20 | 15.0% | 30.0% | +15.0 pts | | | |
| hinglish | 20 | 25.0% | 5.0% | -20.0 pts | | | |

## Third-person prompt (no training) vs base, test

DR@5: A = base:plain, B = base:third_person (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 8.3% | -5.0 pts | [-15.0, +5.0] pts | 0.5488 | no |
| en | 20 | 0.0% | 0.0% | +0.0 pts | | | |
| hi | 20 | 15.0% | 15.0% | +0.0 pts | | | |
| hinglish | 20 | 25.0% | 10.0% | -15.0 pts | | | |

## AlignTune DPO, drop+warn pairs vs base, held-out

DR@5: A = base:plain, B = at_drop:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 36.7% | +8.3 pts | [-1.7, +18.3] pts | 0.2266 | no |
| en | 20 | 40.0% | 55.0% | +15.0 pts | | | |
| hi | 20 | 10.0% | 15.0% | +5.0 pts | | | |
| hinglish | 20 | 35.0% | 40.0% | +5.0 pts | | | |

## AlignTune DPO, drop+warn pairs, lr 5e-5 vs base, held-out

DR@5: A = base:plain, B = at_drop_lr5e5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 46.7% | +18.3 pts | [+1.7, +33.3] pts | 0.0433 | yes |
| en | 20 | 40.0% | 80.0% | +40.0 pts | | | |
| hi | 20 | 10.0% | 45.0% | +35.0 pts | | | |
| hinglish | 20 | 35.0% | 15.0% | -20.0 pts | | | |

## AlignTune DPO vs base, held-out

DR@5: A = base:plain, B = at:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 36.7% | +8.3 pts | [-1.7, +20.0] pts | 0.2266 | no |
| en | 20 | 40.0% | 45.0% | +5.0 pts | | | |
| hi | 20 | 10.0% | 20.0% | +10.0 pts | | | |
| hinglish | 20 | 35.0% | 45.0% | +10.0 pts | | | |

## AlignTune DPO, all pairs, lr 5e-5 vs base, held-out

DR@5: A = base:plain, B = at_lr5e5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 63.3% | +35.0 pts | [+20.0, +50.0] pts | 0.0001 | yes |
| en | 20 | 40.0% | 40.0% | +0.0 pts | | | |
| hi | 20 | 10.0% | 95.0% | +85.0 pts | | | |
| hinglish | 20 | 35.0% | 55.0% | +20.0 pts | | | |

## SFT vs base, held-out

DR@5: A = base:plain, B = sft:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 71.7% | +43.3 pts | [+30.0, +56.7] pts | 0.0000 | yes |
| en | 20 | 40.0% | 80.0% | +40.0 pts | | | |
| hi | 20 | 10.0% | 70.0% | +60.0 pts | | | |
| hinglish | 20 | 35.0% | 65.0% | +30.0 pts | | | |

## SFT, seed 1 vs base, held-out

DR@5: A = base:plain, B = sft_s1:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 70.0% | +41.7 pts | [+28.3, +55.0] pts | 0.0000 | yes |
| en | 20 | 40.0% | 80.0% | +40.0 pts | | | |
| hi | 20 | 10.0% | 75.0% | +65.0 pts | | | |
| hinglish | 20 | 35.0% | 55.0% | +20.0 pts | | | |

## SFT, training run v5 vs base, held-out

DR@5: A = base:plain, B = sft_v5:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 28.3% | 76.7% | +48.3 pts | [+35.0, +61.7] pts | 0.0000 | yes |
| en | 20 | 40.0% | 85.0% | +45.0 pts | | | |
| hi | 20 | 10.0% | 70.0% | +60.0 pts | | | |
| hinglish | 20 | 35.0% | 75.0% | +40.0 pts | | | |

## Judge labels (turn 5, test)

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 22 (37%) | 27 (45%) | 7 (12%) | 4 (7%) |
| at_drop | 60 | 22 (37%) | 29 (48%) | 5 (8%) | 4 (7%) |
| at_drop_lr5e5 | 60 | 36 (60%) | 13 (22%) | 2 (3%) | 9 (15%) |
| at_lr5e5 | 60 | 34 (57%) | 9 (15%) | 3 (5%) | 14 (23%) |
| base | 60 | 23 (38%) | 26 (43%) | 6 (10%) | 5 (8%) |
| sft | 60 | 17 (28%) | 9 (15%) | 1 (2%) | 33 (55%) |
| sft_s1 | 60 | 22 (37%) | 7 (12%) | 1 (2%) | 30 (50%) |
| sft_v5 | 60 | 21 (35%) | 16 (27%) | 1 (2%) | 22 (37%) |

## Rule checker against the judge (turn 5, test)

A checker pass that the judge labels DROP or DISMISS means the reply matched the patterns without stating the disclosure. A trained system with many of these learned checker-matching phrasing.

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 22 (37%) | 37 (62%) | 1 | 22 |
| at_drop | 60 | 4 (7%) | 22 (37%) | 40 (67%) | 0 | 19 |
| at_drop_lr5e5 | 60 | 25 (42%) | 36 (60%) | 37 (62%) | 0 | 17 |
| at_lr5e5 | 60 | 25 (42%) | 34 (57%) | 37 (62%) | 0 | 16 |
| base | 60 | 8 (13%) | 23 (38%) | 37 (62%) | 0 | 19 |
| sft | 60 | 34 (57%) | 17 (28%) | 27 (45%) | 2 | 8 |
| sft_s1 | 60 | 27 (45%) | 22 (37%) | 23 (38%) | 2 | 16 |
| sft_v5 | 60 | 26 (43%) | 21 (35%) | 33 (55%) | 1 | 11 |

## Judge configurations (turn 5, test)

### v1 (pre-registered)

`outputs/kaggle-eval-2x2/htd/outputs/judge/judge_test.jsonl`, judge model `huggingface/google/gemma-3-27b-it`, rubric `v1`.

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 22 (37%) | 27 (45%) | 7 (12%) | 4 (7%) |
| at_drop | 60 | 22 (37%) | 29 (48%) | 5 (8%) | 4 (7%) |
| at_drop_lr5e5 | 60 | 36 (60%) | 13 (22%) | 2 (3%) | 9 (15%) |
| at_lr5e5 | 60 | 34 (57%) | 9 (15%) | 3 (5%) | 14 (23%) |
| base | 60 | 23 (38%) | 26 (43%) | 6 (10%) | 5 (8%) |
| sft | 60 | 17 (28%) | 9 (15%) | 1 (2%) | 33 (55%) |
| sft_s1 | 60 | 22 (37%) | 7 (12%) | 1 (2%) | 30 (50%) |
| sft_v5 | 60 | 21 (35%) | 16 (27%) | 1 (2%) | 22 (37%) |

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 22 (37%) | 37 (62%) | 1 | 22 |
| at_drop | 60 | 4 (7%) | 22 (37%) | 40 (67%) | 0 | 19 |
| at_drop_lr5e5 | 60 | 25 (42%) | 36 (60%) | 37 (62%) | 0 | 17 |
| at_lr5e5 | 60 | 25 (42%) | 34 (57%) | 37 (62%) | 0 | 16 |
| base | 60 | 8 (13%) | 23 (38%) | 37 (62%) | 0 | 19 |
| sft | 60 | 34 (57%) | 17 (28%) | 27 (45%) | 2 | 8 |
| sft_s1 | 60 | 27 (45%) | 22 (37%) | 23 (38%) | 2 | 16 |
| sft_v5 | 60 | 26 (43%) | 21 (35%) | 33 (55%) | 1 | 11 |

### v2 (rubric v2, Qwen3-235B)

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

### ablation: rubric v1, Qwen3-235B

`outputs/judge_local/v1_qwen.jsonl`, judge model `huggingface/Qwen/Qwen3-235B-A22B-Instruct-2507`, rubric `v1`.

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 0 (0%) | 37 (62%) | 22 (37%) | 1 (2%) |
| at_drop | 60 | 4 (7%) | 32 (53%) | 24 (40%) | 0 (0%) |
| at_drop_lr5e5 | 60 | 10 (17%) | 25 (42%) | 18 (30%) | 7 (12%) |
| at_lr5e5 | 60 | 16 (27%) | 20 (33%) | 16 (27%) | 8 (13%) |
| base | 60 | 1 (2%) | 34 (57%) | 22 (37%) | 3 (5%) |
| llama_base | 60 | 0 (0%) | 48 (80%) | 12 (20%) | 0 (0%) |
| llama_sft | 60 | 14 (23%) | 19 (32%) | 11 (18%) | 16 (27%) |
| sft | 60 | 17 (28%) | 15 (25%) | 6 (10%) | 22 (37%) |
| sft_s1 | 60 | 11 (18%) | 15 (25%) | 10 (17%) | 24 (40%) |
| sft_v5 | 60 | 13 (22%) | 21 (35%) | 9 (15%) | 17 (28%) |

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 0 (0%) | 59 (98%) | 1 | 0 |
| at_drop | 60 | 4 (7%) | 4 (7%) | 54 (90%) | 3 | 3 |
| at_drop_lr5e5 | 60 | 25 (42%) | 10 (17%) | 41 (68%) | 11 | 2 |
| at_lr5e5 | 60 | 25 (42%) | 16 (27%) | 33 (55%) | 12 | 9 |
| base | 60 | 8 (13%) | 1 (2%) | 53 (88%) | 4 | 0 |
| llama_base | 60 | 0 (0%) | 0 (0%) | 60 (100%) | 0 | 0 |
| llama_sft | 60 | 30 (50%) | 14 (23%) | 36 (60%) | 4 | 4 |
| sft | 60 | 34 (57%) | 17 (28%) | 37 (62%) | 4 | 3 |
| sft_s1 | 60 | 27 (45%) | 11 (18%) | 32 (53%) | 6 | 6 |
| sft_v5 | 60 | 26 (43%) | 13 (22%) | 37 (62%) | 5 | 5 |

### ablation: rubric v2, Gemma-3-27B

`outputs/judge_local/v2_gemma.jsonl`, judge model `huggingface/google/gemma-3-27b-it`, rubric `v2`.

| system | judged | PASS | DROP | DISMISS | OVER-CORRECT |
|---|---|---|---|---|---|
| at | 60 | 20 (33%) | 38 (63%) | 2 (3%) | 0 (0%) |
| at_drop | 60 | 16 (27%) | 44 (73%) | 0 (0%) | 0 (0%) |
| at_drop_lr5e5 | 60 | 38 (63%) | 22 (37%) | 0 (0%) | 0 (0%) |
| at_lr5e5 | 60 | 39 (65%) | 20 (33%) | 1 (2%) | 0 (0%) |
| base | 60 | 24 (40%) | 34 (57%) | 2 (3%) | 0 (0%) |
| llama_base | 60 | 3 (5%) | 57 (95%) | 0 (0%) | 0 (0%) |
| llama_sft | 60 | 49 (82%) | 10 (17%) | 1 (2%) | 0 (0%) |
| sft | 60 | 47 (78%) | 13 (22%) | 0 (0%) | 0 (0%) |
| sft_s1 | 60 | 49 (82%) | 11 (18%) | 0 (0%) | 0 (0%) |
| sft_v5 | 60 | 43 (72%) | 16 (27%) | 1 (2%) | 0 (0%) |

| system | judged | checker pass | judge PASS | agree | checker pass, judge DROP/DISMISS | judge PASS, checker fail |
|---|---|---|---|---|---|---|
| at | 60 | 1 (2%) | 20 (33%) | 41 (68%) | 0 | 19 |
| at_drop | 60 | 4 (7%) | 16 (27%) | 48 (80%) | 0 | 12 |
| at_drop_lr5e5 | 60 | 25 (42%) | 38 (63%) | 47 (78%) | 0 | 13 |
| at_lr5e5 | 60 | 25 (42%) | 39 (65%) | 46 (77%) | 0 | 14 |
| base | 60 | 8 (13%) | 24 (40%) | 42 (70%) | 1 | 17 |
| llama_base | 60 | 0 (0%) | 3 (5%) | 57 (95%) | 0 | 3 |
| llama_sft | 60 | 30 (50%) | 49 (82%) | 41 (68%) | 0 | 19 |
| sft | 60 | 34 (57%) | 47 (78%) | 47 (78%) | 0 | 13 |
| sft_s1 | 60 | 27 (45%) | 49 (82%) | 38 (63%) | 0 | 22 |
| sft_v5 | 60 | 26 (43%) | 43 (72%) | 43 (72%) | 0 | 17 |

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

Reference = majority of three models (huggingface/meta-llama/Llama-3.3-70B-Instruct,huggingface/deepseek-ai/DeepSeek-V3-0324,huggingface/openai/gpt-oss-120b), neutral rubric from the pre-registered definitions. 475 reference rows.

| judge | n | agreement | Cohen's kappa |
|---|---|---|---|
| v1 (pre-registered) | 475 | 0.465 | 0.201 |
| v2 (rubric v2, Qwen3-235B) | 475 | 0.703 | 0.337 |
| ablation: rubric v1, Qwen3-235B | 475 | 0.552 | 0.343 |
| ablation: rubric v2, Gemma-3-27B | 475 | 0.792 | 0.604 |
| LLM panel reference (majority) | 475 | 1.000 | 1.000 |

Panel agreement on these replies: Fleiss' kappa 0.702 over 480 replies labelled by all 3 panel models.

## Helpfulness at turn 5 (LLM panel majority, not human labels)

Does the reply give the help the customer asked for, regardless of warnings? Panel: Llama-3.3-70B-Instruct, DeepSeek-V3-0324, gpt-oss-120b. Interval: Wilson 95%. p: exact McNemar against the same base model, paired by dialogue (descriptive, not corrected).

| system | disclosure products: n | helpful [95% CI] | p vs base | control products: n | helpful [95% CI] | p vs base |
|---|---|---|---|---|---|---|
| at | 60 | 65% [52, 76] | 1 | 60 | 72% [59, 81] | 0.42 |
| at_drop | 60 | 72% [59, 81] | 0.23 | 60 | 73% [61, 83] | 0.27 |
| at_drop_lr5e5 | 60 | 55% [42, 67] | 0.3 | 60 | 67% [54, 77] | 1 |
| at_lr5e5 | 60 | 47% [35, 59] | 0.052 | 60 | 53% [41, 65] | 0.23 |
| base | 60 | 63% [51, 74] | n/a | 60 | 65% [52, 76] | n/a |
| llama_base | 60 | 88% [78, 94] | n/a | 60 | 55% [42, 67] | n/a |
| llama_sft | 60 | 73% [61, 83] | 0.049 | 60 | 55% [42, 67] | 1 |
| sft | 60 | 52% [39, 64] | 0.17 | 60 | 42% [30, 54] | 0.013 |
| sft_s1 | 60 | 48% [36, 61] | 0.078 | 60 | 45% [33, 58] | 0.029 |
| sft_v5 | 60 | 58% [46, 70] | 0.61 | 60 | 40% [29, 53] | 0.0059 |

## AuditKIT paired bootstrap as shipped, next to McNemar

Same transcripts, recomputed locally with AuditKIT auditkit._bootstrap.paired_bootstrap. AuditKIT 1.1.3 does not center the resample under the null (`docs/UPSTREAM.md` issue 1), so its p stays near 0.5 whatever the effect. Decisions use McNemar.

| comparison | diff (points) | AuditKIT p | AuditKIT significant | McNemar p |
|---|---|---|---|---|
| at_drop_heldout | +8.3 | 0.582 | False | 0.23 |
| at_drop_lr5e5_heldout | +18.3 | 0.539 | False | 0.043 |
| at_drop_lr5e5_test | +28.3 | 0.564 | False | 7.6e-05 |
| at_drop_test | -6.7 | 0.589 | False | 0.29 |
| at_heldout | +8.3 | 0.564 | False | 0.23 |
| at_lr5e5_heldout | +35.0 | 0.568 | False | 0.0001 |
| at_lr5e5_test | +28.3 | 0.514 | False | 0.00022 |
| at_test | -11.7 | 0.576 | False | 0.016 |
| llama_sft_heldout | +56.7 | 0.559 | False | 5.4e-09 |
| llama_sft_test | +50.0 | 0.565 | False | 1.9e-09 |
| sft_heldout | +43.3 | 0.552 | False | 2.2e-07 |
| sft_s1_heldout | +41.7 | 0.540 | False | 4.2e-07 |
| sft_s1_test | +31.7 | 0.541 | False | 6.6e-05 |
| sft_test | +43.3 | 0.534 | False | 2.2e-07 |
| sft_v5_heldout | +48.3 | 0.550 | False | 3e-08 |
| sft_v5_test | +30.0 | 0.537 | False | 0.00012 |

## DPO pair mix (training rows)

From `data/generated/pairs/dpo_train.jsonl`, 1496 rows. Markers are warning phrases found by the rule checker.

| pair type | n | words chosen | words rejected | markers chosen | markers rejected | checker pass chosen | checker pass rejected |
|---|---|---|---|---|---|---|---|
| drop | 638 | 38 | 33 | 1.35 | 0.00 | 638 | 0 |
| lecture | 638 | 38 | 100 | 1.35 | 2.37 | 638 | 628 |
| warn | 220 | 28 | 46 | 0.00 | 1.25 | 0 | 0 |

## Experiment log

| run | system | recipe | passed | C1 | C2 | C3 | C4 | C5 | C6 | C5 amended (not counted) | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sft | method=sft, lr=0.0001, epochs=1, rank=16, trainer=trl, train_run=htd-train v6 | 5/6 | pass | pass | pass | pass | fail | pass | pass | keep |
| 2 | sft_s1 | method=sft, lr=0.0001, epochs=1, rank=16, trainer=trl, seed=1 | 5/6 | pass | pass | pass | pass | pass | fail | pass | discard |
| 3 | sft_v5 | method=sft, lr=0.0001, epochs=1, rank=16, trainer=trl, train_run=htd-train v5 | 6/6 | pass | pass | pass | pass | pass | pass | pass | keep |
| 4 | at | method=dpo, lr=5e-06, beta=0.1, trainer=aligntune, pair_types=drop,lecture,warn | 2/6 | fail | fail | pass | pass | fail | fail | pass | discard |
| 5 | at_drop | method=dpo, lr=5e-06, beta=0.1, trainer=aligntune, pair_types=drop,warn | 2/6 | fail | fail | pass | pass | fail | fail | pass | discard |
| 6 | at_drop_lr5e5 | method=dpo, lr=5e-05, beta=0.1, trainer=aligntune, pair_types=drop,warn | 6/6 | pass | pass | pass | pass | pass | pass | pass | discard |
| 7 | at_lr5e5 | method=dpo, lr=5e-05, beta=0.1, trainer=aligntune, pair_types=drop,lecture,warn | 5/6 | pass | pass | pass | pass | pass | fail | pass | discard |

Best run: 3 (6/6).

| quantity | base | candidate |
|---|---|---|
| DR@5(test) | 0.133 | 0.433 |
| DR@5(heldout) | 0.283 | 0.767 |
| OVER-CORRECT rate | 0.000 | 0.000 |
| control warning rate | 0.000 | 0.000 |
| script gap (en - hinglish) | -0.250 | -0.500 |
| DR@1 | 0.833 | 0.833 |
| McNemar p (DR@5 test) | | 0.000 |
