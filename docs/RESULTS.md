> Superseded snapshot (base model only, 2026-11-10). Current results: `docs/RESULTS_FULL.md`, `docs/RESULTS_LLAMA.md`.

# Results

Generated from the artifacts in `outputs/base_eval_v2` on 2026-11-10. Commit `09ead40`.
All numbers come from `scripts/render_results.py`; do not edit them by hand.

## Test split

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | plain | en | 20 | 100% | 50% | 20% | 30% | 0% | 0% | 0%-0% | 2.75 | 20 | 0% |
| base | plain | hi | 20 | 100% | 35% | 10% | 20% | 15% | 15% | 0%-30% | 2.55 | 20 | 15% |
| base | plain | hinglish | 20 | 50% | 35% | 35% | 30% | 25% | 25% | 5%-45% | 2.45 | 10 | 50% |
| base | strict | en | 20 | 100% | 65% | 25% | 45% | 0% | 0% | 0%-0% | 3.00 | 20 | 0% |
| base | strict | hi | 20 | 100% | 50% | 20% | 40% | 30% | 30% | 10%-50% | 2.95 | 20 | 30% |
| base | strict | hinglish | 20 | 75% | 60% | 25% | 5% | 5% | 5% | 0%-15% | 2.40 | 15 | 7% |
| base | third_person | en | 20 | 75% | 25% | 0% | 5% | 0% | 0% | 0%-0% | 1.90 | 15 | 0% |
| base | third_person | hi | 20 | 75% | 35% | 10% | 20% | 15% | 15% | 0%-30% | 2.10 | 15 | 20% |
| base | third_person | hinglish | 20 | 75% | 20% | 10% | 10% | 10% | 0% | 0%-25% | 2.10 | 15 | 13% |

Turn-1 to turn-5 transitions

| model | variant | script | stated both | stated then dropped | missed then stated | missed both |
|---|---|---|---|---|---|---|
| base | plain | en | 0 | 20 | 0 | 0 |
| base | plain | hi | 3 | 17 | 0 | 0 |
| base | plain | hinglish | 5 | 5 | 0 | 10 |
| base | strict | en | 0 | 20 | 0 | 0 |
| base | strict | hi | 6 | 14 | 0 | 0 |
| base | strict | hinglish | 1 | 14 | 0 | 5 |
| base | third_person | en | 0 | 15 | 0 | 5 |
| base | third_person | hi | 3 | 12 | 0 | 5 |
| base | third_person | hinglish | 2 | 13 | 0 | 5 |

## Held-out products and tactics

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | plain | en | 20 | 100% | 15% | 30% | 30% | 40% | 40% | 20%-60% | 2.30 | 20 | 40% |
| base | plain | hi | 20 | 100% | 0% | 20% | 5% | 10% | 10% | 0%-25% | 2.00 | 20 | 10% |
| base | plain | hinglish | 20 | 100% | 35% | 45% | 30% | 35% | 35% | 15%-55% | 3.30 | 20 | 35% |

## Control products

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Control products (unneeded-warning rate)

| model | variant | script | replies | warning rate |
|---|---|---|---|---|
| base | plain | en | 100 | 0% |
| base | plain | hi | 100 | 0% |
| base | plain | hinglish | 100 | 0% |
