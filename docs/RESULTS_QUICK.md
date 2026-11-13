> Superseded snapshot (quick test-only run with the earliest SFT adapter, 2026-11-10). Current results: `docs/RESULTS_FULL.md`, `docs/RESULTS_LLAMA.md`.

# Results

Generated from the artifacts in `outputs/quick_eval_tight` on 2026-11-10. Commit `fc3c70b`.
All numbers come from `scripts/render_results.py`; do not edit them by hand.

## Test split

| model | variant | script | n | DR@1 | DR@2 | DR@3 | DR@4 | DR@5 | DR@5 in-script | DR@5 95% CI | mean ToF | stated@T1 | kept@5 given T1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| at | plain | en | 20 | 100% | 50% | 0% | 15% | 0% | 0% | 0%-0% | 2.50 | 20 | 0% |
| at | plain | hi | 20 | 100% | 50% | 15% | 15% | 5% | 5% | 0%-15% | 2.65 | 20 | 5% |
| at | plain | hinglish | 20 | 25% | 10% | 10% | 15% | 0% | 0% | 0%-0% | 1.35 | 5 | 0% |
| base | plain | en | 20 | 100% | 50% | 20% | 25% | 0% | 0% | 0%-0% | 2.75 | 20 | 0% |
| base | plain | hi | 20 | 100% | 35% | 10% | 20% | 15% | 15% | 0%-30% | 2.55 | 20 | 15% |
| base | plain | hinglish | 20 | 50% | 35% | 35% | 30% | 25% | 25% | 5%-45% | 2.45 | 10 | 50% |
| sft | plain | en | 20 | 100% | 35% | 60% | 35% | 25% | 25% | 10%-45% | 3.15 | 20 | 25% |
| sft | plain | hi | 20 | 75% | 25% | 30% | 30% | 30% | 30% | 10%-50% | 2.60 | 15 | 40% |
| sft | plain | hinglish | 20 | 75% | 50% | 60% | 65% | 75% | 75% | 55%-95% | 3.30 | 15 | 100% |

Turn-1 to turn-5 transitions

| model | variant | script | stated both | stated then dropped | missed then stated | missed both |
|---|---|---|---|---|---|---|
| at | plain | en | 0 | 20 | 0 | 0 |
| at | plain | hi | 1 | 19 | 0 | 0 |
| at | plain | hinglish | 0 | 5 | 0 | 15 |
| base | plain | en | 0 | 20 | 0 | 0 |
| base | plain | hi | 3 | 17 | 0 | 0 |
| base | plain | hinglish | 5 | 5 | 0 | 10 |
| sft | plain | en | 5 | 15 | 0 | 0 |
| sft | plain | hi | 6 | 9 | 0 | 5 |
| sft | plain | hinglish | 15 | 0 | 0 | 5 |

## SFT vs base, test

DR@5: A = base:plain, B = sft:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 43.3% | +30.0 pts | [+16.7, +43.3] pts | 0.0001 | yes |
| en | 20 | 0.0% | 25.0% | +25.0 pts | | | |
| hi | 20 | 15.0% | 30.0% | +15.0 pts | | | |
| hinglish | 20 | 25.0% | 75.0% | +50.0 pts | | | |

## AlignTune DPO vs base, test

DR@5: A = base:plain, B = at:plain (diff = B - A)

| group | n | A | B | diff (B-A) | 95% CI | McNemar p | significant |
|---|---|---|---|---|---|---|---|
| all | 60 | 13.3% | 1.7% | -11.7 pts | [-20.0, -5.0] pts | 0.0156 | yes |
| en | 20 | 0.0% | 0.0% | +0.0 pts | | | |
| hi | 20 | 15.0% | 5.0% | -10.0 pts | | | |
| hinglish | 20 | 25.0% | 0.0% | -25.0 pts | | | |
