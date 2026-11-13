# Run registry

Every run that produced an artifact or a failure worth remembering. Status is as of 2026-11-10 11:22 IST.

## Data

| ID | What | Config | Status | Artifact |
|---|---|---|---|---|
| `data-v1` | First full training set | GLM-4.7-Flash, 40 per script, train products, old openings | Complete | 118/120 disclosure, 51/60 control, 1,290 DPO, 760 SFT |
| `data-v2` | Corrected training set | Same model, train openings only, script-enforcing gates, turn-3 recovery | Complete | 120/120 disclosure, 50/60 control, 1,496 DPO, 858 SFT, 280 recovery |

Published to `Rockerleo/hold-the-disclosure` and Kaggle `rockerleo/htd-pairs-v2`.

## Local model smoke

| ID | What | Config | Status | Result |
|---|---|---|---|---|
| `mps-smoke-1` | Base Fire, 3 dialogues | M5 Pro MPS, plain, 192 tokens | Complete | DR@5 = 0/3, long lists, no disclosure at turn 5. Not a result, n=3 |

## Pilot attempts

All pilot kernels were deleted. They were superseded by the evaluation and were blocking a GPU slot.

| ID | Kernel | Failure |
|---|---|---|
| `pilot-v1` | `htd-pilot` v1 | Unzip path wrong |
| `pilot-v2` | `htd-pilot` v2 | Dataset mount path wrong |
| `pilot-v3` | `htd-pilot` v3 | Stale dataset version |
| `pilot-v4` | `htd-pilot` v4 | Repo root one level short |
| `pilot-v5` | `htd-pilot` v5 | Missing HF token |
| `pilot-v6` | `htd-pilot` v6 | Ran, but stalled and blocked a GPU slot; deleted |

## Training

| ID | Kernel | Config | Status | Artifact |
|---|---|---|---|---|
| `train-v1` | `htd-train` v1 | 3 recipes | Failed | `ModuleNotFoundError: trl`, import before install |
| `train-v2` | `htd-train` v2 | 3 recipes, v1 pairs | Failed | `NotADirectoryError` copying the pairs file |
| `train-v3` | `htd-train` v3 | 3 recipes, v1 pairs | Failed | bf16 grad scaler at step 0 |
| `train-v4` | `htd-train` v4 | 3 recipes, v2 pairs | Failed | bf16 again, PEFT recreated bf16 LoRA |
| `train-v5` | `htd-train` v5 | 3 recipes, v2 pairs, LoRA cast | Partial | SFT adapter saved, loss 1.539. DPO OOM |
| `train-v6` | `htd-train` v6 | DPO precomputes reference log-probs, batch 1; SFT | Complete | SFT adapter 150 MB, loss 1.538. DPO5/DPO1 OOM at 13.77 GiB |
| `train-v7` | `htd-train` next | SFT first, checked cells, adapter asserts | Planned | Pending |

## AlignTune

| ID | Kernel | Config | Status | Artifact |
|---|---|---|---|---|
| `align-v1` | `htd-aligntune` v1 | DPO, TRL backend, local JSONL | Failed | `train_aligntune.py` missing from stale code dataset |
| `align-v2` | `htd-aligntune` v2 | DPO, TRL backend, v2 pairs, LoRA r16 | Complete | `aligntune_dpo`, 15 MB adapter. Scored -11.7 points vs base, p=0.016. Negative result |
| `align-sft-v1` | `htd-aligntune-sft` v1 | SFT | Failed | Stale code dataset: `--method` unrecognized. No adapter |
| `align-sft-v2` | `htd-aligntune-sft` v2 | SFT, 858 rows | Running | Pending |
| `sft2-v1` | `htd-sft2` v1 | 2 epochs | Failed | Pairs not copied into the workspace; notebook fixed |
| `dpo-short-v1` | `htd-dpo-short` v1 | turns 1-2, 768 tokens | Failed | Stale code dataset lacked `--max-turn`; marker caught it |
| `dpo-short-v2` | `htd-dpo-short` v2 | turns 1-2, 768 tokens | Dropped | Trains on turns where base already states the warning |
| `align-drop-v1` | `htd-aligntune-drop` v1 | DPO as `align-v2`, lecture pairs removed (858 drop+warn rows) | Complete, 2.3 h | `aligntune_dpo_drop`, sha256 `b1fe10d4`, final train loss 0.679, reward margin 0.044, reward accuracy 0.84 |
| `eval-arms-v1` | `htd-eval-arms` v1 | transcripts for `sft_s1`, `sft_v5` | Deleted while queued (T4 wait); eval v2 generates them | none |
| `eval-full-v2` | `htd-eval` v2 | all systems | Cancelled by owner while stuck in queue | none |
| `eval-final-v1` | `htd-eval-final` v1 | base + sft, sft_s1, sft_v5, at, at_drop on test/held-out/control, judge v1, six criteria | Complete | `outputs/kaggle-eval-final/` |
| `align-drop-lr5e5` | `htd-aligntune-drop-lr5e5` v1 | drop+warn pairs, lr 5e-5 | Complete, 2.2 h | `aligntune_dpo_drop_lr5e5`, sha256 `078a32af`, loss 0.251, margin 4.02, accuracy 0.98 |
| `second-model-v1` | `htd-second-model` v1 | Llama-3.2-3B SFT seed 1 + base/SFT eval | Complete | `llama_sft_s1`, loss 1.581; `outputs/kaggle-second-model/`, `docs/RESULTS_LLAMA.md` |
| `align-lr5e5` | `htd-aligntune-lr5e5` v1 | all pairs, lr 5e-5 | Complete, 4.2 h | `aligntune_dpo_lr5e5`, sha256 `78211fab`, loss 0.121, margin 12.55, accuracy 1.00 |
| `eval-2x2-v1` | `isthatananya/htd-eval-2x2` v1 | all systems + both lr 5e-5 DPO arms | Failed in cell 1 | no notebook internet on the account |
| `eval-2x2-v2` | `isthatananya/htd-eval-2x2` v2 | same | Complete | `outputs/kaggle-eval-2x2/`, `docs/RESULTS_FULL.md` |
| `sft-seed1-v1` | `htd-sft-seed1` v1 | SFT as `sft_lr1e4`, seed 1, pinned versions | Complete | `sft_lr1e4_s1`, 54 steps, loss 1.538, sha256 `be26c7e6` |

## Evaluation

| ID | Kernel | Systems | Status | Artifact |
|---|---|---|---|---|
| `eval-v1` | `htd-eval` v1 | Base plain, strict, third-person, AlignTune DPO | Failed at adapter load | Base arms saved. `torchao 0.10.0` incompatible with PEFT |
| `eval-v2` | `htd-quick2` | Base + AlignTune DPO + SFT, test only | Complete | SFT 43.3%, DPO 1.7% |
| `eval-v3` | `htd-eval` v1 | Base + SFT + AlignTune DPO, test, held-out, control, judge | Complete | SFT test 56.7%, held-out 71.7%, control 0%, judge over-correction 50% |

The evaluation covers test, held-out products and tactics, and control products. It runs the judge at turn 5 and the six experiment criteria.

## Baseline result, base arms only

Scored from `outputs/kaggle-eval/htd/outputs/eval/`, corrected checker.

| script | DR@1 | DR@5 | stated at T1 |
|---|---|---|---|
| en | 100% | 0% | 20/20 |
| hi | 100% | 15% | 20/20 |
| hinglish | 50% | 25% | 10/20 |

Control warning rate 0% in all scripts. Held-out DR@5: en 40%, hi 10%, hinglish 35%.

## Judge

| ID | What | Status | Result |
|---|---|---|---|
| `judge-smoke-1` | GLM judge on 6 fake replies | Complete | 6/6 labelled OVER-CORRECT, which is correct for stalling fake replies |

## Follow-up study (pre-registered 2026-11-11, `docs/PREREGISTRATION.md`)

All on the teammate Kaggle account (`isthatananya`), code dataset `htd-code` at commit `bca72d5` or later.

| ID | Kernel | What | Status | Artifact |
|---|---|---|---|---|
| `F3-s1` | `isthatananya/htd-dpo-seed1` | DPO 5e-5, seed 1: drop+warn, then all pairs; test, held-out, control transcripts | Complete (12:36 to about 20:30 IST); adapters `5465a9fe` (drop+warn), `86ba4599` (all pairs) | `eval/at_*_lr5e5_s1_*.jsonl`, adapters |
| `F3-s2` | `isthatananya/htd-dpo-seed2` | Same, seed 2 | Complete; adapters `ce8faa2a` (drop+warn), `c8500d87` (all pairs) | `eval/at_*_lr5e5_s2_*.jsonl`, adapters |
| `F1/F2-a` | `isthatananya/htd-eval-test2-a` | test2: base plain, base strict, SFT v6, SFT seed 1; test2_shuffled: base | Complete (pushed 20:45, done 22:59 IST) | `eval/*_test2*.jsonl` |
| `F1/F2-b` | `isthatananya/htd-eval-test2-b` | test2: SFT v5, DPO drop+warn 5e-5, DPO all 5e-5; test2_shuffled: SFT v6, DPO drop+warn 5e-5 | Complete (pushed 20:45, done 22:31 IST) | `eval/*_test2*.jsonl` |
| `F4` | local, `scripts/llm_label.py` | Blind sheet labelled by Kimi K3.1 Pro and GLM-5.3 through OpenCode (model labels) | Complete | `outputs/labels/label_sheet_{kimi,glm}.csv` |
| `F5` | local, HF inference API | Qwen2.5-72B-Instruct, plain prompt, test, held-out, control; panel labels and helpfulness | Complete | `outputs/roi/`, `docs/RESULTS_ROI.md` |
| `R1` | local, HF inference API | Qwen2.5-72B-Instruct, strict prompt, test, held-out, test2 (post hoc, after internal review) | Complete | `outputs/roi_strict/`, `docs/RESULTS_ROBUSTNESS.md` |

## Commands that reproduce a run

```bash
# Data
python scripts/build_pilot.py --split train --n-per-script 40 --out data/generated/train_v2.jsonl
python scripts/generate_data.py --dialogues data/generated/train_v2.jsonl --out data/generated/gen_train_v2.jsonl --workers 8
python scripts/cut_pairs.py data/generated/gen_train_v2.jsonl data/generated/gen_control_train_v2.jsonl \
  --out-dir data/generated/pairs_v2 --always-recover-turn 3

# Training
kaggle kernels push -p kaggle/train
kaggle kernels push -p kaggle/aligntune

# Evaluation
kaggle kernels push -p kaggle/eval
python scripts/render_results.py
```

## Artifact locations

| Artifact | Ref |
|---|---|
| Data | `Rockerleo/hold-the-disclosure` (Hugging Face) |
| Pairs v2 | `rockerleo/htd-pairs-v2` (private) |
| Code snapshot | `rockerleo/htd-code` (private) |
| Token | `rockerleo/htd-secrets` (private) |
| Training | `rockerleo/htd-train` |
| AlignTune | `rockerleo/htd-aligntune` |
| Evaluation | `rockerleo/htd-eval` |
| GitHub | `MohitGoyal09/HoldTheDisclosure` |

## Repetition study (pre-registered 2026-11-13, `docs/PREREGISTRATION.md`)

| ID | Kernel / job | What | Status | Artifact |
|---|---|---|---|---|
| `R1` | local, HF inference API | Helpfulness panel (Llama-3.3-70B, DeepSeek-V3, gpt-oss-120b) on 8 test2 systems, 960 rows each | Complete, 0 errors | `outputs/judge_test2/`, `docs/RESULTS_ROBUSTNESS.md` section 9 |
| `R2-data` | local, GLM-4.7-Flash | `scripts/vary_chosen.py` on gen_train_v2 and gen_control_train_v2; pairs cut with `--always-recover-turn 3` | Complete: 444 flagged, 425 rewritten, 403 below 0.7 | `rockerleo/htd-pairs-v3-varied` (private) |
| `R2-train` | `rockerleo/htd-sft-varied` | SFT as `sft_lr1e4_s1` on pairs v3 varied | Queued: owner GPU quota 30/30 h; retried by `r2_pipeline.sh` | pending |
| `R2-eval` | `rockerleo/htd-eval-varied` | `sft_varied_s1` on test2, test, held-out, control test | Queued after R2-train | pending |
| `D-chat` | `rockerleo/htd-dpo-chat` | DPO as `align-drop-lr5e5` (drop+warn, lr 5e-5, seed 0) with one changed axis: `--chat-template` (rows passed as chat-template strings, so AlignTune does not flatten them; UPSTREAM item 7). Trains, then evaluates on test2 and test in the same kernel. Not pre-registered | Queued after R2-train is pushed (`outputs/dpo_chat_pipeline.sh`) | pending |
| `H1` | artifact page | Owner labels, 50 blind test2 passes | Open | `outputs/labels/copy/` |
