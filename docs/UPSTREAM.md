# Issues found in Lexsi tools

Found while building this project. Each one is a candidate issue or PR for the Lexsi repos. Nothing here has been filed yet.

## 1. AuditKIT: paired bootstrap significance test never rejects (bug)

- **Where:** `auditkit/_bootstrap.py`, `paired_bootstrap` (AuditKIT 1.1.3). `ModelComparison.significance` uses it.
- **What:** it resamples the observed differences and counts resamples whose mean is at least as large as the observed mean, without centering the differences under the null hypothesis. About half of all resamples then exceed the observed mean, whatever the effect size.
- **Reproduce:**
  ```python
  from auditkit._bootstrap import paired_bootstrap
  paired_bootstrap([(0.0, 1.0 if i < 90 else 0.0) for i in range(100)])
  # 0% vs 90% on 100 pairs -> p_value 0.588, significant False
  ```
- **Fix (one line):** center the resample: count `abs(mean(resample) - observed) >= abs(observed)`. Also return a percentile CI of the resampled means.
- **What we do meanwhile:** `htd.audit.compare_systems` reports AuditKIT's value unchanged, and decides with an exact McNemar test plus our own percentile bootstrap CI.

## 2. AuditKIT: HF backend does not stop at Cohere reply-end tokens

- **Where:** the `hf:` model backend.
- **What:** generation stops only on the model's `generation_config` eos. For Tiny Aya that is token 3, but every reply ends with `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` (ids 261001 and 6). The model can run past its reply.
- **What we do:** pass both markers as `stop_sequences` and trim them from the output (`htd.audit`). Our own backend stops on the ids (`htd.backends.stop_token_ids`).

## 3. CuratorKIT: DPO exporter drops per-row metadata

- **Where:** `curatorkit/exporters/dpo.py`.
- **What:** rows keep only `prompt`, `chosen`, `rejected`. Analysis fields (turn, pair type, recovery flag) are lost.
- **What we do:** `htd.pairs` writes TRL conversational rows itself, with the metadata kept.

## 4. CuratorKIT: no Indic or code-mixed generation support

- **What:** no Hindi or Hinglish handling. The only language setting is `pii_language="en"`.
- **What we do:** our own prompt templates (`src/htd/templates/`) ask for the customer's script and natural code-mixed Hinglish.

## 5. AlignTune: `IndianBFSIBench` is mostly placeholder entries

- **Where:** `src/aligntune/eval/benchmarks/indian_enterprise.py`.
- **What:** about 10 real entries. The rest are generated with the comment "Add 190 more entries... (simulated for brevity)".
- **What we do:** we do not use it.

## 6. AuditKIT: chat evaluation is single-turn by default

- **What:** `ChatAdapter` sends one user message. Multi-turn works only by passing a full `messages` list in request params.
- **What we offer:** our multi-turn replay (`htd.replay` with `htd.audit.auditkit_generate_batch`) and the BFSI disclosure probes could be contributed as an AuditKIT example.

## 7. AlignTune: DPO with chat-format prompts trains on flattened text without roles (bug)

- **Where:** `src/aligntune/data/manager.py`, `DataManager._run_curator` and
  `_flatten_chat_format_columns` (commits e821836 and 4c0872d). Found 2026-11-13, after our DPO runs.
- **What:** for task type `dpo`, a `prompt` given as a list of `{"role", "content"}` messages is
  joined into one plain string ("\n\n" between message contents), and the curated canonical rows
  replace the source rows, so the trainer never sees the message list. Roles are lost and TRL's
  `DPOTrainer` does not apply the chat template. A model then trains on a different text format
  from the one it is served with. TRL warns "Mismatch between tokenized prompt and the start of
  tokenized prompt+chosen" on every row.
- **Reproduce:** one row of our `pairs_v2/dpo_train.jsonl` at turn 4 (prompt roles: system, user,
  assistant, user, assistant, user, assistant, user) through
  `DataManager(task_type="dpo").load_dataset("data")` comes back with `prompt` as a 908-character
  `str`: the system text, then each turn separated by blank lines, with no role markers.
- **Impact on this project:** every released DPO adapter was trained on this flattened format and
  evaluated with the chat template. The SFT adapters are not affected (TRL `SFTTrainer` on the
  message lists directly).
- **Fix suggested:** keep source rows for preference task types as is done for SFT conversations, or
  apply the tokenizer's chat template to `prompt` before flattening.
- **Workaround here:** `scripts/train_aligntune.py --chat-template` passes rows already rendered with
  the tokenizer's chat template (`htd.train_data.chat_template_pairs`); plain strings pass AlignTune
  unchanged. Checked locally on all 858 drop+warn rows (2026-11-14). The corrected run trained with no
  TRL mismatch warning; its test2 results match the flattened run (DR@5 51.7% both, HD@5 18.3% against
  16.7%, p = 0.84; `docs/RESULTS_ROBUSTNESS.md`, section 13), so for our data the bug did not change the outcome.
