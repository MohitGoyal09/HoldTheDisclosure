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
