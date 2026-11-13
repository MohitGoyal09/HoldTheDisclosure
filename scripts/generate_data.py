"""Generate chosen/drop/lecture/warn replies for dialogues.

Offline:  python scripts/generate_data.py --dialogues data/generated/pilot_dialogues.jsonl --fake --out data/generated/gen_pilot.jsonl
Real:     HTD_GENERATOR_MODEL=huggingface/google/gemma-3-27b-it python scripts/generate_data.py --dialogues X.jsonl --out data/generated/gen_X.jsonl
          (needs `pip install curatorkit[generation]` and HF_TOKEN or the provider's API key)

The output is written one record at a time, so a crash or a rate limit keeps the finished
dialogues. Rerun with --resume to continue. --workers runs dialogues in parallel; turns
inside one dialogue stay sequential because each turn needs the model's own history.

Control products get three turns of budget: the control ladder is part of the split.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curatorkit.manifest import build_provenance, dataset_input  # noqa: E402

from htd.dialogues import load_dialogues  # noqa: E402
from htd.generate import (  # noqa: E402
    GENERATOR_MODEL_ENV,
    JUDGE_MODEL_ENV,
    FakeGeneratorLLM,
    append_record,
    build_llm,
    completed_ids,
    gate_report,
    generate_dialogue,
    reward_judge,
)
from htd.sheet import load_sheet  # noqa: E402


def _read_all(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, help="dialogues JSONL")
    ap.add_argument("--out", required=True, help="output JSONL, e.g. data/generated/gen_X.jsonl")
    ap.add_argument("--model", default=None, help=f"LiteLLM model string (default: env {GENERATOR_MODEL_ENV})")
    ap.add_argument("--fake", action="store_true", help="offline deterministic generator, no API calls")
    ap.add_argument("--limit", type=int, default=None, help="only the first N dialogues")
    ap.add_argument("--judge", action="store_true", help=f"enable the g3 judge (CuratorKIT RewardGate, model from env {JUDGE_MODEL_ENV})")
    ap.add_argument("--workers", type=int, default=1, help="dialogues generated in parallel")
    ap.add_argument("--resume", action="store_true", help="skip dialogue ids already in the output file")
    ap.add_argument("--fresh", action="store_true", help="delete the output file before writing")
    a = ap.parse_args()

    sheet = load_sheet()
    dialogues = load_dialogues(a.dialogues)[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.fresh and out.exists():
        out.unlink()
    done = completed_ids(out) if a.resume else set()
    todo = [d for d in dialogues if d.dialogue_id not in done]
    if done:
        print(f"{len(done)} dialogues already done; {len(todo)} to go", file=sys.stderr)
    if not todo:
        print("nothing to do", file=sys.stderr)
        return

    llm = FakeGeneratorLLM(sheet) if a.fake else build_llm(a.model)
    judge = None
    if a.judge:
        judge = reward_judge(FakeGeneratorLLM(sheet) if a.fake else build_llm(os.environ.get(JUDGE_MODEL_ENV)))

    finished = 0

    def work(d):
        return generate_dialogue(d, sheet, llm, judge)

    if a.workers <= 1:
        for d in todo:
            append_record(out, work(d))
            finished += 1
            print(f"{finished}/{len(todo)} dialogues", file=sys.stderr)
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futures = {ex.submit(work, d): d for d in todo}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # never lose the whole run to one dialogue
                    rec = {
                        "dialogue_id": d.dialogue_id,
                        "split": d.split,
                        "product_id": d.product_id,
                        "script": d.script,
                        "requires_disclosure": sheet.product(d.product_id).requires_disclosure,
                        "user_turns": list(d.user_turns),
                        "chosen": [],
                        "drop": None,
                        "lecture": None,
                        "warn": None,
                        "gate_log": [],
                        "generator_model": getattr(llm, "model", ""),
                        "status": "failed",
                        "failure_reason": f"{type(e).__name__}: {e}",
                    }
                append_record(out, rec)
                finished += 1
                print(f"{finished}/{len(todo)} dialogues ({d.dialogue_id})", file=sys.stderr)

    from htd.generate import dedupe_records

    records = dedupe_records(_read_all(out))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    report = gate_report(records)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    provenance = build_provenance(
        "htd-generate",
        inputs=[dataset_input(a.dialogues)],
        params={"generator_model": llm.model, "fake": a.fake, "judge": a.judge, "limit": a.limit},
    )
    out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(records)} records to {out}: {report['ok']} ok, {report['failed']} failed")
    print(f"gate failures: {report['gate_failures'] or 'none'}")


if __name__ == "__main__":
    main()
