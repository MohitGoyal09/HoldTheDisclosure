"""Label replies with the LLM judge.

python scripts/judge_run.py outputs/eval/*.jsonl --turn 5 --out outputs/judge/turn5.jsonl --workers 4 --model huggingface/zai-org/GLM-4.7-Flash

The judge model comes from --model or HTD_JUDGE_MODEL. Output is one JSONL row per reply,
written as soon as it is labelled, so a rerun with --resume continues.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.generate import append_record, build_llm, completed_ids  # noqa: E402
from htd.judge import JudgeError, label_reply  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def row_id(t, turn: int) -> str:
    return f"{t.model_name}|{t.variant}|{t.dialogue_id}|t{turn}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="transcript JSONL files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--turn", type=int, default=5, help="1-based turn to judge")
    ap.add_argument("--model", default=None, help="LiteLLM judge model (default: env HTD_JUDGE_MODEL)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rubric", default="v1", choices=["v1", "v2", "panel"])
    a = ap.parse_args()

    sheet = load_sheet()
    transcripts = [t for f in a.files for t in load_transcripts(f)]
    out = Path(a.out)
    done = completed_ids(out, done=lambda r: r.get("judge_label") not in (None, "ERROR")) if a.resume else set()
    tasks = []
    for t in transcripts:
        product = sheet.product(t.product_id)
        if not product.requires_disclosure or len(t.replies) < a.turn:
            continue
        rid = row_id(t, a.turn)
        if rid in done:
            continue
        tasks.append((rid, t, product))
    tasks = tasks[: a.limit] if a.limit else tasks
    if not tasks:
        print("nothing to judge")
        return

    model = a.model or os.environ.get("HTD_JUDGE_MODEL")
    if not model:
        raise SystemExit("set --model or HTD_JUDGE_MODEL")
    llm = build_llm(model)

    def work(item):
        rid, t, product = item
        try:
            history = tuple(zip(t.user_turns[: a.turn - 1], t.replies[: a.turn - 1]))
            j = label_reply(
                llm, product, sheet, t.user_turns[a.turn - 1], t.replies[a.turn - 1], t.script, history, a.rubric
            )
            label, reason = j.label, j.reason
        except JudgeError as e:
            label, reason = "ERROR", str(e)
        return {
            "dialogue_id": rid,
            "source_dialogue_id": t.dialogue_id,
            "turn": a.turn,
            "script": t.script,
            "product_id": t.product_id,
            "model": t.model_name,
            "variant": t.variant,
            "customer": t.user_turns[a.turn - 1],
            "reply": t.replies[a.turn - 1],
            "judge_label": label,
            "judge_reason": reason,
            "judge_model": model,
            "judge_rubric": a.rubric,
        }

    finished = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(work, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            rec = fut.result()
            append_record(out, rec)
            finished += 1
            if finished % 20 == 0 or finished == len(tasks):
                print(f"{finished}/{len(tasks)}", file=sys.stderr)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["judge_label"]] = counts.get(r["judge_label"], 0) + 1
    print(json.dumps({"rows": len(rows), "labels": counts}, indent=2))
    error_rate = counts.get("ERROR", 0) / max(1, len(rows))
    if error_rate > 0.05:
        raise SystemExit(f"judge error rate {error_rate:.1%} is above 5%")


if __name__ == "__main__":
    main()
