"""Ask one model whether each turn-k reply helps the customer (YES/NO). Run once per panel model.

python scripts/helpful_run.py outputs/eval/*_test.jsonl --turn 5 --model huggingface/meta-llama/Llama-3.3-70B-Instruct \
  --out outputs/judge_local/helpful_llama.jsonl --resume
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.generate import append_record, build_llm, completed_ids  # noqa: E402
from htd.helpful import build_prompt, parse_answer  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402

RETRIES = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--turn", type=int, default=5)
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    sheet = load_sheet()
    out = Path(a.out)
    done = completed_ids(out, done=lambda r: r.get("helpful") not in (None, "ERROR")) if a.resume else set()
    tasks = []
    for t in (t for f in a.files for t in load_transcripts(f)):
        rid = f"{t.model_name}|{t.variant}|{t.dialogue_id}|t{a.turn}"
        if len(t.replies) >= a.turn and rid not in done:
            tasks.append((rid, t))
    if not tasks:
        print("nothing to check")
        return
    llm = build_llm(a.model)

    def work(item):
        rid, t = item
        history = tuple(zip(t.user_turns[: a.turn - 1], t.replies[: a.turn - 1]))
        prompt = build_prompt(sheet.product(t.product_id), t.user_turns[a.turn - 1], t.replies[a.turn - 1], history)
        messages = [{"role": "system", "content": "Reply with one JSON object and nothing else."},
                    {"role": "user", "content": prompt}]
        answer, err = "ERROR", ""
        for _ in range(RETRIES + 1):
            try:
                answer = parse_answer(llm.generate(messages).text)
                break
            except (ValueError, RuntimeError) as e:
                err = str(e)
        return {"dialogue_id": rid, "model": t.model_name, "script": t.script, "product_id": t.product_id,
                "requires_disclosure": sheet.product(t.product_id).requires_disclosure,
                "helpful": answer, "error": err if answer == "ERROR" else "", "judge_model": a.model}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(work, t) for t in tasks]):
            append_record(out, fut.result())
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors = sum(r["helpful"] == "ERROR" for r in rows)
    print(json.dumps({"rows": len(rows), "yes": sum(r["helpful"] == "YES" for r in rows), "errors": errors}))
    if errors > 0.05 * len(rows):
        raise SystemExit(f"helpfulness error rate {errors / len(rows):.1%} is above 5%")


if __name__ == "__main__":
    main()
