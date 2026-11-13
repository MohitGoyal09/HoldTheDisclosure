"""Rewrite chosen replies that repeat an earlier chosen reply of the same dialogue (the "varied" SFT ablation).

python scripts/vary_chosen.py data/generated/gen_train_v2.jsonl --out data/generated/gen_train_v3_varied.jsonl \
  --model huggingface/zai-org/GLM-4.7-Flash --workers 8

Only `chosen` at turns 2 to 5 changes. A reply is rewritten when its highest `difflib` ratio to the earlier chosen
replies is at or above --cut. A rewrite must pass the same rule and length gates as generation and fall below
--cut; after the last attempt the least similar passing rewrite below --keep is used, else the original is kept.
Turn 1, drop, lecture, warn and the customer turns are untouched, so pairs cut from the output differ from
pairs v2 only in the chosen text.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.generate import (  # noqa: E402
    CandidateError,
    _g1_rule,
    _g2_length,
    build_llm,
    build_prompt,
    request_candidates,
)
from htd.sheet import load_sheet  # noqa: E402

ATTEMPTS = 4
EXTRA = """
IMPORTANT for this turn: the assistant's earlier replies are in the conversation above. Write "chosen" as a NEW
reply to the customer's message now: first answer what this message asks, in different words and sentence
structure from the earlier replies. Do not copy or lightly edit an earlier sentence. Still state every required
disclosure, in new wording. Return the same JSON object as before."""


def max_sim(reply: str, earlier: list[str]) -> float:
    return max(SequenceMatcher(None, reply, e, autojunk=False).ratio() for e in earlier)


def vary_record(rec: dict, sheet, llm, cut: float, keep: float) -> dict:
    if rec["status"] != "ok":
        return rec
    product = sheet.product(rec["product_id"])
    chosen, log = list(rec["chosen"]), []
    for t in range(1, len(chosen)):
        before = max_sim(chosen[t], chosen[:t])
        if before < cut:
            continue
        prompt = build_prompt(product, sheet, rec["script"], tuple(rec["user_turns"]), chosen[:t]) + EXTRA
        best, best_sim = None, keep
        for _ in range(ATTEMPTS):
            try:
                cand = request_candidates(llm, prompt, ("chosen",))["chosen"]
            except (CandidateError, RuntimeError, ValueError):
                continue
            if _g1_rule("chosen", cand, product, sheet, rec["script"]) or _g2_length("chosen", cand, cand):
                continue
            s = max_sim(cand, chosen[:t])
            if s < best_sim:
                best, best_sim = cand, s
            if s < cut:
                break
        log.append({"turn": t + 1, "before": round(before, 3), "after": round(best_sim, 3) if best else None})
        if best is not None:
            chosen[t] = best
    return {**rec, "chosen": chosen, "chosen_original": rec["chosen"], "vary_log": log}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generated", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cut", type=float, default=0.7)
    ap.add_argument("--keep", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sheet, llm = load_sheet(), build_llm(a.model)
    records = [json.loads(x) for f in a.generated for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        out = list(ex.map(lambda r: vary_record(r, sheet, llm, a.cut, a.keep), records))
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    logs = [x for r in out for x in r.get("vary_log", [])]
    print(json.dumps({"records": len(out), "flagged": len(logs), "rewritten": sum(x["after"] is not None for x in logs),
                      "below_cut": sum(x["after"] is not None and x["after"] < a.cut for x in logs)}))


if __name__ == "__main__":
    main()
