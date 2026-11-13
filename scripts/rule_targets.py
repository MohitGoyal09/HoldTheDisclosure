"""Rule-distillation targets: base Tiny Aya replies on the training dialogues, with the rule applied turn by turn.

python scripts/rule_targets.py data/generated/gen_train_v2.jsonl data/generated/gen_control_train_v2.jsonl \
  --out data/generated/gen_rule.jsonl --backend hf --device cuda

Each record keeps its customer turns, product and drop/lecture/warn replies; only `chosen` is replaced by the
base model's own reply plus every disclosure statement the rule checker finds missing (`scripts/rule_baseline.py`).
The edited reply stays in the history, so later turns see what a deployed filter would have sent. A record whose
edited reply still fails the checker (an explicit dismissal) or answers in the wrong script gets status
`rule_fail` and is left out by `scripts/cut_pairs.py`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.dialogues import Dialogue  # noqa: E402
from htd.replay import replay  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from rule_baseline import append_missing  # noqa: E402


def rule_records(records: list[dict], generate_batch, sheet) -> list[dict]:
    ok = [r for r in records if r.get("status") == "ok"]
    ds = [Dialogue(r["dialogue_id"], r["split"], r["product_id"], r["script"], tuple(r["user_turns"]), ()) for r in ok]
    edit = lambda d, reply: append_missing(reply, sheet.product(d.product_id), sheet, d.script)  # noqa: E731
    out = []
    for r, t in zip(ok, replay(ds, sheet, "plain", generate_batch, "base_rule", edit=edit)):
        p = sheet.product(r["product_id"])
        checks = [check_reply(x, p, sheet, r["script"]) for x in t.replies]
        good = all(c.script_ok and (c.all_present or not p.requires_disclosure) for c in checks)
        out.append({**r, "chosen": list(t.replies), "generator_model": "tiny-aya-fire base + rule",
                    "status": "ok" if good else "rule_fail"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("generated", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    ap.add_argument("--backend", choices=["hf", "fake"], default="hf")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    a = ap.parse_args()
    sheet = load_sheet()
    records = [json.loads(x) for f in a.generated for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.backend == "fake":
        from htd.backends import FakeBackend
        backend = FakeBackend(sheet=sheet)
    else:
        from htd.backends import HFChatBackend
        backend = HFChatBackend(a.model, max_new_tokens=a.max_new_tokens, batch_size=a.batch_size,
                                hf_token=os.environ.get("HF_TOKEN"), device=a.device)
    try:
        out = rule_records(records, backend.generate_batch, sheet)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    fails = sum(r["status"] != "ok" for r in out)
    print(json.dumps({"records": len(out), "rule_fail": fails, "out": a.out}))


if __name__ == "__main__":
    main()
