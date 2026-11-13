"""Build the transcript replay gallery (a static page, no GPU) from the published test2 transcripts.

python scripts/build_gallery.py --data-dir <snapshot of HF Rockerleo/hold-the-disclosure> --out space

Writes space/data.json: per system and dialogue, the 5 customer turns, the replies, whether each reply states
every required disclosure (rule checker), the turn-5 reply's highest similarity to an earlier reply of the
same model, and the LLM panel's helpfulness majority at turn 5. space/index.html renders it.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.helpful import helpful_votes  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.rule_checker import check_reply  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from render_robustness import max_self_sim, system_votes  # noqa: E402

SYSTEMS = [
    ("base", "Tiny Aya Fire, base", "followup/eval/base_test2.jsonl"),
    ("base_strict", "Tiny Aya Fire, strict prompt", "followup/eval/base_strict_test2.jsonl"),
    ("sft", "SFT (main run)", "followup/eval/sft_test2.jsonl"),
    ("dpo", "DPO drop+warn, lr 5e-5", "followup/eval/at_drop_lr5e5_test2.jsonl"),
    ("qwen_strict", "Qwen2.5-72B, strict prompt", "robustness/roi_strict/qwen72b_strict_test2.jsonl"),
]
COPY_CUT = 0.9


def build(data_dir: Path) -> dict:
    sheet = load_sheet()
    rd = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
    votes = helpful_votes([rd(data_dir / f"followup/judges_test2/helpful_{n}.jsonl") for n in ("llama", "deepseek", "gptoss")])
    systems, dialogues = [], {}
    for key, label, rel in SYSTEMS:
        ts = load_transcripts(data_dir / rel)
        helpful = system_votes(ts, votes)
        systems.append({"key": key, "label": label})
        for t in ts:
            p = sheet.product(t.product_id)
            if not p.requires_disclosure:
                continue
            d = dialogues.setdefault(t.dialogue_id, {"id": t.dialogue_id, "script": t.script, "product": p.name.get("en", p.id),
                                                     "disclosures": [x.id for x in p.disclosures],
                                                     "customer": list(t.user_turns), "runs": {}})
            kept = [check_reply(r, p, sheet, t.script).all_present for r in t.replies]
            sim = round(max_self_sim(t), 3) if len(t.replies) >= 5 else None
            d["runs"][key] = {"replies": list(t.replies), "kept": kept, "sim5": sim,
                              "copied5": bool(sim is not None and sim >= COPY_CUT), "helpful5": helpful.get(t.dialogue_id)}
    return {"systems": systems, "copy_cut": COPY_CUT, "dialogues": sorted(dialogues.values(), key=lambda d: d["id"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("space"))
    a = ap.parse_args()
    data = build(a.data_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    n = len(data["dialogues"])
    for s in data["systems"]:
        runs = [d["runs"][s["key"]] for d in data["dialogues"] if s["key"] in d["runs"]]
        dr5 = sum(r["kept"][4] for r in runs if len(r["kept"]) >= 5)
        hd5 = sum(bool(r["kept"][4] and r["helpful5"]) for r in runs if len(r["kept"]) >= 5)
        print(f"{s['label']}: {len(runs)} dialogues, DR@5 {dr5}/{len(runs)}, HD@5 {hd5}/{len(runs)}")
    print(f"{n} dialogues written to {a.out / 'data.json'}")


if __name__ == "__main__":
    main()
