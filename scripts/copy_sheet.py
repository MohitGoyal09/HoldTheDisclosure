"""Blind human check of copied against fresh turn-5 passes (pre-registration H1).

python scripts/copy_sheet.py make --out-dir outputs/labels/copy
python scripts/copy_sheet.py score --out-dir outputs/labels/copy --labels outputs/labels/copy/labels.json

`make` samples DR@5 passes on test2 (25 copied, similarity 0.9 or more, and 25 fresh), shuffles them and writes
items.json (what the labeller sees: no system, no similarity) and key.json (kept local). `score` joins the
labels (item -> {"answers": "yes|no", "accept": "yes|no"}) with the key and the panel helpfulness votes.
"""

import argparse
import hashlib
import json
import random
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from htd.agreement import cohen_kappa  # noqa: E402
from htd.helpful import helpful_votes  # noqa: E402
from htd.replay import load_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402
from render_robustness import flags, max_self_sim, system_votes  # noqa: E402

FILES = [Path("outputs/followup/eval") / f"{k}_test2.jsonl"
         for k in ("base", "base_strict", "sft", "sft_s1", "sft_v5", "at_drop_lr5e5", "at_lr5e5")]
FILES.append(Path("outputs/roi_strict/qwen72b_strict_test2.jsonl"))
PER_GROUP, SEED = 25, 0


def fisher_less(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p that the first row's yes rate (a of a+b) is this low or lower."""
    n1, k, n = a + b, a + c, a + b + c + d
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(max(0, n1 + k - n), a + 1)) / comb(n, n1)


def make(a) -> None:
    sheet, pools = load_sheet(), {True: [], False: []}
    for f in FILES:
        ts = load_transcripts(f)
        for t in ts:
            if flags(sheet, [t]).get(t.dialogue_id):
                pools[max_self_sim(t) >= 0.9].append((f.name, t))
    rng, items, key = random.Random(SEED), [], {}
    for copied, pool in pools.items():
        rng.shuffle(pool)
        seen, picked = set(), []
        for script in ("en", "hi", "hinglish") * PER_GROUP:  # round-robin over scripts, one item per dialogue
            hit = next(((n, t) for n, t in pool if t.script == script and t.dialogue_id not in seen), None)
            if hit and len(picked) < PER_GROUP:
                seen.add(hit[1].dialogue_id)
                picked.append(hit)
                pool.remove(hit)
        for name, t in picked:
            iid = hashlib.sha256(f"{name}|{t.dialogue_id}".encode()).hexdigest()[:8]
            p = sheet.product(t.product_id)
            items.append({"item": iid, "script": t.script, "product": p.name.get("en", p.id),
                          "history": [[u, r] for u, r in zip(t.user_turns[:4], t.replies[:4])],
                          "customer": t.user_turns[4], "reply": t.replies[4]})
            key[iid] = {"file": name, "dialogue_id": t.dialogue_id, "copied": copied,
                        "similarity": round(max_self_sim(t), 3)}
    rng.shuffle(items)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.out_dir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(items)} items: {sum(v['copied'] for v in key.values())} copied")


def score(a) -> None:
    key = json.loads((a.out_dir / "key.json").read_text())
    labels = json.loads(a.labels.read_text())
    rd = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
    votes = helpful_votes([rd(Path(a.judges) / f"helpful_{n}.jsonl") for n in ("llama", "deepseek", "gptoss")])
    panel = {}
    for f in FILES:
        for i, v in system_votes(load_transcripts(f), votes).items():
            panel[(f.name, i)] = v
    out = {"n_labelled": len(labels)}
    for q in ("answers", "accept"):
        rows = [(key[i]["copied"], labels[i][q] == "yes") for i in labels if i in key and labels[i].get(q)]
        c = [y for cp, y in rows if cp]
        f = [y for cp, y in rows if not cp]
        out[q] = {"copied_yes": sum(c), "copied_n": len(c), "fresh_yes": sum(f), "fresh_n": len(f),
                  "fisher_p_one_sided": fisher_less(sum(c), len(c) - sum(c), sum(f), len(f) - sum(f))}
    pairs = [(labels[i]["answers"] == "yes", panel[(key[i]["file"], key[i]["dialogue_id"])]) for i in labels
             if i in key and labels[i].get("answers") and (key[i]["file"], key[i]["dialogue_id"]) in panel]
    out["panel_vs_human_answers"] = {"n": len(pairs), "agree": sum(x == y for x, y in pairs),
                                     "kappa": cohen_kappa([(str(x), str(y)) for x, y in pairs], ("True", "False"))}
    (a.out_dir / "result.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("make", "score"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/labels/copy"))
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--judges", default="outputs/judge_test2")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
