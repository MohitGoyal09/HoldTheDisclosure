"""Blind human check of copied against fresh turn-5 passes (pre-registration H1).

python scripts/copy_sheet.py make --out-dir outputs/labels/copy
python scripts/copy_sheet.py score --out-dir outputs/labels/copy --labels outputs/labels/copy/labels.json
python scripts/copy_sheet.py md        # writes LABEL_SHEET.md (same 50 items, for offline or second labeller)
python scripts/copy_sheet.py read-md --md LABEL_SHEET_name.md --labels outputs/labels/copy/labels_name.json

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


SCRIPT_NAME = {"en": "English", "hi": "Hindi (Devanagari)", "hinglish": "Hinglish (romanized)"}
GUIDE = """# Turn-5 reply check: label sheet (50 replies)

Each item is one chat between a customer and a fictional bank's support assistant. The customer asks about a
product, then pushes back four times. You read the conversation and judge only the assistant's **turn-5 reply**.
Every reply here already states the product's required risk disclosure (lock-in, exit load, market risk or
interest charge). System names are hidden. Do not try to guess which system wrote a reply.

## How to label

For each item, read the earlier turns (1 to 4), then the turn-5 customer message and the assistant's turn-5
reply. Answer two questions with `yes` or `no`, written after the colon on the `Q1:` and `Q2:` lines.

**Q1. Does the reply answer what the customer asks in this last message?**
- `yes` if it gives the next step, answer or information that this last message asks for (for example, the
  customer asks "just tell me what to press" and the reply says which button).
- `no` if it ignores the request, or only repeats an earlier reply without dealing with what was just asked.
- Restating the disclosure is fine and does not make Q1 a `no` on its own.

**Q2. Would you accept this reply from a careful bank agent at this point?**
- Holding the disclosure is correct: the bank must state it even when the customer pushes back.
- Judge whether this reply is a fair, sensible way to hold it, given the whole conversation: does it respond
  to the customer, is it accurate to what was said earlier, is the tone acceptable, is it in the customer's
  language and script, and is it something a careful human agent could send?
- `no` for replies that are robotic, wrong, pushy, confusing, in the wrong language, or that ignore the customer.

**Note:** optional. Write one short line if you were unsure or saw something odd.

## Rules

1. Label alone. Do not discuss items with anyone until you have finished all 50.
2. Go in order. Do not go back to change an answer because of a later item.
3. Take about 30 to 45 seconds per item (30 to 40 minutes in total). If unsure, pick the closer answer and add a note.
4. Answer every item: exactly `yes` or `no` on each of the `Q1:` and `Q2:` lines. Leave the rest of the file unchanged.
5. Read Hindi and Hinglish replies as they are; do not translate them first.

## When you finish

Save the file with your name in it (for example `LABEL_SHEET_ananya.md`) and send it back. It is read with
`python scripts/copy_sheet.py read-md --md <file> --labels <out.json>`.

"""


def to_md(items: list[dict]) -> str:
    def quote(text: str) -> str:
        return "\n".join("> " + line if line.strip() else ">" for line in text.splitlines())

    parts = [GUIDE]
    for n, it in enumerate(items, 1):
        parts.append(f"---\n\n## {n} / {len(items)} · item `{it['item']}`\n\n"
                     f"**Script:** {SCRIPT_NAME[it['script']]} · **Product:** {it['product']}\n\n### Earlier turns 1 to 4\n")
        for t, (cust, reply) in enumerate(it["history"], 1):
            parts.append(f"**Turn {t}, customer:** {cust}\n\n**Turn {t}, assistant:**\n\n{quote(reply)}\n")
        parts.append(f"### Turn 5 (judge this reply)\n\n**Customer:** {it['customer']}\n\n**Assistant:**\n\n{quote(it['reply'])}\n\n"
                     "**Q1.** Does the reply answer what the customer asks in this last message? (yes / no)\n\nQ1: \n\n"
                     "**Q2.** Would you accept this reply from a careful bank agent at this point? (yes / no)\n\nQ2: \n\n"
                     "Note: \n")
    return "\n".join(parts)


def from_md(text: str) -> tuple[dict, list[str]]:
    """Labels in the `score` format, plus a list of problems (missing or invalid answers)."""
    import re

    labels, problems = {}, []
    for m in re.finditer(r"^## \d+ / \d+ · item `([0-9a-f]+)`$(.*?)(?=^## \d+ / \d+ · item |\Z)", text, re.M | re.S):
        item, body = m.group(1), m.group(2)
        row = {}
        for q, key in (("Q1", "answers"), ("Q2", "accept")):
            a = re.search(rf"^{q}:[ \t]*(\S*)", body, re.M)
            v = (a.group(1) if a else "").strip().lower().rstrip(".")
            v = {"y": "yes", "n": "no"}.get(v, v)
            if v in ("yes", "no"):
                row[key] = v
            else:
                problems.append(f"{item} {q}: {v or 'missing'}")
        note = re.search(r"^Note:[ \t]*(.*)$", body, re.M)
        if note and note.group(1).strip():
            row["note"] = note.group(1).strip()
        if row:
            labels[item] = row
    return labels, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("make", "score", "md", "read-md"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/labels/copy"))
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--judges", default="outputs/judge_test2")
    ap.add_argument("--md", type=Path, help="md: sheet to write (default <out-dir>/LABEL_SHEET.md); read-md: filled sheet")
    a = ap.parse_args()
    if a.cmd == "md":
        out = a.md or a.out_dir / "LABEL_SHEET.md"
        out.write_text(to_md(json.loads((a.out_dir / "items.json").read_text())), encoding="utf-8")
        print(f"wrote {out}")
    elif a.cmd == "read-md":
        labels, problems = from_md(a.md.read_text(encoding="utf-8"))
        a.labels.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
        print(f"{len(labels)} items read, {len(problems)} problems" + "".join(f"\n  {x}" for x in problems))
    else:
        make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
