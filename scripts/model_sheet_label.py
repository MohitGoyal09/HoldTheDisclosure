"""Label the H1 or H2 sheet with a model through the OpenCode CLI, in place of the owner (amendment of 2026-11-14).

python scripts/model_sheet_label.py --sheet copy --model opencode-go/kimi-k3.1-pro --out outputs/labels/copy/labels_kimi.json
python scripts/model_sheet_label.py --sheet rule --model opencode-go/glm-5.3 --out outputs/labels/rule/labels_glm.json

The model reads the human guide and the items exactly as the Markdown sheet shows them (no system names, no
similarity), in batches, and returns yes/no per question. Output is the label JSON that `copy_sheet.py score` and
`rule_sheet.py score` read (use `--all-items`). These are model labels, not human labels.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import copy_sheet  # noqa: E402
import rule_sheet  # noqa: E402

SHEETS = {"copy": (copy_sheet, (("Q1", "answers"), ("Q2", "accept"))),
          "rule": (rule_sheet, tuple((q, f) for q, f, _ in rule_sheet.QUESTIONS))}


def prompt(sheet: str, items: list[dict]) -> str:
    mod, qs = SHEETS[sheet]
    md = mod.to_md(items)
    guide, body = md.split("\n---\n", 1)
    guide = guide.split("## When you finish", 1)[0].replace("written after the colon on the", "for the")
    keys = ", ".join(f'"{q}": "yes" or "no"' for q, _ in qs)
    return ("You are labelling assistant replies for a research study. Do not use tools; answer from the text only.\n\n"
            + guide + "\n---\n" + body
            + f"\n\nReturn only a JSON array, one object per item, in order: "
              f'[{{"item": "<item id>", {keys}, "note": "<optional short note>"}}]')


def parse(out: str, qs, want: set[str]) -> dict | None:
    for m in reversed(re.findall(r"\[\s*\{.*?\}\s*\]", out, re.S)):
        try:
            rows = json.loads(m)
        except json.JSONDecodeError:
            continue
        got = {}
        for d in rows:
            if not isinstance(d, dict) or d.get("item") not in want:
                continue
            vals = {f: str(d.get(q, "")).strip().lower().rstrip(".") for q, f in qs}
            if all(v in ("yes", "no") for v in vals.values()):
                note = str(d.get("note") or "").strip()
                got[d["item"]] = {**vals, **({"note": note} if note else {})}
        if set(got) == want:
            return got
    return None


def ask(sheet: str, model: str, items: list[dict], tries: int = 3) -> dict:
    qs, want = SHEETS[sheet][1], {it["item"] for it in items}
    for _ in range(tries):
        with tempfile.TemporaryDirectory() as cwd:  # empty cwd: the agent has no files to touch
            out = subprocess.run(["opencode", "run", "--archived", "-m", model, prompt(sheet, items)],
                                 cwd=cwd, capture_output=True, text=True, timeout=900).stdout
        got = parse(out, qs, want)
        if got:
            return got
    raise RuntimeError(f"{model}: no valid labels for items {sorted(want)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", choices=tuple(SHEETS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    items = json.loads((ROOT / "outputs/labels" / a.sheet / "items.json").read_text())
    batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]
    labels: dict = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for got in ex.map(lambda b: ask(a.sheet, a.model, b), batches):
            labels.update(got)
    a.out.write_text(json.dumps({it["item"]: labels[it["item"]] for it in items}, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(labels)} items, model {a.model}")


if __name__ == "__main__":
    main()
