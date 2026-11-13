"""Score transcript files: python scripts/score_run.py outputs/pilot/*.jsonl --out outputs/pilot/summary.json"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.replay import load_transcripts  # noqa: E402
from htd.score import score  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="outputs/pilot/summary.json")
    ap.add_argument("--expect", type=int, default=None, help="exact number of transcripts per input file")
    a = ap.parse_args()
    files = [f for f in a.files if Path(f).resolve() != Path(a.out).resolve() and f.endswith(".jsonl")]
    if not files:
        raise SystemExit("no input transcript files")
    per_file = {}
    for f in files:
        path = Path(f)
        if not path.exists():
            raise SystemExit(f"missing transcript file: {f}")
        ts = load_transcripts(path)
        if not ts:
            raise SystemExit(f"empty transcript file: {f}")
        if a.expect is not None and len(ts) != a.expect:
            raise SystemExit(f"{f} has {len(ts)} transcripts, expected {a.expect}")
        per_file[f] = {t.dialogue_id for t in ts}
    reference = next(iter(per_file.values()))
    for f, ids in per_file.items():
        if ids != reference:
            raise SystemExit(f"dialogue ids differ between inputs: {f} has {len(ids)}, expected {len(reference)}")
    transcripts = [t for f in files for t in load_transcripts(f)]
    report = score(transcripts, load_sheet())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
