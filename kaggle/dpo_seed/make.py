"""Write the two DPO seed-replicate notebooks (pre-registration F3): kaggle/dpo_seed/s{1,2}/.

python kaggle/dpo_seed/make.py

Each notebook trains DPO at lr 5e-5 for drop+warn, then all pairs, with one seed, and generates
test, held-out and control transcripts for each adapter. Scoring and judging run locally.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r'''import hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

SEED = __SEED__

def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)

def find_repo():
    hits = [p for p in Path("/kaggle/input").rglob("rule_checker.py")
            if p.exists() and (p.parents[2] / "scripts" / "train.py").exists()]
    exact = [p for p in hits if "/htd-code/" in str(p)]
    hits = exact or hits
    hits.sort(key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit("code dataset not mounted")
    return hits[0].parents[2]

WORK = Path("/kaggle/working/htd")
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(find_repo(), WORK)
os.chdir(WORK)
sys.path.insert(0, str(WORK / "src"))

MARKERS = {
    "scripts/train_aligntune.py": "set_seed(a.seed)",
    "src/htd/train_data.py": "def filter_pair_types",
    "src/htd/dialogues.py": "test2_shuffled",
}
for rel, marker in MARKERS.items():
    assert marker in (WORK / rel).read_text(encoding="utf-8"), f"stale code dataset: {rel} lacks {marker!r}"
print("code markers ok")

for p in Path("/kaggle/input").rglob("hf_token.txt"):
    os.environ.setdefault("HF_TOKEN", p.read_text().strip())
assert os.environ.get("HF_TOKEN"), "attach htd-secrets"

hits = sorted(Path("/kaggle/input").rglob("dpo_train.jsonl"), key=lambda p: len(p.parts))
assert hits, "attach htd-pairs-v2"
DPO_DATA = hits[0]
assert hashlib.sha256(DPO_DATA.read_bytes()).hexdigest().startswith("f3b9ca4e7617f7d9"), "unexpected DPO pairs"
print("DPO data:", DPO_DATA)

run([sys.executable, "-m", "pip", "install", "-q", "aligntune"])
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
assert importlib.util.find_spec("torchao") is None, "torchao is still installed"
print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)

FIRE = "CohereLabs/tiny-aya-fire"
SPLITS = {"test": "test", "heldout": "heldout", "control_test": "control"}
for split in SPLITS:
    run([sys.executable, "scripts/build_pilot.py", "--split", split, "--n-per-script", "20",
         "--out", f"data/generated/{split}_dialogues.jsonl"])
Path("outputs/eval").mkdir(parents=True, exist_ok=True)

def ids(path):
    return {json.loads(x)["dialogue_id"] for x in Path(path).read_text().splitlines() if x.strip()}

def sha(adapter):
    return hashlib.sha256((Path(adapter) / "adapter_model.safetensors").read_bytes()).hexdigest()

def train(out, pair_types, extra=()):
    run([sys.executable, "scripts/train_aligntune.py", "--data", str(DPO_DATA), "--pair-types", pair_types,
         "--output", out, "--epochs", "1", "--lr", "5e-5", "--beta", "0.1", "--rank", "16",
         "--seed", str(SEED), *extra])
    w = Path(out) / "adapter_model.safetensors"
    assert (Path(out) / "adapter_config.json").exists() and w.exists() and w.stat().st_size > 1_000_000, out

def generate(label, adapter, split, out, limit_file=None):
    dialogues = limit_file or f"data/generated/{split}_dialogues.jsonl"
    run([sys.executable, "scripts/run_pilot.py", "--dialogues", dialogues, "--model", FIRE,
         "--adapter", adapter, "--model-name", label, "--variant", "plain", "--backend", "hf",
         "--device", "cuda", "--batch-size", "8", "--max-new-tokens", "384", "--out", out])
    assert ids(out) == ids(dialogues), out
'''

SMOKE = r'''# Fail fast: a 5-step training and a 3-dialogue generation before the hours-long runs.
train("outputs/smoke", "drop,warn", ("--max-steps", "5", "--grad-accum", "2"))
lines = Path("data/generated/test_dialogues.jsonl").read_text().splitlines()[:3]
Path("data/generated/smoke_dialogues.jsonl").write_text("\n".join(lines) + "\n")
generate("smoke", "outputs/smoke", "test", "outputs/smoke_test.jsonl", "data/generated/smoke_dialogues.jsonl")
print("smoke ok")
'''

MAIN = r'''# Pre-registration F3: drop+warn first (the 6/6 recipe), then all pairs.
RUNS = {f"at_drop_lr5e5_s{SEED}": "drop,warn", f"at_lr5e5_s{SEED}": "drop,lecture,warn"}
ROWS = {"drop,warn": 858, "drop,lecture,warn": 1496}
failed, done = [], []
for label, types in RUNS.items():
    out = f"outputs/{label}"
    try:
        train(out, types)
        rows = sum(1 for x in open(f"outputs/{label}_train.jsonl") if x.strip())
        assert rows == ROWS[types], f"{label}: {rows} training rows"
        digest = sha(out)
        print(label, "sha256", digest, flush=True)
        shutil.copytree(out, f"/kaggle/working/adapters/{label}", dirs_exist_ok=True)
    except Exception as e:
        print(f"ARM FAILED: {label} training: {e}", flush=True)
        failed.append(label)
        continue
    for split, tag in SPLITS.items():
        dest = f"outputs/eval/{label}_{tag}.jsonl"
        try:
            generate(label, out, split, dest)
            Path(f"outputs/eval/{label}_{tag}.adapter.json").write_text(json.dumps({"adapter": out, "sha256": digest}))
            done.append(f"{label}_{tag}")
        except Exception as e:
            print(f"ARM FAILED: {label}_{tag}: {e}", flush=True)
            failed.append(f"{label}_{tag}")
print("done:", done)
'''

SAVE = r'''shutil.copytree(WORK / "outputs" / "eval", "/kaggle/working/eval", dirs_exist_ok=True)
for p in (WORK / "outputs").glob("*_train.jsonl"):
    shutil.copy(p, "/kaggle/working/")
print("FAILED:", failed)
if failed:
    raise SystemExit(f"{len(failed)} stages failed: {failed}")
'''


def notebook(seed: int) -> dict:
    cells = [SETUP.replace("__SEED__", str(seed)), SMOKE, MAIN, SAVE]
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


for seed in (1, 2):
    d = HERE / f"s{seed}"
    d.mkdir(exist_ok=True)
    (d / "dpo_seed.ipynb").write_text(json.dumps(notebook(seed), indent=1), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"isthatananya/htd-dpo-seed{seed}", "title": f"htd dpo seed{seed}", "code_file": "dpo_seed.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["isthatananya/htd-code", "isthatananya/htd-secrets", "isthatananya/htd-pairs-v2"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
print("wrote kaggle/dpo_seed/s1 and s2")
