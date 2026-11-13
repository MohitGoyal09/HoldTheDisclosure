"""Write the two test2 evaluation notebooks (pre-registration F1, F2): kaggle/eval_test2/{a,b}/.

python kaggle/eval_test2/make.py

F1: every system on `test2` (120 dialogues, new pushback lines, fixed order).
F2: base, SFT v6 and DPO drop+warn 5e-5 on `test2_shuffled` (same dialogues, Williams order).
The work is split over two kernels so both GPU slots run at once. Scoring runs locally.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

# (label, adapter key or None for base, prompt variant, split)
JOBS = {
    "a": [("base", None, "plain", "test2"), ("base", None, "strict", "test2"), ("sft", "sft", "plain", "test2"),
          ("sft_s1", "sft_s1", "plain", "test2"), ("base", None, "plain", "test2_shuffled")],
    "b": [("sft_v5", "sft_v5", "plain", "test2"), ("at_drop_lr5e5", "at_drop_lr5e5", "plain", "test2"),
          ("at_lr5e5", "at_lr5e5", "plain", "test2"), ("sft", "sft", "plain", "test2_shuffled"),
          ("at_drop_lr5e5", "at_drop_lr5e5", "plain", "test2_shuffled")],
}

SETUP = r'''import hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

JOBS = __JOBS__

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
assert "test2_shuffled" in (WORK / "src/htd/dialogues.py").read_text(), "stale code dataset"
# The reviewed seed lines must be the ones generated from (sha of the test2 lines, fixed before the run).
lines = [x for x in (WORK / "data/seeds/pushback_seeds.jsonl").read_text(encoding="utf-8").splitlines() if '"test2"' in x]
assert len(lines) == 16, f"expected 16 test2 seed lines, got {len(lines)}"
SEED_SHA = hashlib.sha256("\n".join(lines).encode()).hexdigest()
print("test2 seed sha", SEED_SHA)
assert SEED_SHA.startswith("__SEEDSHA__"), "test2 seed lines differ from the reviewed version"

for p in Path("/kaggle/input").rglob("hf_token.txt"):
    os.environ.setdefault("HF_TOKEN", p.read_text().strip())
assert os.environ.get("HF_TOKEN"), "attach htd-secrets"
# PEFT refuses the image's torchao 0.10.0 when loading an adapter (log, Nov 10).
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
assert importlib.util.find_spec("torchao") is None, "torchao is still installed"
print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)

for split in ("test2", "test2_shuffled"):
    run([sys.executable, "scripts/build_pilot.py", "--split", split, "--n-per-script", "40",
         "--out", f"data/generated/{split}_dialogues.jsonl"])
Path("outputs/eval").mkdir(parents=True, exist_ok=True)

def ids(path):
    return {json.loads(x)["dialogue_id"] for x in Path(path).read_text().splitlines() if x.strip()}

def adapter_path(name):
    hits = [p.parent for p in Path("/kaggle/input").rglob("adapter_config.json") if p.parent.name == name]
    hits.sort(key=lambda p: len(p.parts))
    return str(hits[0]) if hits else None

def dataset_adapter(slug):
    hits = [p.parent for p in Path("/kaggle/input").rglob("adapter_config.json") if f"/{slug}/" in str(p)]
    hits.sort(key=lambda p: len(p.parts))
    return str(hits[0]) if hits else None

FOUND = {
    "sft": adapter_path("sft_lr1e4"),
    "sft_s1": adapter_path("sft_lr1e4_s1") or dataset_adapter("htd-adapter-sft-s1"),
    "sft_v5": adapter_path("sft_lr1e4_v5") or dataset_adapter("htd-sft-v5"),
    "at_drop_lr5e5": adapter_path("aligntune_dpo_drop_lr5e5"),
    "at_lr5e5": adapter_path("aligntune_dpo_lr5e5"),
}
EXPECTED = {"sft": "85162fabffb954a3", "sft_v5": "56cc24690269ab4a", "sft_s1": "be26c7e63c7a51bc",
            "at_drop_lr5e5": "078a32af7f419a21", "at_lr5e5": "78211fab448b7ace"}
SHA = {}
for key in {k for _, k, _, _ in JOBS if k}:
    assert FOUND.get(key), f"adapter {key} not mounted"
    SHA[key] = hashlib.sha256((Path(FOUND[key]) / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert SHA[key].startswith(EXPECTED[key]), f"{key}: sha {SHA[key][:16]} != {EXPECTED[key]}"
print("adapters ok:", {k: v[:12] for k, v in SHA.items()})
'''

MAIN = r'''FIRE = "CohereLabs/tiny-aya-fire"
TAG = {"test2": "test2", "test2_shuffled": "test2shuf"}
failed, done = [], []
for label, key, variant, split in JOBS:
    name = label if variant == "plain" else f"{label}_{variant}"
    out = f"outputs/eval/{name}_{TAG[split]}.jsonl"
    cmd = [sys.executable, "scripts/run_pilot.py", "--dialogues", f"data/generated/{split}_dialogues.jsonl",
           "--model", FIRE, "--model-name", label, "--variant", variant, "--backend", "hf", "--device", "cuda",
           "--batch-size", "8", "--max-new-tokens", "384", "--out", out]
    if key:
        cmd[cmd.index("--model-name"):cmd.index("--model-name")] = ["--adapter", FOUND[key]]
    try:
        run(cmd)
        assert ids(out) == ids(f"data/generated/{split}_dialogues.jsonl"), out
        Path(out.replace(".jsonl", ".adapter.json")).write_text(
            json.dumps({"adapter": FOUND.get(key), "sha256": SHA.get(key), "seed_sha": SEED_SHA}))
        done.append(out)
    except Exception as e:
        print(f"ARM FAILED: {out}: {e}", flush=True)
        failed.append(out)
print("done:", done)
shutil.copytree(WORK / "outputs" / "eval", "/kaggle/working/eval", dirs_exist_ok=True)
print("FAILED:", failed)
if failed:
    raise SystemExit(f"{len(failed)} stages failed: {failed}")
'''


def seed_sha() -> str:
    import hashlib

    path = HERE.parents[1] / "data" / "seeds" / "pushback_seeds.jsonl"
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if '"test2"' in x]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def notebook(jobs: list) -> dict:
    setup = SETUP.replace("__JOBS__", repr(jobs)).replace("__SEEDSHA__", seed_sha())
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in (setup, MAIN)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


for part, jobs in JOBS.items():
    d = HERE / part
    d.mkdir(exist_ok=True)
    (d / "eval_test2.ipynb").write_text(json.dumps(notebook(jobs), indent=1), encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"isthatananya/htd-eval-test2-{part}", "title": f"htd eval test2 {part}", "code_file": "eval_test2.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["isthatananya/htd-code", "isthatananya/htd-secrets", "isthatananya/htd-adapters"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }, indent=2) + "\n")
print("wrote kaggle/eval_test2/a and b; test2 seed sha", seed_sha())
