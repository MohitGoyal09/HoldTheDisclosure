"""Write the evaluation notebook for `sft_rule_s1` (rule distillation, post hoc): kaggle/eval_rule/.

python kaggle/eval_rule/make.py

Same decoding and dialogue files as every other evaluation (batch 8, 384 new tokens, plain prompt): test2
(40 per script), test and control test (20 per script). The adapter comes from the training kernel
`rockerleo/htd-sft-rule`. Scoring runs locally.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SPLITS = [("test2", 40, "test2"), ("test", 20, "test"), ("control_test", 20, "control")]

CELL = r'''import hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

SPLITS = __SPLITS__

def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)

hits = sorted((p.parents[2] for p in Path("/kaggle/input").rglob("rule_checker.py")
               if (p.parents[2] / "scripts" / "train.py").exists() and "/htd-code/" in str(p)), key=lambda p: len(p.parts))
assert hits, "code dataset not mounted"
WORK = Path("/kaggle/working/htd")
if WORK.exists():
    shutil.rmtree(WORK)
shutil.copytree(hits[0], WORK)
os.chdir(WORK)
assert "test2_shuffled" in (WORK / "src/htd/dialogues.py").read_text(), "stale code dataset"
for p in Path("/kaggle/input").rglob("hf_token.txt"):
    os.environ.setdefault("HF_TOKEN", p.read_text().strip())
assert os.environ.get("HF_TOKEN"), "attach htd-secrets"
# PEFT refuses the image's torchao 0.10.0 when loading an adapter (log, Nov 10).
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
assert importlib.util.find_spec("torchao") is None, "torchao is still installed"
print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)

ad = sorted((p.parent for p in Path("/kaggle/input").rglob("adapter_config.json") if p.parent.name == "sft_rule_s1"),
            key=lambda p: len(p.parts))
assert ad, "attach the htd-sft-rule kernel output"
ADAPTER = str(ad[0])
SHA = hashlib.sha256((ad[0] / "adapter_model.safetensors").read_bytes()).hexdigest()
print("adapter", ADAPTER, SHA[:16])

def ids(path):
    return {json.loads(x)["dialogue_id"] for x in Path(path).read_text().splitlines() if x.strip()}

Path("outputs/eval").mkdir(parents=True, exist_ok=True)
failed = []
for split, n, tag in SPLITS:
    dialogues = f"data/generated/{split}_dialogues.jsonl"
    run([sys.executable, "scripts/build_pilot.py", "--split", split, "--n-per-script", str(n), "--out", dialogues])
    out = f"outputs/eval/sft_rule_s1_{tag}.jsonl"
    try:
        run([sys.executable, "scripts/run_pilot.py", "--dialogues", dialogues, "--model", "CohereLabs/tiny-aya-fire",
             "--adapter", ADAPTER, "--model-name", "sft_rule_s1", "--variant", "plain", "--backend", "hf",
             "--device", "cuda", "--batch-size", "8", "--max-new-tokens", "384", "--out", out])
        assert ids(out) == ids(dialogues), out
        Path(out.replace(".jsonl", ".adapter.json")).write_text(json.dumps({"adapter": ADAPTER, "sha256": SHA}))
    except Exception as e:
        print(f"ARM FAILED: {out}: {e}", flush=True)
        failed.append(out)
shutil.copytree(WORK / "outputs" / "eval", "/kaggle/working/eval", dirs_exist_ok=True)
print("FAILED:", failed)
if failed:
    raise SystemExit(f"{len(failed)} stages failed: {failed}")
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": CELL.replace("__SPLITS__", repr(SPLITS)).splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "eval_rule.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(HERE / "kernel-metadata.json").write_text(json.dumps({
    "id": "rockerleo/htd-eval-rule", "title": "htd eval rule", "code_file": "eval_rule.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": ["rockerleo/htd-code", "rockerleo/htd-secrets"],
    "competition_sources": [], "kernel_sources": ["rockerleo/htd-sft-rule"], "model_sources": []}, indent=2))
print("wrote", HERE / "eval_rule.ipynb")
