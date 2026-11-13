"""Write the corrected DPO notebook (train then evaluate in one kernel): kaggle/dpo_chat/.

python kaggle/dpo_chat/make.py

Same recipe as `align-drop-lr5e5` (drop+warn pairs, lr 5e-5, beta 0.1, rank 16, 1 epoch, seed 0) with one
changed axis: `--chat-template`, so AlignTune trains on the chat format that evaluation uses
(docs/UPSTREAM.md, item 7). Evaluation as every other run (batch 8, 384 new tokens, plain prompt) on test2
(40 per script) and test (20 per script). Scoring runs locally.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
NAME = "aligntune_dpo_drop_lr5e5_chat"
SPLITS = [("test2", 40, "test2"), ("test", 20, "test")]

CELL = r'''import hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path

NAME = __NAME__
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
assert "def chat_template_pairs" in (WORK / "src/htd/train_data.py").read_text(), "stale code dataset"
assert "--chat-template" in (WORK / "scripts/train_aligntune.py").read_text(), "stale code dataset"
for p in Path("/kaggle/input").rglob("hf_token.txt"):
    os.environ.setdefault("HF_TOKEN", p.read_text().strip())
    os.environ.setdefault("HUGGINGFACE_API_KEY", p.read_text().strip())
assert os.environ.get("HF_TOKEN"), "attach htd-secrets"
data = sorted((p for p in Path("/kaggle/input").rglob("dpo_train.jsonl") if "htd-pairs-v2" in str(p)), key=lambda p: len(p.parts))
assert data, "attach rockerleo/htd-pairs-v2"
DPO_DATA = str(data[0])
print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)

def assert_adapter(name):
    p = WORK / "outputs" / name
    assert (p / "adapter_config.json").exists(), f"{name}: adapter_config.json missing"
    w = p / "adapter_model.safetensors"
    assert w.exists() and w.stat().st_size > 1_000_000, f"{name}: adapter weights missing or empty"

def check_rows(name):
    rows = [json.loads(x) for x in open(WORK / "outputs" / f"{name}_train_chat.jsonl") if x.strip()]
    assert len(rows) == 858, f"expected 858 drop+warn rows, got {len(rows)}"
    assert {r["pair_type"] for r in rows} == {"drop", "warn"}, "wrong pair types"
    assert all(isinstance(r[k], str) for r in rows for k in ("prompt", "chosen", "rejected")), "rows are not strings"
    assert all("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" in r["prompt"] for r in rows), "chat template missing"

run([sys.executable, "-m", "pip", "install", "-q", "aligntune"])
common = ["--data", DPO_DATA, "--pair-types", "drop,warn", "--chat-template"]
run([sys.executable, "scripts/train_aligntune.py", *common, "--output", "outputs/dpo_chat_smoke", "--max-steps", "5", "--grad-accum", "2"])
assert_adapter("dpo_chat_smoke")
check_rows("dpo_chat_smoke")
print("smoke ok: 858 chat-template rows")
train = run([sys.executable, "scripts/train_aligntune.py", *common, "--output", f"outputs/{NAME}",
             "--epochs", "1", "--lr", "5e-5", "--beta", "0.1", "--rank", "16"], capture_output=True, encoding="utf-8")
print(train.stdout[-20000:]); print(train.stderr[-20000:])
assert "Mismatch between tokenized prompt" not in train.stdout + train.stderr, "TRL still sees a prompt mismatch"
assert_adapter(NAME)
check_rows(NAME)
shutil.copytree(WORK / "outputs" / NAME, f"/kaggle/working/{NAME}", dirs_exist_ok=True)
Path(f"/kaggle/working/{NAME}_train.log").write_text(train.stdout + "\n" + train.stderr)

# PEFT refuses the image's torchao 0.10.0 when loading an adapter (log, Nov 10).
run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
assert importlib.util.find_spec("torchao") is None, "torchao is still installed"
ADAPTER = str(WORK / "outputs" / NAME)
SHA = hashlib.sha256((Path(ADAPTER) / "adapter_model.safetensors").read_bytes()).hexdigest()
print("adapter", ADAPTER, SHA[:16])

def ids(path):
    return {json.loads(x)["dialogue_id"] for x in Path(path).read_text().splitlines() if x.strip()}

Path("outputs/eval").mkdir(parents=True, exist_ok=True)
failed = []
for split, n, tag in SPLITS:
    dialogues = f"data/generated/{split}_dialogues.jsonl"
    run([sys.executable, "scripts/build_pilot.py", "--split", split, "--n-per-script", str(n), "--out", dialogues])
    out = f"outputs/eval/{NAME}_{tag}.jsonl"
    try:
        run([sys.executable, "scripts/run_pilot.py", "--dialogues", dialogues, "--model", "CohereLabs/tiny-aya-fire",
             "--adapter", ADAPTER, "--model-name", NAME, "--variant", "plain", "--backend", "hf",
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
                 "source": CELL.replace("__NAME__", repr(NAME)).replace("__SPLITS__", repr(SPLITS)).splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "dpo_chat.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(HERE / "kernel-metadata.json").write_text(json.dumps({
    "id": "rockerleo/htd-dpo-chat", "title": "htd dpo chat", "code_file": "dpo_chat.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": ["rockerleo/htd-code", "rockerleo/htd-pairs-v2", "rockerleo/htd-secrets"],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
print("wrote", HERE / "dpo_chat.ipynb")
