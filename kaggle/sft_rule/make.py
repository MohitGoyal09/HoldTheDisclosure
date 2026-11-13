"""Write the rule-distillation SFT notebook: kaggle/sft_rule/.

python kaggle/sft_rule/make.py

1. Base Tiny Aya Fire replays the 180 training dialogues (image libraries, as every base evaluation), and the rule
   appends each missing disclosure statement turn by turn (`scripts/rule_targets.py`).
2. `scripts/cut_pairs.py --always-recover-turn 3`, the command that built pairs v2, so only the chosen text differs.
3. SFT exactly like `sft_lr1e4_s1` (seed 1, same arguments and pinned library versions). Post hoc, not
   pre-registered. Evaluation is a separate kernel (kaggle/eval_rule).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
NAME = "sft_rule_s1"

CELL = r'''import json, os, shutil, subprocess, sys
from pathlib import Path

NAME = __NAME__

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
assert "edit: Callable" in (WORK / "src/htd/replay.py").read_text(), "stale code dataset"
assert (WORK / "scripts/rule_targets.py").exists(), "stale code dataset"
for p in Path("/kaggle/input").rglob("hf_token.txt"):
    os.environ.setdefault("HF_TOKEN", p.read_text().strip())
    os.environ.setdefault("HUGGINGFACE_API_KEY", p.read_text().strip())
assert os.environ.get("HF_TOKEN"), "attach htd-secrets"
gen = {n: sorted((p for p in Path("/kaggle/input").rglob(n) if "htd-gen-v2" in str(p)), key=lambda p: len(p.parts))
       for n in ("gen_train_v2.jsonl", "gen_control_train_v2.jsonl")}
assert all(gen.values()), "attach rockerleo/htd-gen-v2"
print(run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)

# 1. Targets: base replies with the rule applied, generated with the image libraries (as the base evaluations).
run([sys.executable, "scripts/rule_targets.py", str(gen["gen_train_v2.jsonl"][0]), str(gen["gen_control_train_v2.jsonl"][0]),
     "--out", "data/generated/gen_rule.jsonl", "--backend", "hf", "--device", "cuda", "--batch-size", "8",
     "--max-new-tokens", "384"])
recs = [json.loads(x) for x in open("data/generated/gen_rule.jsonl") if x.strip()]
ok = sum(r["status"] == "ok" for r in recs)
print({"records": len(recs), "ok": ok, "by_script": {s: sum(r["status"] == "ok" for r in recs if r["script"] == s)
                                                     for s in ("en", "hi", "hinglish")}})
assert ok >= 0.8 * len(recs), f"only {ok} of {len(recs)} records pass the rule gate"
shutil.copy("data/generated/gen_rule.jsonl", "/kaggle/working/gen_rule.jsonl")

# 2. Pairs, cut by the command that built pairs v2.
run([sys.executable, "scripts/cut_pairs.py", "data/generated/gen_rule.jsonl", "--out-dir", "data/generated/pairs",
     "--always-recover-turn", "3"])
n_sft = sum(1 for x in open("data/generated/pairs/sft_train.jsonl") if x.strip())
print("sft rows:", n_sft)
shutil.copytree("data/generated/pairs", "/kaggle/working/pairs_rule", dirs_exist_ok=True)

# 3. SFT with the library versions and arguments of sft_lr1e4_s1.
run([sys.executable, "-m", "pip", "install", "-q",
     "transformers==5.20.2", "trl==0.29.1", "peft==0.21.2", "bitsandbytes>=0.45", "accelerate>=1.0"])
import importlib.metadata as md
v = tuple(md.version(p) for p in ("transformers", "trl", "peft"))
assert v == ("5.20.2", "0.29.1", "0.21.2"), v
run([sys.executable, "scripts/train.py", "--method", "sft", "--base", "CohereLabs/tiny-aya-fire",
     "--data-dir", "data/generated/pairs", "--output", f"outputs/{NAME}",
     "--epochs", "1", "--lr", "1e-4", "--rank", "16", "--alpha", "32",
     "--batch", "1", "--grad-accum", "8", "--seed", "1"])
p = WORK / "outputs" / NAME
assert (p / "adapter_config.json").exists(), "adapter_config.json missing"
assert (p / "adapter_model.safetensors").stat().st_size > 1_000_000, "adapter weights missing or empty"
meta = json.loads((p / "train_meta.json").read_text())
assert meta["args"]["seed"] == 1 and meta["train"]["rows"] == n_sft, meta
shutil.copytree(p, f"/kaggle/working/{NAME}", dirs_exist_ok=True)
print(sorted(x.name for x in p.iterdir()))
'''

nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                 "source": CELL.replace("__NAME__", repr(NAME)).splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
(HERE / "sft_rule.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(HERE / "kernel-metadata.json").write_text(json.dumps({
    "id": "rockerleo/htd-sft-rule", "title": "htd sft rule", "code_file": "sft_rule.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True, "dataset_sources": ["rockerleo/htd-code", "rockerleo/htd-gen-v2", "rockerleo/htd-secrets"],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
print("wrote", HERE / "sft_rule.ipynb")
