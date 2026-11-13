"""Run a Kaggle notebook's code cells locally against a fake /kaggle mount, before pushing.

python scripts/nbsim.py kaggle/eval/eval.ipynb <fake_root>

<fake_root>/kaggle/input mirrors Kaggle's mounts. Heavy steps are stubbed: pip, nvidia-smi and
aligntune are skipped; run_pilot copies the base transcript for the same split (or runs the real
script with the fake backend when none exists); judge_run writes PASS rows; training scripts run
with --dry-run and get placeholder adapters. Everything else (scoring, comparisons, asserts) runs
for real, so mount, path, stale-code and missing-file bugs show up here instead of on a GPU.
"""

import importlib.metadata as md
import json
import subprocess
import sys
from pathlib import Path

nb_path, root = sys.argv[1], Path(sys.argv[2]).resolve()
real_run = subprocess.run
PINS = {"transformers": "5.20.2", "trl": "0.29.1", "peft": "0.21.2", "bitsandbytes": "0.0-sim", "torch": "0.0-sim"}
_real_version = md.version
md.version = lambda name: PINS.get(name) or _real_version(name)
BASE_FOR = {"test": "fire_test_plain", "heldout": "fire_heldout_plain", "control_test": "fire_control_plain"}


def fake_adapter(out: str) -> None:
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "adapter_config.json").write_text("{}")
    (p / "adapter_model.safetensors").write_bytes(out.encode() + b"0" * 1_100_000)


def fake_generate(cmd: list[str]):
    arg = lambda k: cmd[cmd.index(k) + 1]  # noqa: E731
    split = Path(arg("--dialogues")).name.replace("_dialogues.jsonl", "")
    base = Path("outputs/eval") / f"{BASE_FOR.get(split, '-')}.jsonl"
    if split not in BASE_FOR or not base.exists():
        return real_run([("fake" if c == "hf" else c) for c in cmd], check=True)
    rows = [json.loads(x) for x in base.read_text().splitlines() if x.strip()]
    Path(arg("--out")).write_text("".join(json.dumps(dict(r, model_name=arg("--model-name"))) + "\n" for r in rows))
    return subprocess.CompletedProcess(cmd, 0)


def fake_judge(cmd: list[str]):
    files = [c for c in cmd if c.endswith(".jsonl") and "outputs/eval" in c]
    out = Path(cmd[cmd.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in files:
        for r in (json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()):
            rows.append({"dialogue_id": f"{r['model_name']}|{r['variant']}|{r['dialogue_id']}|t5", "model": r["model_name"],
                         "variant": r["variant"], "script": r["script"], "product_id": r["product_id"],
                         "reply": r["replies"][4], "judge_label": "PASS"})
    out.write_text("".join(json.dumps(x) + "\n" for x in (rows[:1] if "--limit" in cmd else rows)))
    return subprocess.CompletedProcess(cmd, 0)


def stub_run(cmd, *a, **kw):
    cmd = [str(c) for c in cmd]
    joined = " ".join(cmd)
    print("SIM $", joined[:200], flush=True)
    if "pip" in cmd or cmd[0] in ("nvidia-smi", "aligntune"):
        return subprocess.CompletedProcess(cmd, 0, stdout="stub\n", stderr="")
    if "run_pilot.py" in joined:
        return fake_generate(cmd)
    if "judge_run.py" in joined:
        return fake_judge(cmd)
    if "train_aligntune.py" in joined or "scripts/train.py" in joined:
        kw.pop("capture_output", None)
        r = real_run(cmd + ["--dry-run"], *a, capture_output=True, text=True, **kw)
        print(r.stdout[-300:], r.stderr[-1500:])
        if r.returncode == 0 and "--output" in cmd:
            out = cmd[cmd.index("--output") + 1]
            fake_adapter(out)
            if "scripts/train.py" in joined:
                Path(out, "train_meta.json").write_text(r.stdout[r.stdout.index("{"):])
        return r
    return real_run(cmd, *a, **kw)


subprocess.run = stub_run
g = {"__name__": "__main__"}
for i, c in enumerate(json.loads(Path(nb_path).read_text())["cells"]):
    if c["cell_type"] == "code":
        print(f"=== cell {i}", flush=True)
        exec(compile("".join(c["source"]).replace('"/kaggle/', f'"{root}/kaggle/'), f"cell{i}", "exec"), g)
print("SIM OK")
