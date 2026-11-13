"""Train LoRA DPO or SFT on Tiny Aya with TRL.

DPO:  python scripts/train.py --method dpo --data-dir data/generated/pairs --output outputs/dpo
SFT:  python scripts/train.py --method sft --data-dir data/generated/pairs --output outputs/sft
Dry:  python scripts/train.py --method dpo --data-dir data/generated/pairs --dry-run

Heavy imports are lazy, so --dry-run and the tests work without torch.
GPU extras: pip install -e ".[gpu]" (Kaggle/Colab).

The validation slice comes from `dpo_val.jsonl` / `sft_val.jsonl`, cut by whole
dialogue in `htd.pairs`, so it never shares a dialogue with training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.backends import _dtype_kwarg  # noqa: E402
from htd.train_data import dpo_columns, filter_max_turn, load_jsonl, row_stats, sft_columns  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    ap.add_argument("--data-dir", default="data/generated/pairs")
    ap.add_argument("--output", default="outputs/dpo")
    ap.add_argument("--base", default="CohereLabs/tiny-aya-fire")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="default 5e-6 for DPO, 1e-4 for SFT")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO only")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1280)
    ap.add_argument("--max-turn", type=int, default=None, help="train only on turns up to this number")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--push-to-hub", default=None, help="repo id, e.g. Rockerleo/htd-tiny-aya-dpo")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def load_data(method: str, data_dir: Path, max_turn: int | None = None) -> tuple[list[dict], list[dict]]:
    train_path = data_dir / f"{method}_train.jsonl"
    val_path = data_dir / f"{method}_val.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} does not exist: run scripts/cut_pairs.py first")
    train = filter_max_turn(load_jsonl(train_path), max_turn)
    val = filter_max_turn(load_jsonl(val_path), max_turn) if val_path.exists() else []
    convert = dpo_columns if method == "dpo" else sft_columns
    return convert(train), convert(val)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    data_dir = Path(a.data_dir)
    train_rows, val_rows = load_data(a.method, data_dir, a.max_turn)
    stats = {
        "method": a.method,
        "base": a.base,
        "train": row_stats(train_rows, "chosen" if a.method == "dpo" else "messages"),
        "val": row_stats(val_rows, "chosen" if a.method == "dpo" else "messages"),
        "args": {k: v for k, v in vars(a).items() if k not in ("push_to_hub",)},
    }
    print(json.dumps(stats, indent=2))
    if a.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    # TRL attaches LoRA before the Trainer seeds, so seed here or the adapter init is unseeded.
    set_seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(a.base, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"token": token, "device_map": {"": 0}, _dtype_kwarg(): torch.float16}
    if a.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(a.base, **kwargs)
    if a.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft = LoraConfig(
        r=a.rank,
        lora_alpha=a.alpha,
        lora_dropout=a.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train_ds, val_ds = Dataset.from_list(train_rows), (Dataset.from_list(val_rows) if val_rows else None)
    lr = a.lr if a.lr is not None else (5e-6 if a.method == "dpo" else 1e-4)
    common = {
        "output_dir": str(out),
        "per_device_train_batch_size": a.batch,
        "gradient_accumulation_steps": a.grad_accum,
        "learning_rate": lr,
        "num_train_epochs": a.epochs,
        "max_steps": a.max_steps,
        "max_length": a.max_length,
        "logging_steps": 10,
        "save_strategy": "no",
        "fp16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "report_to": "none",
        "seed": a.seed,
        "remove_unused_columns": False,
    }

    if a.method == "dpo":
        from trl import DPOConfig, DPOTrainer

        cfg = DPOConfig(beta=a.beta, precompute_ref_log_probs=True, precompute_ref_batch_size=1, **common)
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=cfg,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft,
        )
    else:
        from trl import SFTConfig, SFTTrainer

        cfg = SFTConfig(**common)
        trainer = SFTTrainer(
            model=model,
            args=cfg,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            peft_config=peft,
        )

    # PEFT creates fresh LoRA tensors in the model dtype (bf16 here). The fp16 grad
    # scaler cannot unscale bf16, so cast the trainable tensors after the adapter exists.
    cast = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
            cast += 1
    print(f"cast {cast} trainable tensors to fp32")

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}")
    if trainable == 0:
        raise SystemExit("no trainable parameters: the adapter was not attached")

    trainer.train()
    steps = getattr(trainer.state, "global_step", 0)
    if steps <= 0:
        raise SystemExit("training finished with zero steps")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    weights = out / "adapter_model.safetensors"
    if not (out / "adapter_config.json").exists() or not weights.exists() or weights.stat().st_size < 1_000_000:
        raise SystemExit(f"adapter files missing or empty in {out}")
    print(f"saved adapter after {steps} steps to {out}")
    if a.push_to_hub:
        try:
            trainer.push_to_hub(a.push_to_hub)
            print(f"pushed adapter to {a.push_to_hub}")
        except Exception as e:  # keep the adapter on disk even if the upload fails
            print(f"push failed ({type(e).__name__}: {e}); adapter stays at {out}")
    print(f"saved adapter to {out}")


if __name__ == "__main__":
    main()
