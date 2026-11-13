"""Train the DPO adapter through AlignTune's unified API.

python scripts/train_aligntune.py --data data/generated/pairs/dpo_train.jsonl --output outputs/aligntune_dpo --max-steps 5
python scripts/train_aligntune.py --data ... --output ... --epochs 1
python scripts/train_aligntune.py --data ... --output ... --chat-template   # DPO in the evaluation format

AlignTune routes to the TRL backend and handles the config. This file exists so the
pipeline uses CuratorKIT (data), AlignTune (post-training) and AuditKIT (evaluation),
which is the Track 1 chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.train_data import chat_template_pairs, filter_pair_types, load_jsonl  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["dpo", "sft"], default="dpo")
    ap.add_argument("--data", default=None)
    ap.add_argument("--base", default="CohereLabs/tiny-aya-fire")
    ap.add_argument("--output", default="outputs/aligntune_dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-types", default=None, help="comma list of DPO pair types to keep, e.g. drop,warn")
    ap.add_argument("--chat-template", action="store_true",
                    help="DPO only: pass chat-template strings so AlignTune does not flatten the messages (UPSTREAM item 7)")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    if a.data is None:
        a.data = "data/generated/pairs/sft_train.jsonl" if a.method == "sft" else "data/generated/pairs/dpo_train.jsonl"
    plan = {
        "base": a.base,
        "data": a.data,
        "algorithm": a.method,
        "backend": "trl",
        "output": a.output,
        "epochs": a.epochs,
        "max_steps": a.max_steps,
        "lr": 1e-4 if a.method == "sft" else a.lr,
        "beta": a.beta,
        "lora": {"r": a.rank, "alpha": a.alpha, "dropout": a.dropout},
        "max_length": a.max_length,
        "seed": a.seed,
    }
    if a.chat_template and a.method != "dpo":
        raise SystemExit("--chat-template applies to DPO only")
    out = Path(a.output)
    if a.pair_types:
        types = [t.strip() for t in a.pair_types.split(",") if t.strip()]
        rows = filter_pair_types(load_jsonl(a.data), types)
        filtered = out.parent / f"{out.name}_train.jsonl"
        filtered.parent.mkdir(parents=True, exist_ok=True)
        filtered.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        a.data = str(filtered)
        plan.update(data=a.data, pair_types=types, rows=len(rows))
    if a.chat_template:
        from transformers import AutoTokenizer

        rows = chat_template_pairs(load_jsonl(a.data), AutoTokenizer.from_pretrained(a.base))
        templated = out.parent / f"{out.name}_train_chat.jsonl"
        templated.parent.mkdir(parents=True, exist_ok=True)
        templated.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        a.data = str(templated)
        plan.update(data=a.data, chat_template=True, rows=len(rows))
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        return

    from aligntune.core.backend_factory import create_rl_trainer, create_sft_trainer
    from transformers import set_seed

    # The LoRA weights are created when the model loads, before the trainer seeds anything.
    set_seed(a.seed)

    if a.method == "sft":
        trainer = create_sft_trainer(
            model_name=a.base,
            dataset_name=a.data,
            backend="trl",
            output_dir=a.output,
            num_epochs=a.epochs,
            max_steps=a.max_steps,
            batch_size=a.batch,
            learning_rate=1e-4,
            max_seq_length=a.max_length,
            use_lora=True,
            lora_r=a.rank,
            lora_alpha=a.alpha,
            lora_dropout=a.dropout,
            gradient_accumulation_steps=a.grad_accum,
            gradient_checkpointing=True,
            fp16=True,
            optim="paged_adamw_8bit",
            seed=a.seed,
            report_to="none",
            save_strategy="no",
            quantization={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"},
        )
        trainer.train()
        saved = trainer.save_model(a.output) or a.output
        p = Path(saved)
        if not (p / "adapter_config.json").exists() or not (p / "adapter_model.safetensors").exists():
            raise SystemExit(f"AlignTune SFT saved no adapter in {p}")
        if (p / "adapter_model.safetensors").stat().st_size < 1_000_000:
            raise SystemExit(f"AlignTune SFT adapter is empty in {p}")
        print(f"saved adapter to {p}")
        return

    trainer = create_rl_trainer(
        model_name=a.base,
        dataset_name=a.data,
        algorithm="dpo",
        backend="trl",
        output_dir=a.output,
        num_epochs=a.epochs,
        max_steps=a.max_steps,
        batch_size=a.batch,
        learning_rate=a.lr,
        max_seq_length=a.max_length,
        use_lora=True,
        lora_r=a.rank,
        lora_alpha=a.alpha,
        lora_dropout=a.dropout,
        gradient_accumulation_steps=a.grad_accum,
        gradient_checkpointing=True,
        fp16=True,
        optim="paged_adamw_8bit",
        beta=a.beta,
        seed=a.seed,
        report_to="none",
        save_strategy="no",
        quantization={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"},
    )
    trainer.train()
    saved = trainer.save_model(a.output) or a.output
    p = Path(saved)
    if not (p / "adapter_config.json").exists() or not (p / "adapter_model.safetensors").exists():
        raise SystemExit(f"AlignTune DPO saved no adapter in {p}")
    if (p / "adapter_model.safetensors").stat().st_size < 1_000_000:
        raise SystemExit(f"AlignTune DPO adapter is empty in {p}")
    print(f"saved adapter to {p}")


if __name__ == "__main__":
    main()
