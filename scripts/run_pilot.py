"""Replay dialogues through a chat model and save transcripts."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from htd.dialogues import load_dialogues  # noqa: E402
from htd.prompts import VARIANTS  # noqa: E402
from htd.replay import replay, save_transcripts  # noqa: E402
from htd.sheet import load_sheet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dialogues", default="data/generated/pilot_dialogues.jsonl")
    ap.add_argument("--model", default="CohereLabs/tiny-aya-fire")
    ap.add_argument("--variant", choices=list(VARIANTS), default="plain")
    ap.add_argument("--backend", choices=["hf", "fake", "api"], default="hf")
    ap.add_argument("--workers", type=int, default=4, help="parallel calls for --backend api")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--adapter", default=None, help="LoRA adapter path or HF repo id")
    ap.add_argument("--device-map", default=None, help="accelerate device map; default is one GPU")
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--model-name", default=None, help="label used in scoring (default: --model)")
    a = ap.parse_args()

    sheet = load_sheet()
    dialogues = load_dialogues(a.dialogues)
    backend = None
    if a.backend == "fake":
        from htd.backends import FakeBackend

        backend = FakeBackend(sheet=sheet)
    elif a.backend == "api":
        # A LiteLLM model string, e.g. huggingface/Qwen/Qwen2.5-72B-Instruct. Greedy where the API allows.
        from curatorkit.llm.litellm import LiteLLMBackend

        from htd.backends import APIBackend

        llm = LiteLLMBackend(model=a.model, temperature=0.0, max_tokens=a.max_new_tokens, max_retries=1, timeout=120)
        backend = APIBackend(llm, workers=a.workers)
    else:
        from htd.backends import HFChatBackend

        backend = HFChatBackend(
            a.model,
            max_new_tokens=a.max_new_tokens,
            batch_size=a.batch_size,
            load_in_4bit=a.load_in_4bit,
            hf_token=os.environ.get("HF_TOKEN"),
            adapter=a.adapter,
            device_map=a.device_map,
            device=a.device,
        )
    label = a.model_name or a.model
    try:
        transcripts = replay(dialogues, sheet, a.variant, backend.generate_batch, label)
        save_transcripts(transcripts, a.out)
        print(f"wrote {len(transcripts)} transcripts to {a.out}")
    finally:
        if hasattr(backend, "close"):
            backend.close()


if __name__ == "__main__":
    main()
