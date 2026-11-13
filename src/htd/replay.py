"""Multi-turn replay: the model's own replies stay in the history."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from htd.dialogues import Dialogue
from htd.prompts import system_prompt
from htd.sheet import Sheet

Messages = list[dict[str, str]]
GenerateBatch = Callable[[list[Messages]], list[str]]


@dataclass(frozen=True)
class Transcript:
    dialogue_id: str
    product_id: str
    script: str
    variant: str
    model_name: str
    user_turns: tuple[str, ...]
    replies: tuple[str, ...]


def replay(
    dialogues: Sequence[Dialogue],
    sheet: Sheet,
    variant: str,
    generate_batch: GenerateBatch,
    model_name: str,
    max_turns: int = 5,
    edit: Callable[[Dialogue, str], str] | None = None,
) -> list[Transcript]:
    """`edit(dialogue, reply)` rewrites each reply before it enters the history (e.g. a post-processing rule)."""
    histories: list[Messages] = [
        [{"role": "system", "content": system_prompt(sheet.product(d.product_id), sheet, variant, d.script)}]
        for d in dialogues
    ]
    replies: list[list[str]] = [[] for _ in dialogues]
    for t in range(max_turns):
        for h, d in zip(histories, dialogues):
            h.append({"role": "user", "content": d.user_turns[t]})
        outputs = generate_batch([list(h) for h in histories])
        if len(outputs) != len(dialogues):
            raise ValueError(f"backend returned {len(outputs)} replies for {len(dialogues)} dialogues")
        if edit is not None:
            outputs = [edit(d, out) for d, out in zip(dialogues, outputs)]
        for h, r, out in zip(histories, replies, outputs):
            h.append({"role": "assistant", "content": out})
            r.append(out)
    return [
        Transcript(d.dialogue_id, d.product_id, d.script, variant, model_name, tuple(d.user_turns[:max_turns]), tuple(r))
        for d, r in zip(dialogues, replies)
    ]


def save_transcripts(transcripts: Sequence[Transcript], path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(asdict(t), ensure_ascii=False) + "\n" for t in transcripts), encoding="utf-8")


def load_transcripts(path: Path | str) -> list[Transcript]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["user_turns"] = tuple(r["user_turns"])
            r["replies"] = tuple(r["replies"])
            out.append(Transcript(**r))
    return out
