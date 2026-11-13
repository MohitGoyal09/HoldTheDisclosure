"""Generate good and bad assistant replies per dialogue turn, behind quality gates.

LLM calls go through CuratorKIT's `BaseLLM.generate(messages)` (a plain completion call
with our own prompt). The real backend is `curatorkit.llm.litellm.LiteLLMBackend`, built
lazily from a LiteLLM model string (env `HTD_GENERATOR_MODEL`). Tests inject a fake `BaseLLM`.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from string import Template
from typing import Any

from curatorkit.llm.base import BaseLLM, LLMResponse

from htd.dialogues import Dialogue
from htd.rule_checker import check_reply
from htd.sheet import Product, Sheet

TEMPLATE_DIR = Path(__file__).parent / "templates"
GENERATOR_MODEL_ENV = "HTD_GENERATOR_MODEL"
JUDGE_MODEL_ENV = "HTD_JUDGE_MODEL"
TURNS = 5
MAX_CHOSEN_WORDS = 80
LECTURE_MIN_RATIO = 1.5
MAX_REGENERATIONS = 2  # extra attempts per turn after the first
MAX_JSON_RETRIES = 2  # extra LLM calls when the reply is not valid JSON
DISCLOSURE_KINDS = ("chosen", "drop", "lecture")
CONTROL_KINDS = ("chosen", "warn")

# (kind, reply, customer message) -> True/False or (ok, reason). Reserved for an LLM judge.
Judge = Callable[[str, str, str], "bool | tuple[bool, str]"]


class CandidateError(ValueError):
    """The generator did not return usable candidates."""


def _read(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def kinds_for(product: Product) -> tuple[str, ...]:
    return DISCLOSURE_KINDS if product.requires_disclosure else CONTROL_KINDS


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------- prompt

def _format_history(user_turns: tuple[str, ...], chosen: list[str]) -> str:
    if not chosen:
        return "(none: this is the customer's first message)"
    lines = []
    for user, reply in zip(user_turns, chosen):
        lines += [f"Customer: {user}", f"Assistant: {reply}"]
    return "\n".join(lines)


def build_prompt(
    product: Product, sheet: Sheet, script: str, user_turns: tuple[str, ...], chosen: list[str]
) -> str:
    turn = len(chosen) + 1
    name = "disclosure.txt" if product.requires_disclosure else "control.txt"
    statements = "\n".join(f"- {d.statement[script]}" for d in product.disclosures)
    return Template(_read(name)).safe_substitute(
        bank=sheet.bank,
        product_id=product.id,
        script=script,
        turn=turn,
        facts=product.facts_for_prompt,
        statements=statements,
        language_name=json.loads(_read("language_names.json"))[script],
        language_rule=json.loads(_read("language_rules.json"))[script],
        history=_format_history(user_turns, chosen),
        user_turn=user_turns[turn - 1],
    )


# ---------------------------------------------------------------- LLM + JSON

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_candidates(text: str, kinds: tuple[str, ...]) -> dict[str, str]:
    """Parse the generator's JSON reply. Raise CandidateError if it is unusable."""
    body = _FENCE.sub("", text.strip())
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start == -1 or end <= start:
            raise CandidateError("no JSON object in reply") from None
        try:
            obj = json.loads(body[start : end + 1])
        except json.JSONDecodeError as e:
            raise CandidateError(f"bad JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise CandidateError("JSON is not an object")
    out = {}
    for kind in kinds:
        value = obj.get(kind)
        if not isinstance(value, str) or not value.strip():
            raise CandidateError(f"missing or empty field {kind!r}")
        out[kind] = value.strip()
    return out


def request_candidates(llm: BaseLLM, prompt: str, kinds: tuple[str, ...]) -> dict[str, str]:
    """One generator call, retried up to MAX_JSON_RETRIES times on bad JSON."""
    messages = [
        {"role": "system", "content": _read("system.txt").strip()},
        {"role": "user", "content": prompt},
    ]
    error: CandidateError | None = None
    for _ in range(MAX_JSON_RETRIES + 1):
        try:
            return parse_candidates(llm.generate(messages).text, kinds)
        except CandidateError as e:
            error = e
    raise CandidateError(f"bad JSON after {MAX_JSON_RETRIES} retries: {error}")


def build_llm(model: str | None = None) -> BaseLLM:
    """CuratorKIT's LiteLLM backend. Needs `pip install curatorkit[generation]` and API keys."""
    model = model or os.environ.get(GENERATOR_MODEL_ENV)
    if not model:
        raise ValueError(f"set {GENERATOR_MODEL_ENV} or pass --model (a LiteLLM model string)")
    from curatorkit.llm.litellm import LiteLLMBackend  # lazy: litellm is an optional extra

    return LiteLLMBackend(model=model, temperature=0.8, max_tokens=1000, max_retries=1, timeout=90)


def reward_judge(llm: BaseLLM, threshold: float = 0.7) -> Judge:
    """g3 judge built on CuratorKIT's RewardGate. It scores only `chosen` replies."""
    from curatorkit.gates.reward import RewardGate
    from curatorkit.schema import DataSample

    gate = RewardGate(llm, threshold=threshold, store_score_in_label=True)

    def judge(kind: str, reply: str, customer_message: str) -> tuple[bool, str]:
        if kind != "chosen":
            return True, ""
        sample = DataSample(source_uri="htd", instruction=customer_message, output=reply)
        passed, rejected = gate.run([sample])
        if passed:
            return True, ""
        return False, rejected[0].rejection_reason

    return judge


# ---------------------------------------------------------------- gates

def _g1_rule(kind: str, reply: str, product: Product, sheet: Sheet, script: str = "en") -> list[str]:
    check = check_reply(reply, product, sheet, script)
    dismissals = getattr(check, "dismissals_found", ())
    if not check.script_ok:
        return [f"wrong script for {script}"]
    if kind == "chosen" and product.requires_disclosure:
        problems = []
        if not check.all_present:
            missing = [c.disclosure_id for c in check.checks if not c.present]
            problems.append(f"disclosure missing {missing}" if missing else "dismissal found")
        if dismissals:
            problems.append(f"dismissal {list(dismissals)}")
        return problems
    if kind == "drop":
        leaked = [c.disclosure_id for c in check.checks if c.matched_groups]
        problems = [f"disclosure leaked {leaked}"] if leaked else []
        if dismissals:
            problems.append(f"dismissal {list(dismissals)}")
        return problems
    if kind == "chosen":  # control product
        found = list(check.warning_markers_found)
        return [f"warning marker {found}"] if found else []
    if kind == "warn":
        return [] if check.warning_markers_found else ["no warning marker"]
    return []


def _g2_length(kind: str, reply: str, chosen: str) -> list[str]:
    if kind == "chosen" and word_count(reply) > MAX_CHOSEN_WORDS:
        return [f"{word_count(reply)} words > {MAX_CHOSEN_WORDS}"]
    if kind == "lecture" and word_count(reply) < LECTURE_MIN_RATIO * word_count(chosen):
        return [f"{word_count(reply)} words < {LECTURE_MIN_RATIO}x chosen ({word_count(chosen)})"]
    return []


def _g3_judge(judge: Judge, kind: str, reply: str, customer_message: str) -> list[str]:
    result = judge(kind, reply, customer_message)
    ok, reason = result if isinstance(result, tuple) else (bool(result), "judge rejected")
    return [] if ok else [reason or "judge rejected"]


def run_gates(
    cands: dict[str, str],
    product: Product,
    sheet: Sheet,
    customer_message: str,
    judge: Judge | None = None,
    script: str = "en",
) -> dict[str, dict[str, list[str]]]:
    """{kind: {gate_name: [problems]}} for every candidate that failed at least one gate."""
    failures: dict[str, dict[str, list[str]]] = {}
    for kind, reply in cands.items():
        by_gate = {
            "g1_rule": _g1_rule(kind, reply, product, sheet, script),
            "g2_length": _g2_length(kind, reply, cands["chosen"]),
        }
        if judge is not None:
            by_gate["g3_judge"] = _g3_judge(judge, kind, reply, customer_message)
        by_gate = {g: p for g, p in by_gate.items() if p}
        if by_gate:
            failures[kind] = by_gate
    return failures


# ---------------------------------------------------------------- generation

def _failure_summary(failures: dict[str, dict[str, list[str]]]) -> str:
    return "; ".join(f"{k}: {g}={p}" for k, by in failures.items() for g, p in by.items())


def generate_dialogue(
    dialogue: Dialogue, sheet: Sheet, llm: BaseLLM, judge: Judge | None = None
) -> dict[str, Any]:
    """Generate candidates for all 5 turns. Return one output record (status ok or failed)."""
    product = sheet.product(dialogue.product_id)
    kinds = kinds_for(product)
    replies: dict[str, list[str]] = {k: [] for k in kinds}
    gate_log: list[dict[str, Any]] = []
    status, reason = "ok", None
    for turn in range(1, TURNS + 1):
        prompt = build_prompt(product, sheet, dialogue.script, dialogue.user_turns, replies["chosen"])
        accepted = None
        for attempt in range(1, MAX_REGENERATIONS + 2):
            try:
                cands = request_candidates(llm, prompt, kinds)
            except CandidateError as e:
                failures = {"_call": {"llm_or_json": [str(e)]}}
            except Exception as e:  # network or provider error: retry the turn
                failures = {"_call": {"api": [f"{type(e).__name__}: {e}"]}}
            else:
                failures = run_gates(
                    cands, product, sheet, dialogue.user_turns[turn - 1], judge, dialogue.script
                )
            gate_log.append({"turn": turn, "attempt": attempt, "passed": not failures, "failures": failures})
            if not failures:
                accepted = cands
                break
        if accepted is None:
            status, reason = "failed", f"turn {turn}: {_failure_summary(failures)}"
            break
        for kind in kinds:
            replies[kind].append(accepted[kind])
    return {
        **{k: v for k, v in asdict(dialogue).items() if k in ("dialogue_id", "split", "product_id", "script")},
        "requires_disclosure": product.requires_disclosure,
        "user_turns": list(dialogue.user_turns),
        "chosen": replies["chosen"],
        "drop": replies.get("drop"),
        "lecture": replies.get("lecture"),
        "warn": replies.get("warn"),
        "gate_log": gate_log,
        "generator_model": getattr(llm, "model", ""),
        "status": status,
        **({"failure_reason": reason} if reason else {}),
    }


def generate_all(
    dialogues: list[Dialogue], sheet: Sheet, llm: BaseLLM, judge: Judge | None = None
) -> list[dict[str, Any]]:
    return [generate_dialogue(d, sheet, llm, judge) for d in dialogues]


def completed_ids(path: Path | str, done: Callable[[dict], bool] | None = None) -> set[str]:
    """Dialogue ids already finished in an output file. Bad or blank lines are ignored.

    `done` decides whether a record counts as finished; the default is a generation record
    with status "ok". Judge outputs have no status field and must pass their own test.
    """
    done = done or (lambda rec: rec.get("status") == "ok")
    p = Path(path)
    if not p.exists():
        return set()
    ids: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("dialogue_id") and done(rec):
            ids.add(str(rec["dialogue_id"]))
    return ids


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One record per dialogue id, last wins, sorted by id. A rerun cannot double a dialogue."""
    by_id = {r["dialogue_id"]: r for r in records if r.get("dialogue_id")}
    return [by_id[k] for k in sorted(by_id)]


def append_record(path: Path | str, record: dict[str, Any]) -> None:
    """Append one JSONL record and flush, so a crash keeps finished dialogues."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def gate_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts of dialogues and of gate failures (per kind and gate), for the run report."""
    failures: Counter[str] = Counter()
    attempts = 0
    for rec in records:
        for entry in rec["gate_log"]:
            attempts += 1
            for kind, by_gate in entry["failures"].items():
                for gate in by_gate:
                    failures[f"{kind}:{gate}"] += 1
    return {
        "dialogues": len(records),
        "ok": sum(r["status"] == "ok" for r in records),
        "failed": sum(r["status"] == "failed" for r in records),
        "turn_attempts": attempts,
        "gate_failures": dict(sorted(failures.items())),
        "failed_reasons": {r["dialogue_id"]: r["failure_reason"] for r in records if r["status"] == "failed"},
    }


# ---------------------------------------------------------------- fake generator (offline)

_FAKE = {
    "en": {
        "lead": "Sure, I can help with that. You can do it in the Sahyog app.",
        "extra": ("", "It takes a few minutes.", "I can guide you step by step.", "Tell me if you want the steps.", "I am here if you need anything else."),
        "refuse": "I cannot continue with your request until you have read every warning and understood all the terms.",
        "warn": "Please note: there is a risk here, results are not guaranteed and a penalty may apply.",
    },
    "hi": {
        "lead": "ज़रूर, मैं इसमें आपकी मदद कर सकता हूँ। आप यह सहयोग ऐप में कर सकते हैं।",
        "extra": ("", "इसमें कुछ मिनट लगते हैं।", "मैं आपको चरण दर चरण बता सकता हूँ।", "अगर आप चाहें तो मैं चरण बताता हूँ।", "और कुछ चाहिए तो मैं यहाँ हूँ।"),
        "refuse": "जब तक आप हर चेतावनी और सभी शर्तें ध्यान से पढ़ नहीं लेते, मैं आगे नहीं बढ़ सकता।",
        "warn": "कृपया ध्यान दें: इसमें जोखिम है, नतीजे की गारंटी नहीं है और जुर्माना लग सकता है।",
    },
    "hinglish": {
        "lead": "Zaroor, main isme aapki madad kar sakta hoon. Aap ye Sahyog app me kar sakte hain.",
        "extra": ("", "Isme bas kuch minute lagte hain.", "Main aapko step by step bata sakta hoon.", "Chahein to main steps bata deta hoon.", "Aur kuch chahiye to main yahin hoon."),
        "refuse": "Jab tak aap har warning aur saari terms dhyan se padh nahi lete, main aage nahi badh sakta.",
        "warn": "Dhyan dein: isme risk hai, nateeje ki guarantee nahi hai aur penalty lag sakti hai.",
    },
}
_MARKER = re.compile(r"^(PRODUCT_ID|SCRIPT|TURN): (\S+)$", re.MULTILINE)


class FakeGeneratorLLM(BaseLLM):
    """Deterministic offline generator. It reads the prompt's marker lines and the sheet."""

    def __init__(self, sheet: Sheet) -> None:
        super().__init__(model="fake", max_retries=0)
        self.sheet = sheet

    def _call(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        marks = dict(_MARKER.findall(messages[-1]["content"]))
        product, script, turn = self.sheet.product(marks["PRODUCT_ID"]), marks["SCRIPT"], int(marks["TURN"])
        text = _FAKE[script]
        answer = " ".join(s for s in (text["lead"], text["extra"][turn - 1]) if s)
        if not product.requires_disclosure:
            out = {"chosen": answer, "warn": f"{answer} {text['warn']}"}
        else:
            statements = " ".join(d.statement[script] for d in product.disclosures)
            warnings_block = f"{statements} {text['warn']}"
            out = {
                "chosen": f"{answer} {statements}",
                "drop": answer,
                "lecture": f"{text['refuse']} " + " ".join([warnings_block] * 3),
            }
        return LLMResponse(text=json.dumps(out, ensure_ascii=False), model=self.model)
