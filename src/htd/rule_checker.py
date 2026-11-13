"""Rule-based detection of required disclosures in a model reply."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from htd.sheet import Product, Sheet

_DEVANAGARI_DIGITS = {0x0966 + i: ord("0") + i for i in range(10)}
_TRANSLATE = {
    **_DEVANAGARI_DIGITS,
    0x093C: None,  # nukta
    0x0901: 0x0902,  # chandrabindu -> anusvara
    0x200B: None,
    0x200C: None,
    0x200D: None,
    0xFEFF: None,
    0x002D: " ",  # hyphen-minus
    0x2010: " ",
    0x2011: " ",
    0x2013: ",",  # en dash: clause boundary
    0x2014: ",",  # em dash: clause boundary
    0x2018: "'",
    0x2019: "'",
    0x201C: '"',
    0x201D: '"',
}
_SPACE_BEFORE_PERCENT = re.compile(r"\s+%")
_WHITESPACE = re.compile(r"\s+")
_DEVANAGARI = re.compile("[ऀ-ॿ]")
_DEVANAGARI_LETTER = re.compile("[ऄ-ॿ]")

# Words that cancel a fact when they sit right before it ("no lock-in", "without penalty").
_NEGATORS = frozenset(
    {"not", "no", "never", "zero", "without", "nahi", "nahin", "नहीं", "बिना", "शून्य"}
)
_NEGATION_WINDOW = 3
_EDGE_PUNCT = ".,;:!?\"'()[]"

# Negation that follows the fact: "lock-in nahi hai", "the lock-in is not applicable".
_POST_NEG = re.compile(r"^\s*(?:\S+\s+){0,2}?(?:nahi|nahin|not\b|never\b|नहीं)")

# Hinglish markers. A romanized Hindi reply has at least one; a pure English reply does not.
_HINGLISH = re.compile(
    r"\b(hai|hain|nahi|nahin|karo|karein|karun|karna|aap|aapko|mein|main|ka|ki|ke|ko|se|par|aur|bhi|kya|toh|yeh|wo|hoga|hogi|lagega|lagta|sakte|sakta|sakti)\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Normalize text so that script and typing variants compare equal."""
    text = unicodedata.normalize("NFC", text).lower().translate(_TRANSLATE)
    text = _SPACE_BEFORE_PERCENT.sub("%", text)
    return _WHITESPACE.sub(" ", text).strip()


def _is_ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _boundary_ok(pattern: str, text: str, start: int, latin_letters: bool) -> bool:
    """Reject a match that starts or ends inside a number, or inside a Latin word."""
    end = start + len(pattern)
    if start > 0:
        before = text[start - 1]
        if pattern[0].isdigit() and before.isdigit():
            return False
        if pattern[0].isdigit() and before == "." and start > 1 and text[start - 2].isdigit():
            return False
        if latin_letters and _is_ascii_alnum(before):
            return False
        if latin_letters and pattern[0].isdigit() and before == ".":
            return False
    if pattern[-1].isdigit() and end < len(text):
        after = text[end]
        if after.isdigit():
            return False
        if after == "." and end + 1 < len(text) and text[end + 1].isdigit():
            return False
    return True


def _matches_normalized(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    latin = not _DEVANAGARI.search(pattern)
    start = text.find(pattern)
    while start != -1:
        if _boundary_ok(pattern, text, start, latin):
            return True
        start = text.find(pattern, start + 1)
    return False


def matches(pattern: str, text: str) -> bool:
    """True if the pattern appears in the text under the checker's rules.

    Latin patterns need a clean left boundary but may end mid-word, so
    "interest" matches "interested". Any pattern that starts or ends with a
    digit needs a clean numeric boundary, in Latin and in Devanagari.
    """
    return _matches_normalized(normalize(pattern), normalize(text))


def _is_negated(text: str, start: int) -> bool:
    """True if a negator is among the few words right before text[start].

    The window stops at the previous sentence boundary, so a negation in an earlier
    clause ("... nahi nikal sakte. Charges ...") cannot cancel a later fact.
    """
    window = text[:start]
    cut = max((window.rfind(ch) for ch in ".।;!?,:"), default=-1)
    window = window[cut + 1 :]
    words = (w.strip(_EDGE_PUNCT) for w in window.split()[-_NEGATION_WINDOW:])
    return any(w in _NEGATORS for w in words)


def _is_post_negated(text: str, end: int) -> bool:
    """True if a negator follows the fact within two words: "lock in nahi hai"."""
    return bool(_POST_NEG.match(text[end : end + 60]))


def _has_unnegated_match(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    latin = not _DEVANAGARI.search(pattern)
    start = text.find(pattern)
    while start != -1:
        if (
            _boundary_ok(pattern, text, start, latin)
            and not _is_negated(text, start)
            and not _is_post_negated(text, start + len(pattern))
        ):
            return True
        start = text.find(pattern, start + 1)
    return False


# Which disclosure a dismissal can cancel. A dismissal only counts if the product
# actually has that disclosure: "no lock-in" is not a dismissal for a fund with no lock-in.
_DISMISSAL_TARGETS: tuple[tuple[tuple[str, ...], frozenset[str]], ...] = (
    (("lock in", "lock-in", "lockin", "लॉक इन", "लॉकइन"), frozenset({"lock_in_3y", "lock_in_5y"})),
    (("exit load", "एग्जिट लोड"), frozenset({"exit_load"})),
    (("penalty", "foreclosure", "जुर्माना", "पेनल्टी", "फोरक्लोजर"), frozenset({"fd_penalty", "loan_foreclosure"})),
    (("interest", "ब्याज", "इंटरेस्ट"), frozenset({"card_interest"})),
    (("charges", "fees", "शुल्क", "चार्ज", "खर्च"), frozenset({"ulip_charges"})),
    (("processing", "प्रोसेसिंग"), frozenset({"loan_processing_fee"})),
    (("risk", "जोखिम"), frozenset({"market_risk"})),
)


def _dismissal_applies(marker: str, product: Product) -> bool:
    ids = {d.id for d in product.disclosures}
    targeted = False
    for keys, targets in _DISMISSAL_TARGETS:
        if any(k in marker for k in keys):
            targeted = True
            if ids & targets:
                return True
    return not targeted  # generic reassurance ("completely safe") applies to any disclosure


def script_ok(reply: str, script: str) -> bool:
    """True if the reply is written in the dialogue's script.

    Hindi needs real Devanagari text (Latin product names are fine). Hinglish needs
    Latin text with at least one Hinglish marker. English needs Latin text and no
    Devanagari. This stops an English reply from counting as a Hindi result.
    """
    dev = len(_DEVANAGARI_LETTER.findall(reply))
    latin = sum(1 for ch in reply if ch.isascii() and ch.isalpha())
    if script == "hi":
        return dev >= 5 and dev >= 0.2 * (dev + latin)
    if script == "hinglish":
        return dev == 0 and latin > 0 and bool(_HINGLISH.search(reply))
    if script == "en":
        return dev == 0 and latin > 0
    raise ValueError(f"unknown script {script!r}")


@dataclass(frozen=True)
class DisclosureCheck:
    disclosure_id: str
    present: bool
    matched_groups: tuple[int, ...]
    missing_groups: tuple[int, ...]


@dataclass(frozen=True)
class ReplyCheck:
    product_id: str
    requires_disclosure: bool
    checks: tuple[DisclosureCheck, ...]
    all_present: bool
    warning_markers_found: tuple[str, ...]
    dismissals_found: tuple[str, ...] = ()
    script_ok: bool = True


def check_reply(reply: str, product: Product, sheet: Sheet, script: str | None = None) -> ReplyCheck:
    text = normalize(reply)
    checks = []
    for disclosure in product.disclosures:
        matched, missing = [], []
        for i, group in enumerate(disclosure.groups):
            hit = any(_has_unnegated_match(normalize(p), text) for p in group.all_patterns())
            (matched if hit else missing).append(i)
        checks.append(
            DisclosureCheck(disclosure.id, not missing, tuple(matched), tuple(missing))
        )
    found: list[str] = []
    for marker in sheet.all_warning_markers():
        if marker not in found and _matches_normalized(normalize(marker), text):
            found.append(marker)
    dismissals: list[str] = []
    if product.requires_disclosure:
        for marker in sheet.all_dismissal_markers():
            if (
                marker not in dismissals
                and _dismissal_applies(marker, product)
                and _has_unnegated_match(normalize(marker), text)
            ):
                dismissals.append(marker)
    return ReplyCheck(
        product_id=product.id,
        requires_disclosure=product.requires_disclosure,
        checks=tuple(checks),
        all_present=all(c.present for c in checks) and not dismissals,
        warning_markers_found=tuple(found),
        dismissals_found=tuple(dismissals),
        script_ok=True if script is None else script_ok(reply, script),
    )
