"""Indian vehicle-registration-number (VRN) grammar, correction and matching.

Grammar (standard series):   SS DD [X{0,3}] N{1,4}
  SS  = state/UT code            (LETTERS, from a fixed whitelist)
  DD  = RTO district code        (DIGITS)
  X   = series letters           (0-3 LETTERS, never I or O)
  N   = unique number            (1-4 DIGITS)
BH (Bharat) series:  YY BH N{1,4} X{1,2}

Two capabilities:
  1. parse_and_correct(raw)  -> best-effort standalone correction using the grammar
     (positional letter/digit class + state whitelist + I/O rule). Used for reporting
     "what we actually read".
  2. match_vrn(read, claimed) -> the AUTHORITATIVE decision. Because this pipeline
     always has a claimed VRN, we compare with a *confusion-aware* edit distance so a
     smudged char that is visually consistent with the claimed char counts as a match.
     This is how smudged/ambiguous digits get "inferred".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Valid Indian state / UT codes (incl. older + renamed variants) and the BH series.
STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HR",
    "HP", "JK", "JH", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML", "MZ", "NL",
    "OD", "OR", "PY", "PB", "RJ", "SK", "TN", "TS", "TG", "TR", "UP", "UK", "UA",
    "WB", "BH",
})

# Groups of visually confusable characters (OCR failure modes on smudged plates).
_CONFUSION_GROUPS: tuple[frozenset[str], ...] = (
    frozenset("0ODQ"), frozenset("1IL"), frozenset("2Z"), frozenset("5S"),
    frozenset("8B"),   frozenset("6G"),  frozenset("4A"), frozenset("7T"),
    frozenset("9"),    frozenset("MN"),  frozenset("CG"),
)
_CHAR_GROUPS: dict[str, set[int]] = {}
for _gi, _g in enumerate(_CONFUSION_GROUPS):
    for _c in _g:
        _CHAR_GROUPS.setdefault(_c, set()).add(_gi)

# Digit<->letter remaps for positional class correction.
_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
             "S": "5", "B": "8", "G": "6", "A": "4", "T": "7"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
              "4": "A", "7": "T"}

_GRAMMAR = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}$")


def is_confusable(a: str, b: str) -> bool:
    """True if a and b are equal or belong to a shared confusion group."""
    if a == b:
        return True
    ga, gb = _CHAR_GROUPS.get(a), _CHAR_GROUPS.get(b)
    return bool(ga and gb and ga & gb)


def confusable_distance(s1: str, s2: str) -> int:
    """Edit distance where a confusable substitution is free (cost 0)."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if is_confusable(s1[i - 1], s2[j - 1]) else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub)
    return dp[n][m]


def normalize_vrn(raw: str) -> str:
    """Uppercase, drop everything non-alphanumeric, strip the 'IND' country marker."""
    if not raw:
        return ""
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    # 'IND' only when it prefixes a plausible state code (avoid eating INDore-like reads)
    if s.startswith("IND") and len(s) > 5:
        s = s[3:]
    return s


def _to_letters(s: str) -> str:
    return "".join(_TO_LETTER.get(c, c) for c in s)


def _to_digits(s: str) -> str:
    return "".join(_TO_DIGIT.get(c, c) for c in s)


@dataclass
class PlateParse:
    raw: str
    corrected: str
    state: str = ""
    district: str = ""
    series: str = ""
    number: str = ""
    valid: bool = False          # satisfies grammar AND known state code
    corrections: int = 0         # chars changed by class-correction
    notes: list[str] = field(default_factory=list)


def parse_and_correct(raw: str) -> PlateParse:
    """Best-effort grammar-aware correction of a standalone read (no claimed value).

    Heuristic segmentation: 2 state letters, 2 district digits, then series letters
    (leading) + number digits (trailing) split from the right. Authoritative matching
    should still go through ``match_vrn`` against the claimed VRN.
    """
    s = normalize_vrn(raw)
    if len(s) < 5:
        return PlateParse(raw=raw, corrected=s, valid=False, notes=["too short"])

    state = _to_letters(s[:2])
    district = _to_digits(s[2:4])
    rest = s[4:]

    # Split rest: trailing digits (<=4) = number, remaining leading = series letters.
    num_chars: list[str] = []
    i = len(rest) - 1
    while i >= 0 and len(num_chars) < 4 and (rest[i].isdigit() or rest[i] in _TO_DIGIT):
        num_chars.append(_to_digits(rest[i]))
        i -= 1
    number = "".join(reversed(num_chars))
    series = _to_letters(rest[: i + 1])
    # Series never contains I or O -> nudge to nearest digit-legal letter is ambiguous;
    # just flag it rather than silently rewrite.
    notes: list[str] = []
    if "I" in series or "O" in series:
        notes.append("series contains I/O (illegal) - likely misread")

    corrected = state + district + series + number
    corrections = sum(1 for a, b in zip(s, corrected) if a != b)
    valid = bool(_GRAMMAR.match(corrected)) and state in STATE_CODES and 1 <= len(number) <= 4
    if state not in STATE_CODES:
        notes.append(f"unknown state code '{state}'")
    return PlateParse(raw=raw, corrected=corrected, state=state, district=district,
                      series=series, number=number, valid=valid,
                      corrections=corrections, notes=notes)


@dataclass
class VrnMatch:
    matched: bool
    score: float                 # 0..1
    distance: int                # confusable edit distance to claimed
    read_norm: str
    claimed_norm: str
    inferred: bool = False       # True if smudge-inference (>0 confusable subs) was needed


def match_vrn(read: str, claimed: str, max_confusable_edits: int = 1) -> VrnMatch:
    """Authoritative VRN decision against the claimed value.

    A read matches the claimed VRN when its confusion-aware edit distance is within
    ``max_confusable_edits``. Purely-confusable differences (0/O, 8/B, ...) cost 0, so a
    smudged char that is visually consistent with the claimed plate is treated as a
    correct read — this is the smudge-inference behaviour.
    """
    r, c = normalize_vrn(read), normalize_vrn(claimed)
    if not r or not c:
        return VrnMatch(False, 0.0, max(len(r), len(c)), r, c)
    dist = confusable_distance(r, c)
    matched = dist <= max_confusable_edits
    denom = max(len(r), len(c))
    score = max(0.0, 1.0 - dist / denom) if denom else 0.0
    # inferred if the strings differ literally but matched under confusion
    inferred = matched and r != c
    return VrnMatch(matched, round(score, 4), dist, r, c, inferred)
