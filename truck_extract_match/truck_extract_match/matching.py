"""Generic string-matching utilities (Levenshtein ratio + token-set ratio).

Prefers ``rapidfuzz`` when installed; otherwise uses a pure-Python fallback so the
package runs with zero third-party deps for the logic layer.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when rapidfuzz is present
    from rapidfuzz import fuzz as _rf_fuzz

    def token_set_ratio(a: str, b: str) -> float:
        return _rf_fuzz.token_set_ratio(a, b) / 100.0

    def ratio(a: str, b: str) -> float:
        return _rf_fuzz.ratio(a, b) / 100.0

    _BACKEND = "rapidfuzz"

except ImportError:  # pure-Python fallback
    _BACKEND = "pure-python"

    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def ratio(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        return 1.0 - _levenshtein(a, b) / max(len(a), len(b))

    def token_set_ratio(a: str, b: str) -> float:
        """Approximation of rapidfuzz.token_set_ratio for the fallback path."""
        ta, tb = set(a.split()), set(b.split())
        if not ta or not tb:
            return 0.0
        inter = sorted(ta & tb)
        s0 = " ".join(inter)
        s1 = " ".join(inter + sorted(ta - tb))
        s2 = " ".join(inter + sorted(tb - ta))
        # if one token set is a subset of the other, that's a strong signal
        if inter and (ta <= tb or tb <= ta):
            return max(0.9, ratio(s1, s2))
        return max(ratio(s0, s1), ratio(s0, s2), ratio(s1, s2))


def backend() -> str:
    return _BACKEND
