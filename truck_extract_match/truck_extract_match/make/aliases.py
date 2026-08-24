"""Make canonicalisation: bridge the painted brand <-> Parivahan legal-entity name.

The word on the truck ("EICHER") rarely equals Parivahan's ``maker`` field
("VE COMMERCIAL VEHICLES LTD"). We canonicalise BOTH sides to a set of brand keys and
match when the sets intersect. Ambiguous makers (VECV builds Eicher AND Volvo) map to a
set, so either painted brand satisfies the claim.

Curate ``_MAKER_TO_BRANDS`` and ``_BRAND_KEYWORDS`` once from your VAHAN distinct-maker
list — it is a bounded set of manufacturers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..matching import token_set_ratio

# Tokens that carry no brand signal in a maker string.
_STOPWORDS = frozenset({
    "LTD", "LIMITED", "PVT", "PRIVATE", "CO", "COMPANY", "CORP", "CORPORATION",
    "MOTORS", "MOTOR", "INDIA", "INDIAN", "COMMERCIAL", "VEHICLES", "VEHICLE",
    "AUTOMOBILES", "AUTOMOBILE", "AUTO", "INDUSTRIES", "AND", "THE", "OF",
    "MFG", "MANUFACTURING", "TRUCKS", "BUS", "BUSES",
})

# Canonical brand keys.
TATA, ASHOK_LEYLAND, EICHER, BHARATBENZ, MAHINDRA, VOLVO = (
    "TATA", "ASHOK_LEYLAND", "EICHER", "BHARATBENZ", "MAHINDRA", "VOLVO")
SML, FORCE, MAN, SCANIA, BENZ = "SML_ISUZU", "FORCE", "MAN", "SCANIA", "MERCEDES"
AMW, JBM, KAMAZ, OLECTRA, HINO, PMI_ELECTRO = (
    "AMW", "JBM", "KAMAZ", "OLECTRA", "HINO", "PMI_ELECTRO")

# Parivahan maker legal-entity substrings -> set of brands they can appear as.
# Match is done on normalised (stopword-stripped) tokens; order-independent.
_MAKER_TO_BRANDS: dict[str, set[str]] = {
    "TATA": {TATA},
    "ASHOK LEYLAND": {ASHOK_LEYLAND},
    "EICHER": {EICHER},
    "VE COMMERCIAL": {EICHER, VOLVO},          # VECV builds both Eicher and Volvo trucks
    "VECV": {EICHER, VOLVO},                   # common abbreviation for the same entity
    "VOLVO": {VOLVO},
    "DAIMLER": {BHARATBENZ, BENZ},             # BharatBenz = Daimler India Commercial Vehicles
    "BHARATBENZ": {BHARATBENZ},
    "MERCEDES": {BENZ},
    "MAHINDRA": {MAHINDRA},
    "SML": {SML}, "ISUZU": {SML},              # SML Isuzu (ex-Swaraj Mazda)
    "SWARAJ MAZDA": {SML},
    "FORCE": {FORCE},
    "MAN": {MAN},
    "SCANIA": {SCANIA},
    "AMW": {AMW}, "ASIA MOTOR WORKS": {AMW},
    "JBM": {JBM},
    "KAMAZ": {KAMAZ},
    "OLECTRA": {OLECTRA},
    "HINO": {HINO},
    "PMI ELECTRO": {PMI_ELECTRO}, "PMI": {PMI_ELECTRO},
}

# Keywords/brand words as painted on the truck -> canonical brand.
_BRAND_KEYWORDS: dict[str, str] = {
    "TATA": TATA,
    "ASHOK": ASHOK_LEYLAND, "LEYLAND": ASHOK_LEYLAND,
    "EICHER": EICHER,
    "BHARATBENZ": BHARATBENZ, "BHARAT": BHARATBENZ,
    "MAHINDRA": MAHINDRA,
    "VOLVO": VOLVO,
    "SML": SML, "ISUZU": SML,
    "FORCE": FORCE,
    "MAN": MAN,
    "SCANIA": SCANIA,
    "MERCEDES": BENZ, "BENZ": BENZ,
    "AMW": AMW,
    "JBM": JBM,
    "KAMAZ": KAMAZ,
    "OLECTRA": OLECTRA,
    "HINO": HINO,
    "PMI": PMI_ELECTRO,
}


def normalize_maker(text: str) -> str:
    """Uppercase, strip punctuation and legal/stopword tokens, collapse whitespace."""
    if not text:
        return ""
    toks = re.sub(r"[^A-Za-z0-9 ]", " ", text.upper()).split()
    kept = [t for t in toks if t not in _STOPWORDS]
    return " ".join(kept or toks)  # if everything was a stopword, keep original tokens


def canonical_brands(text: str, fuzzy_threshold: float = 0.86) -> set[str]:
    """Map an arbitrary make string (painted word OR Parivahan legal name) to brand keys.

    Order of resolution: exact legal-entity substring -> brand keyword token ->
    fuzzy token-set match against known maker strings. Returns a (possibly empty) set;
    ambiguous makers (e.g. VECV) return multiple brands.
    """
    if not text:
        return set()
    norm = normalize_maker(text)
    upper = text.upper()
    brands: set[str] = set()

    # 1. legal-entity substrings (checked on raw upper so multi-word keys work)
    for key, bset in _MAKER_TO_BRANDS.items():
        if key in upper:
            brands |= bset

    # 2. single-word brand keywords
    for tok in norm.split():
        if tok in _BRAND_KEYWORDS:
            brands.add(_BRAND_KEYWORDS[tok])

    if brands:
        return brands

    # 3. fuzzy fallback against known maker strings
    best_brand, best_score = None, 0.0
    for key, bset in _MAKER_TO_BRANDS.items():
        sc = token_set_ratio(norm, normalize_maker(key))
        if sc > best_score:
            best_brand, best_score = bset, sc
    if best_brand and best_score >= fuzzy_threshold:
        return set(best_brand)
    return set()


@dataclass
class MakeMatch:
    matched: bool
    score: float
    extracted_brands: set[str]
    claimed_brands: set[str]
    method: str  # "exact" | "logo" | "fuzzy" | "none"


def match_make(extracted: str, claimed: str, logo_brand: str | None = None,
               logo_prob: float = 0.0) -> MakeMatch:
    """Decide whether the extracted make (+ optional logo) agrees with the claimed maker."""
    claimed_brands = canonical_brands(claimed)
    ex_brands = canonical_brands(extracted)
    method = "exact" if ex_brands else "none"

    if logo_brand:
        lb = logo_brand.upper()
        if lb in _BRAND_KEYWORDS.values() or lb in _BRAND_KEYWORDS:
            ex_brands.add(_BRAND_KEYWORDS.get(lb, lb))
            if not method or method == "none":
                method = "logo"

    if not ex_brands:
        return MakeMatch(False, 0.0, ex_brands, claimed_brands, "none")
    if not claimed_brands:
        return MakeMatch(False, 0.0, ex_brands, claimed_brands, method)

    matched = bool(ex_brands & claimed_brands)
    if matched:
        score = max(0.9, logo_prob)  # brand-set intersection is a strong signal
        return MakeMatch(True, round(score, 4), ex_brands, claimed_brands, method)

    # no intersection: report best fuzzy similarity for the manual-review UI
    fuzzy = token_set_ratio(normalize_maker(extracted), normalize_maker(claimed))
    return MakeMatch(False, round(fuzzy, 4), ex_brands, claimed_brands, "fuzzy")
