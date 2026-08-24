"""Tests for make canonicalisation and matching (pure logic, no models)."""

import pytest

from truck_extract_match.make.aliases import (canonical_brands, match_make,
                                              normalize_maker)


def test_normalize_strips_legal_tokens():
    assert normalize_maker("TATA MOTORS LTD") == "TATA"
    assert normalize_maker("Mahindra & Mahindra Ltd") == "MAHINDRA MAHINDRA"


def test_painted_word_matches_legal_entity():
    # painted "EICHER" vs Parivahan "VE COMMERCIAL VEHICLES LTD"
    m = match_make("EICHER", "VE COMMERCIAL VEHICLES LTD")
    assert m.matched


def test_bharatbenz_maps_to_daimler():
    m = match_make("BharatBenz", "DAIMLER INDIA COMMERCIAL VEHICLES PVT LTD")
    assert m.matched


def test_tata_variants():
    assert match_make("TATA", "TATA MOTORS LTD").matched
    assert match_make("Tata Motors", "TATA MOTORS LIMITED").matched


def test_vecv_ambiguity_allows_both():
    # VE Commercial Vehicles builds both Eicher and Volvo -> either painted brand is OK
    assert match_make("VOLVO", "VE COMMERCIAL VEHICLES LTD").matched
    assert match_make("EICHER", "VE COMMERCIAL VEHICLES LTD").matched


def test_true_mismatch():
    m = match_make("TATA", "ASHOK LEYLAND LTD")
    assert not m.matched


def test_logo_only_match():
    # no readable word, but the logo classifier says ASHOK_LEYLAND
    m = match_make("", "ASHOK LEYLAND LTD", logo_brand="ASHOK_LEYLAND", logo_prob=0.95)
    assert m.matched and m.method == "logo"


def test_canonical_brands_unknown():
    assert canonical_brands("SOME RANDOM TEXT") == set()


def test_vecv_abbreviation_matches_eicher_and_volvo():
    assert match_make("Eicher", "VECV").matched
    assert match_make("Volvo", "VECV").matched


@pytest.mark.parametrize("extracted,claimed", [
    ("AMW", "Asia Motor Works Ltd"),
    ("JBM", "JBM Auto Ltd"),
    ("Kamaz", "KAMAZ"),
    ("Olectra", "Olectra Greentech Ltd"),
    ("Hino", "Hino Motors Ltd"),
    ("PMI", "PMI Electro Mobility Pvt Ltd"),
])
def test_additional_manufacturers_match(extracted, claimed):
    assert match_make(extracted, claimed).matched
