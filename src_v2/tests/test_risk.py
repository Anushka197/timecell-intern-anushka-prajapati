"""
Tests for core/scenarios.py — preset scenarios library.

Task 4.1: Unit tests for preset scenarios
  Validates: Requirements 2.1
"""

import pytest

from src_v2.core.scenarios import PRESET_SCENARIOS, get_preset, get_preset_names


# ── Unit tests: preset scenario names ─────────────────────────────────────

def test_all_five_preset_names_present() -> None:
    """All 5 preset scenario names must be returned by get_preset_names()."""
    names = get_preset_names()
    assert len(names) == 5
    assert "2008 GFC" in names
    assert "2020 COVID Crash" in names
    assert "2022 Rate Hike Cycle" in names
    assert "Crypto Winter" in names
    assert "Dot-com Bust" in names


def test_get_preset_names_returns_list() -> None:
    """get_preset_names() must return a list."""
    names = get_preset_names()
    assert isinstance(names, list)


# ── Unit tests: deep copy isolation ───────────────────────────────────────

def test_get_preset_returns_deep_copy() -> None:
    """Mutating the returned Scenario must not affect PRESET_SCENARIOS."""
    s = get_preset("2008 GFC")
    original_equities = PRESET_SCENARIOS["2008 GFC"].asset_crashes["Equities"]
    s.asset_crashes["Equities"] = 999.0
    assert PRESET_SCENARIOS["2008 GFC"].asset_crashes["Equities"] == original_equities


def test_get_preset_deep_copy_name_mutation() -> None:
    """Mutating the name on the returned copy must not affect PRESET_SCENARIOS."""
    s = get_preset("Crypto Winter")
    s.name = "Modified Name"
    assert PRESET_SCENARIOS["Crypto Winter"].name == "Crypto Winter"


def test_get_preset_returns_different_objects() -> None:
    """Two calls to get_preset must return distinct objects."""
    s1 = get_preset("2008 GFC")
    s2 = get_preset("2008 GFC")
    assert s1 is not s2
    assert s1.asset_crashes is not s2.asset_crashes


# ── Unit tests: expected crash percentages ────────────────────────────────

def test_2008_gfc_equities_crash() -> None:
    """2008 GFC: Equities crash must be -55.0%."""
    s = get_preset("2008 GFC")
    assert s.asset_crashes["Equities"] == -55.0


def test_2020_covid_crypto_crash() -> None:
    """2020 COVID Crash: Crypto crash must be -50.0%."""
    s = get_preset("2020 COVID Crash")
    assert s.asset_crashes["Crypto"] == -50.0


def test_2022_rate_hike_bonds_crash() -> None:
    """2022 Rate Hike Cycle: Bonds crash must be -18.0%."""
    s = get_preset("2022 Rate Hike Cycle")
    assert s.asset_crashes["Bonds"] == -18.0


def test_crypto_winter_crypto_crash() -> None:
    """Crypto Winter: Crypto crash must be -85.0%."""
    s = get_preset("Crypto Winter")
    assert s.asset_crashes["Crypto"] == -85.0


def test_dotcom_bust_equities_crash() -> None:
    """Dot-com Bust: Equities crash must be -78.0%."""
    s = get_preset("Dot-com Bust")
    assert s.asset_crashes["Equities"] == -78.0


# ── Unit tests: is_preset flag ────────────────────────────────────────────

def test_all_presets_have_is_preset_true() -> None:
    """All preset scenarios must have is_preset=True."""
    for name in get_preset_names():
        s = get_preset(name)
        assert s.is_preset is True, f"Expected is_preset=True for '{name}'"


# ── Unit tests: preset names match scenario names ─────────────────────────

def test_preset_name_matches_key() -> None:
    """Each preset's .name attribute must match its dict key."""
    for key, scenario in PRESET_SCENARIOS.items():
        assert scenario.name == key, f"Key '{key}' does not match scenario.name '{scenario.name}'"


# ── Unit tests: stress engine (Task 5.3) ──────────────────────────────────
# Validates: Requirements 3.1, 3.2

from src_v2.core.models import Asset, Portfolio, Scenario, ScenarioResult
from src_v2.core.risk import (
    compute_scenario,
    compute_all_scenarios,
    build_loss_matrix,
    validate_crash_pct,
)


def test_empty_portfolio_returns_zero_post_crash_value() -> None:
    """Portfolio with no assets must produce post_crash_value = 0.0."""
    p = Portfolio(assets=[], total_value=1_000_000.0, monthly_expenses=5_000.0)
    s = Scenario(name="Empty Test", asset_crashes={})
    r = compute_scenario(p, s)
    assert r.post_crash_value == 0.0


def test_single_asset_50pct_crash() -> None:
    """Single asset with 100% allocation and -50% crash → post_crash_value = total_value * 0.5."""
    total = 200_000.0
    p = Portfolio(
        assets=[Asset("AAPL", allocation_pct=100.0)],
        total_value=total,
        monthly_expenses=1_000.0,
    )
    s = Scenario(name="Half Crash", asset_crashes={"AAPL": -50.0})
    r = compute_scenario(p, s)
    assert abs(r.post_crash_value - total * 0.5) < 1e-6


def test_concentration_warning_triggers() -> None:
    """concentration_warning must be True when any asset has allocation_pct > 40."""
    p = Portfolio(
        assets=[
            Asset("BTC", allocation_pct=50.0),
            Asset("CASH", allocation_pct=50.0),
        ],
        total_value=100_000.0,
        monthly_expenses=1_000.0,
    )
    s = Scenario(name="Concentration Test", asset_crashes={})
    r = compute_scenario(p, s)
    assert r.concentration_warning is True


def test_concentration_warning_not_triggered() -> None:
    """concentration_warning must be False when all assets have allocation_pct <= 40."""
    p = Portfolio(
        assets=[
            Asset("A", allocation_pct=30.0),
            Asset("B", allocation_pct=30.0),
            Asset("C", allocation_pct=40.0),
        ],
        total_value=100_000.0,
        monthly_expenses=1_000.0,
    )
    s = Scenario(name="No Concentration", asset_crashes={})
    r = compute_scenario(p, s)
    assert r.concentration_warning is False


def test_ruin_test_fail_when_runway_le_12() -> None:
    """ruin_test must be 'FAIL' when runway_months <= 12."""
    # post_crash_value = 1000 * 1.0 * (1 - 0.5) = 500; monthly_expenses = 100 → runway = 5 months
    p = Portfolio(
        assets=[Asset("X", allocation_pct=100.0)],
        total_value=1_000.0,
        monthly_expenses=100.0,
    )
    s = Scenario(name="Ruin Scenario", asset_crashes={"X": -50.0})
    r = compute_scenario(p, s)
    assert r.runway_months <= 12
    assert r.ruin_test == "FAIL"


def test_ruin_test_pass_when_runway_gt_12() -> None:
    """ruin_test must be 'PASS' when runway_months > 12."""
    # post_crash_value = 100_000 * 1.0 * (1 - 0.1) = 90_000; monthly_expenses = 1_000 → runway = 90
    p = Portfolio(
        assets=[Asset("Y", allocation_pct=100.0)],
        total_value=100_000.0,
        monthly_expenses=1_000.0,
    )
    s = Scenario(name="Safe Scenario", asset_crashes={"Y": -10.0})
    r = compute_scenario(p, s)
    assert r.runway_months > 12
    assert r.ruin_test == "PASS"


def test_asset_not_in_scenario_gets_zero_crash() -> None:
    """Assets absent from scenario.asset_crashes must receive crash_pct = 0 (no loss)."""
    total = 50_000.0
    p = Portfolio(
        assets=[Asset("GOLD", allocation_pct=100.0)],
        total_value=total,
        monthly_expenses=500.0,
    )
    # GOLD is not in asset_crashes → should be unaffected
    s = Scenario(name="No GOLD Crash", asset_crashes={"BTC": -80.0})
    r = compute_scenario(p, s)
    assert abs(r.post_crash_value - total) < 1e-6
    assert r.asset_losses["GOLD"] == 0.0
