import copy
from core.models import Scenario

PRESET_SCENARIOS: dict[str, Scenario] = {
    "2008 GFC": Scenario(
        name="2008 GFC",
        asset_crashes={
            "Equities":    -55.0,
            "Real Estate": -30.0,
            "Bonds":       -10.0,
            "Commodities": -25.0,
            "Crypto":        0.0,
            "Cash":          0.0,
            "Gold":          5.0,  # safe haven gain stored as positive
        },
        is_preset=True,
    ),
    "2020 COVID Crash": Scenario(
        name="2020 COVID Crash",
        asset_crashes={
            "Equities":    -34.0,
            "Real Estate": -10.0,
            "Bonds":        -5.0,
            "Commodities": -40.0,
            "Crypto":      -50.0,
            "Cash":          0.0,
            "Gold":         -5.0,
        },
        is_preset=True,
    ),
    "2022 Rate Hike Cycle": Scenario(
        name="2022 Rate Hike Cycle",
        asset_crashes={
            "Equities":    -25.0,
            "Real Estate": -15.0,
            "Bonds":       -18.0,
            "Commodities":  -5.0,
            "Crypto":      -75.0,
            "Cash":          0.0,
            "Gold":         -3.0,
        },
        is_preset=True,
    ),
    "Crypto Winter": Scenario(
        name="Crypto Winter",
        asset_crashes={
            "Equities":    -10.0,
            "Real Estate":  -5.0,
            "Bonds":         0.0,
            "Commodities":  -5.0,
            "Crypto":      -85.0,
            "Cash":          0.0,
            "Gold":          0.0,
        },
        is_preset=True,
    ),
    "Dot-com Bust": Scenario(
        name="Dot-com Bust",
        asset_crashes={
            "Equities":    -78.0,
            "Real Estate":   5.0,
            "Bonds":         8.0,
            "Commodities":  -5.0,
            "Crypto":        0.0,
            "Cash":          0.0,
            "Gold":          5.0,
        },
        is_preset=True,
    ),
}

def get_preset_names() -> list[str]:
    """Returns list of preset scenario names."""
    return list(PRESET_SCENARIOS.keys())

def get_preset(name: str) -> Scenario:
    """Returns a deep copy of the named preset scenario."""
    return copy.deepcopy(PRESET_SCENARIOS[name])
