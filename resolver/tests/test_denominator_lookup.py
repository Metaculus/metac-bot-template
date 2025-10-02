import pandas as pd

from resolver.tools.denominators import (
    clear_population_cache,
    get_population,
    safe_pct_to_people,
)


def test_population_lookup_prefers_latest_year(tmp_path):
    csv_path = tmp_path / "population.csv"
    df = pd.DataFrame(
        [
            {"iso3": "SDN", "year": 2024, "population": 38000000},
            {"iso3": "SDN", "year": 2025, "population": 40000000},
            {"iso3": "KEN", "year": 2023, "population": 52000000},
        ]
    )
    df.to_csv(csv_path, index=False)

    clear_population_cache()
    assert get_population("SDN", 2025, csv_path) == 40000000
    # Future years fall back to the latest available
    assert get_population("SDN", 2026, csv_path) == 40000000

    # Remove the 2025 row to exercise <= fallback
    df = df[df["year"] != 2025]
    df.to_csv(csv_path, index=False)
    clear_population_cache()
    assert get_population("SDN", 2025, csv_path) == 38000000

    # Percent to people conversion
    df = pd.DataFrame([
        {"iso3": "SDN", "year": 2025, "population": 40000000},
    ])
    df.to_csv(csv_path, index=False)
    clear_population_cache()
    assert safe_pct_to_people(25.0, "SDN", 2025, denom_path=csv_path) == 10000000
