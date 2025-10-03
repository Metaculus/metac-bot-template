import pandas as pd
import pytest

from resolver.ingestion import acled_client


@pytest.mark.parametrize(
    "case_name,iso3,country,history,current,expected_prev12,expected_onset",
    [
        (
            "positive_onset",
            "ETH",
            "Ethiopia",
            [("2022-01", 6), ("2022-03", 6), ("2022-07", 6), ("2022-11", 6)],
            ("2022-12", 25),
            24,
            True,
        ),
        (
            "just_below_threshold",
            "ETH",
            "Ethiopia",
            [("2022-02", 6), ("2022-05", 6), ("2022-08", 6), ("2022-11", 6)],
            ("2023-01", 24),
            24,
            False,
        ),
        (
            "no_onset_due_to_history",
            "SDN",
            "Sudan",
            [("2022-02", 20), ("2022-08", 15)],
            ("2022-12", 60),
            35,
            False,
        ),
    ],
)
def test_conflict_onset_rule_cases(
    case_name,
    iso3,
    country,
    history,
    current,
    expected_prev12,
    expected_onset,
):
    current_month, current_value = current
    all_months = history + [current]
    df_flags = pd.DataFrame(
        [
            {
                "iso3": iso3,
                "month": month,
                "event_type": "Battles",
                "fatalities": value,
            }
            for month, value in all_months
        ]
    )

    flags = acled_client.compute_conflict_onset_flags(df_flags)
    target = flags[(flags["iso3"] == iso3) & (flags["month"] == current_month)]
    assert not target.empty, f"missing onset row for {case_name}"
    row = target.iloc[0]
    assert row["prev12_battle_fatalities"] == expected_prev12
    assert bool(row["is_onset"]) is expected_onset

    records = [
        {
            "event_date": f"{month}-15",
            "event_type": "Battles",
            "country": country,
            "iso3": iso3,
            "fatalities": str(value),
            "notes": "",
        }
        for month, value in all_months
    ]

    config = acled_client.load_config()
    countries, shocks = acled_client.load_registries()
    rows = acled_client._build_rows(
        records,
        config,
        countries,
        shocks,
        "https://example.com/acled",
        f"{current_month}-28",
        f"{current_month}-28T00:00:00Z",
    )

    conflict_rows = [
        row
        for row in rows
        if row["iso3"] == iso3
        and row["metric"] == "fatalities_battle_month"
        and row["as_of_date"] == current_month
    ]
    escalation = [row for row in conflict_rows if row["hazard_code"] == "ACE"]
    assert escalation, f"expected escalation row for {case_name}"
    assert escalation[0]["value"] == current_value

    onset_rows = [row for row in conflict_rows if row["hazard_code"] == "ACO"]
    if expected_onset:
        assert len(onset_rows) == 1, f"expected single onset row for {case_name}"
        assert onset_rows[0]["value"] == current_value
    else:
        assert not onset_rows, f"did not expect onset row for {case_name}"
