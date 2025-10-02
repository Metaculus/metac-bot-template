from resolver.ingestion import dtm_client


def test_flood_keyword_maps_to_flood_code():
    _, shocks = dtm_client.load_registries()
    cfg = dtm_client.load_config()
    hazard = dtm_client.infer_hazard(
        ["Flood Displacement Tracking – Sudan – Jan 2025"],
        shocks=shocks,
        keywords_cfg=cfg.get("shock_keywords", {}),
    )
    assert hazard.code == "FL"
    assert "flood" in hazard.label.lower()


def test_flow_monitoring_defaults_to_displacement_influx():
    _, shocks = dtm_client.load_registries()
    cfg = dtm_client.load_config()
    hazard = dtm_client.infer_hazard(
        ["DTM Flow Monitoring – Yemen – Mar 2025"],
        shocks=shocks,
        keywords_cfg=cfg.get("shock_keywords", {}),
        default_key="displacement_influx",
    )
    assert hazard.code == "DI"
    assert "displacement" in hazard.label.lower()


def test_ambiguous_titles_return_multi():
    _, shocks = dtm_client.load_registries()
    cfg = dtm_client.load_config()
    hazard = dtm_client.infer_hazard(
        ["DTM Situation Update – Flood and Drought Impacts"],
        shocks=shocks,
        keywords_cfg=cfg.get("shock_keywords", {}),
    )
    assert hazard.code == "multi"
    assert hazard.label == "Multi-shock Displacement/Needs"
    assert hazard.hclass == "all"
