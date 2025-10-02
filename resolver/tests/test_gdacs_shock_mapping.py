from resolver.ingestion.gdacs_client import hazard_from_key, load_config, map_hazard


def test_gdacs_hazard_mapping_known_types():
    cfg = load_config()
    hazard_map = cfg.get("hazard_map", {})
    default_hazard = cfg.get("default_hazard", "other")

    eq = map_hazard("EQ", hazard_map, default_hazard)
    tc = map_hazard("TC", hazard_map, default_hazard)
    fl = map_hazard("FL", hazard_map, default_hazard)
    vo = map_hazard("VO", hazard_map, default_hazard)
    other = map_hazard("UNK", hazard_map, default_hazard)

    assert eq.code == "earthquake" and eq.hazard_class == "geophysical"
    assert tc.code == "tropical_cyclone" and tc.hazard_class == "meteorological"
    assert fl.code == "flood" and fl.hazard_class == "hydrological"
    assert vo.code == "volcano" and vo.hazard_class == "volcanic"
    assert other.code == hazard_from_key("other").code
