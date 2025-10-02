from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_EXTS = {".py", ".yml", ".yaml", ".csv", ".md", ".txt"}
SEARCH_DIRS = [
    ROOT / "ingestion",
    ROOT / "tools",
    ROOT / "data",
    ROOT / "tests",
]
FORBIDDEN = "armed_conflict_cessation"


def test_conflict_cessation_removed():
    shocks_path = ROOT / "data" / "shocks.csv"
    text = shocks_path.read_text(encoding="utf-8")
    assert FORBIDDEN not in text

    offenders = []
    this_file = Path(__file__).resolve()
    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path == this_file:
                continue
            if path.suffix.lower() not in SEARCH_EXTS:
                continue
            try:
                contents = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if FORBIDDEN in contents:
                offenders.append(path)
    assert not offenders, f"found forbidden term in: {offenders}"
