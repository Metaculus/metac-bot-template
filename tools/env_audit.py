# tools/env_audit.py
import os, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "tools") else Path.cwd()
PATTERNS = [
    re.compile(r'os\.getenv\(\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]'),
    re.compile(r'os\.environ\[\s*[\'"]([A-Za-z_][A-Za-z0-9_]*)[\'"]\s*\]')
]

def find_env_names(root: Path):
    names = set()
    for p in root.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in PATTERNS:
            for m in pat.finditer(text):
                names.add(m.group(1))
    return sorted(names)

if __name__ == "__main__":
    names = find_env_names(ROOT)
    print("\n--- Environment variables referenced in this repo ---")
    for n in names:
        print(n)
    print("\nTotal:", len(names))
