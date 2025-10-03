#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-}" 
if [[ -z "$BASE" ]]; then
  BASE="origin/main"
fi
python tools/context_pack.py --base "$BASE"
echo "Context pack created under ./context/"
