"""
check_serper.py — minimal, friendly Serper API key checker

Usage:
  poetry run python spagbot/check_serper.py

What it does:
- Loads SERPER_API_KEY from environment (and tries to read .env if present)
- Calls Serper's /news endpoint with a tiny query
- Prints a clear PASS/FAIL with helpful hints
"""

from __future__ import annotations
import os, json, sys, time
from typing import Optional
import requests

SERPER_NEWS_URL = "https://google.serper.dev/news"
TIMEOUT_SEC = 12

def _maybe_load_dotenv(dotenv_path: str = ".env") -> None:
    """
    Very small .env loader (so we don't need extra deps).
    It only sets variables that aren't already in os.environ.
    Lines like KEY=VALUE; quotes around VALUE are ok.
    Lines starting with # are ignored.
    """
    try:
        if not os.path.exists(dotenv_path):
            return
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception:
        # Non-fatal: just skip if any parsing issue
        pass

def check_serper(api_key: Optional[str]) -> int:
    if not api_key:
        print("❌ SERPER_API_KEY is not set. Set it in your .env or environment.")
        print("   Example in .env:\n     ENABLE_SERPER=1\n     SERPER_API_KEY=YOUR_REAL_KEY\n")
        return 1

    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": "metaculus", "num": 1}

    print(f"→ Testing Serper at {SERPER_NEWS_URL} ...")
    try:
        r = requests.post(SERPER_NEWS_URL, headers=headers, json=payload, timeout=TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("   Hints: check your internet connection, corporate proxy, or DNS. "
              "If behind a proxy, set HTTPS_PROXY / HTTP_PROXY env vars.")
        return 1

    code = r.status_code
    head = (r.text or "")[:200].replace("\n", " ")

    if code == 200:
        # try to confirm the shape
        try:
            j = r.json()
            if isinstance(j, dict) and "news" in j:
                print("✅ PASS — Key appears valid, got 200 and a 'news' result.")
                return 0
        except Exception:
            pass
        print("✅ PASS — Got 200 OK. Response head:", head)
        return 0
    elif code == 429:
        print("✅ Key works, but you are rate-limited (HTTP 429).")
        print("   Action: wait for your quota window or upgrade your Serper plan.")
        return 0
    elif code in (401, 403):
        print(f"❌ FAIL — Unauthorized ({code}). The key is missing/invalid or not permitted.")
        print("   Check that SERPER_API_KEY is correct, active, and copied without spaces.")
        return 1
    elif code == 400:
        print(f"❌ FAIL — Bad request (400). Response head: {head}")
        print("   This usually means the payload was malformed; re-run without edits.")
        return 1
    else:
        print(f"❌ FAIL — Unexpected status {code}. Response head: {head}")
        return 1

if __name__ == "__main__":
    # Load .env if present (does not overwrite existing env vars)
    _maybe_load_dotenv(".env")
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    rc = check_serper(api_key)
    sys.exit(rc)