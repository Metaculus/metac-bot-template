# asknews_smoketest.py
# Minimal, self-contained credential + connectivity check for AskNews.

import os, sys, json, re
from pathlib import Path

def load_dotenv_if_present():
    try:
        import dotenv
        # Try local folder, then parent
        for p in [Path(".") / ".env", Path("..") / ".env"]:
            if p.exists():
                dotenv.load_dotenv(p)
                break
    except Exception:
        pass

def print_secret_ok(name, value):
    if not value:
        print(f"{name}: MISSING")
        return False
    masked = value[:2] + "…" + value[-2:] if len(value) > 6 else "***"
    print(f"{name}: present ({masked})")
    return True

def extract_items(res):
    """Normalize possible return shapes into a list of dicts with a title."""
    if res is None:
        return []
    data = res
    if isinstance(res, dict):
        for k in ("results","articles","hits","data"):
            if isinstance(res.get(k), list):
                data = res[k]
                break
    if not isinstance(data, list):
        data = [data] if isinstance(data, dict) else []
    out = []
    for it in data:
        if not isinstance(it, dict): 
            continue
        title = (it.get("title") or it.get("headline") or "").strip()
        if title:
            out.append(it)
    return out

def try_one(sdk, query, strategy):
    params = dict(query=query, n_articles=5, return_type="both", strategy=strategy)
    # Try common SDK shapes
    res = None
    if hasattr(sdk, "news") and hasattr(sdk.news, "search"):
        res = sdk.news.search(**params)
    elif hasattr(sdk, "search"):
        res = sdk.search(**params)
    elif hasattr(sdk, "search_news"):
        res = sdk.search_news(**params)  # type: ignore
    items = extract_items(res)
    print(f"[smoke] strategy={strategy:9s} -> {len(items)} item(s)")
    if items:
        print("        first title:", (items[0].get("title") or items[0].get("headline")))
    return len(items)

def main():
    load_dotenv_if_present()
    cid = os.getenv("ASKNEWS_CLIENT_ID")
    sec = os.getenv("ASKNEWS_SECRET")

    ok1 = print_secret_ok("ASKNEWS_CLIENT_ID", cid)
    ok2 = print_secret_ok("ASKNEWS_SECRET", sec)
    if not (ok1 and ok2):
        print("=> Add these to your .env and rerun.")
        sys.exit(1)

    try:
        from asknews_sdk import AskNewsSDK
    except Exception as e:
        print("asknews_sdk import failed. Install with:  poetry add asknews-sdk")
        print("Error:", repr(e))
        sys.exit(1)

    # Construct SDK
    try:
        sdk = AskNewsSDK(client_id=cid, client_secret=sec)
        print("SDK constructed ✅")
    except Exception as e:
        print("SDK construction failed ❌:", repr(e))
        sys.exit(1)

    # Three simple queries, no lang or since filters
    try:
        total = 0
        total += try_one(sdk, "ukraine", "latest")
        total += try_one(sdk, "artificial intelligence", "relevance")
        total += try_one(sdk, "election", "top")
        if total == 0:
            print("\nNo items returned across simple queries.")
            print("- If this persists, it's often an auth/plan issue (401/403) hidden by the SDK,")
            print("  or an API version mismatch. Try upgrading the SDK:  poetry update asknews-sdk")
    except Exception as e:
        # Try to surface HTTP details if available
        print("Search call raised an exception ❌")
        print(repr(e))
        # Some SDKs attach response; best-effort inspection:
        for attr in ("status", "status_code", "response"):
            if hasattr(e, attr):
                print(attr, "=", getattr(e, attr))
        if hasattr(e, "response") and getattr(e, "response") is not None:
            try:
                print("response.text:", e.response.text)  # type: ignore
            except Exception:
                pass

if __name__ == "__main__":
    main()
