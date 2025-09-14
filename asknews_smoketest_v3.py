# asknews_smoketest_v3.py
import os, sys, json
from pathlib import Path

def load_dotenv_if_present():
    try:
        import dotenv
        for p in [Path(".")/".env", Path("..")/".env"]:
            if p.exists():
                dotenv.load_dotenv(p)
                break
    except Exception:
        pass

def show_ver():
    try:
        import importlib.metadata as m
        print("asknews version =", m.version("asknews"))
    except Exception as e:
        print("asknews version lookup failed:", repr(e))

def extract_items(res):
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
    return [it for it in data if isinstance(it, dict) and (it.get("title") or it.get("headline"))]

def try_one(sdk, label, method, params):
    print(f"\n[try] {label}  params={params}")
    res = None
    try:
        if method == "news.search_news" and hasattr(sdk, "news") and hasattr(sdk.news, "search_news"):
            res = sdk.news.search_news(**params)   # type: ignore
        elif method == "news.search" and hasattr(sdk, "news") and hasattr(sdk.news, "search"):
            res = sdk.news.search(**params)        # type: ignore
        elif method == "search_news" and hasattr(sdk, "search_news"):
            res = sdk.search_news(**params)        # type: ignore
        elif method == "search" and hasattr(sdk, "search"):
            res = sdk.search(**params)             # type: ignore
        else:
            print(f"[skip] method {method} not found")
            return
    except Exception as e:
        print("[error] exception:", type(e).__name__, repr(e))
        for attr in ("status","status_code","response"):
            if hasattr(e, attr):
                print(f"    {attr} =", getattr(e, attr))
        if hasattr(e, "response") and getattr(e, "response") is not None:
            try:
                print("    response.text:", e.response.text)  # type: ignore
            except Exception:
                pass
        return
    items = extract_items(res)
    print(f"[result] items={len(items)}")
    if isinstance(res, dict):
        print("[raw keys]", list(res.keys()))
        for k in ("error", "errors", "message", "detail"):
            if k in res:
                print(f"[raw {k}]", res[k])

def main():
    load_dotenv_if_present()
    cid = os.getenv("ASKNEWS_CLIENT_ID")
    sec = os.getenv("ASKNEWS_SECRET") or os.getenv("ASKNEWS_CLIENT_SECRET")
    print("ASKNEWS_CLIENT_ID present:", bool(cid))
    print("ASKNEWS_SECRET present:", bool(sec))
    show_ver()
    if not cid or not sec:
        print("Missing creds; add to .env and retry.")
        sys.exit(1)

    try:
        from asknews_sdk import AskNewsSDK
        print("import asknews_sdk ✅")
    except Exception as e:
        print("Import failed. Install/update with: poetry add asknews")
        print("Import error:", repr(e)); sys.exit(1)

    # Build SDK with explicit scope
    try:
        sdk = AskNewsSDK(client_id=cid, client_secret=sec, scopes=["news"])
        print("SDK constructed ✅ (scopes=['news'])")
    except Exception as e:
        print("SDK construction failed ❌:", repr(e)); sys.exit(1)

    # Try multiple methods & param styles, minimal query, no filters
    methods = ["news.search_news","news.search","search_news","search"]
    pairs = [
        ("n_articles", dict(query="ukraine", n_articles=5, strategy="latest", return_type="both")),
        ("limit",     dict(query="ukraine", limit=5, strategy="latest", return_type="both")),
    ]
    for m in methods:
        for label, params in pairs:
            try_one(sdk, f"{m} ({label})", m, params)

if __name__ == "__main__":
    main()
