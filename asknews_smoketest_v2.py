# asknews_smoketest_v2.py
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
        print("asknews-sdk version =", m.version("asknews-sdk"))
    except Exception as e:
        print("asknews-sdk version lookup failed:", repr(e))

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
    out=[]
    for it in data:
        if isinstance(it, dict) and (it.get("title") or it.get("headline")):
            out.append(it)
    return out

def try_search(sdk, label, params):
    print(f"\n[try] {label} params={params}")
    res=None
    try:
        if hasattr(sdk, "news") and hasattr(sdk.news, "search"):
            res = sdk.news.search(**params)
        elif hasattr(sdk, "search"):
            res = sdk.search(**params)
        elif hasattr(sdk, "search_news"):
            res = sdk.search_news(**params)  # type: ignore
        else:
            print("No search method found on SDK object.")
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
    # Show top-level keys of raw result for clues
    if isinstance(res, dict):
        print("[raw keys]", list(res.keys()))
        # Print any obvious error fields
        for k in ("error", "errors", "message", "detail"):
            if k in res:
                print(f"[raw {k}]", res[k])
    elif isinstance(res, list):
        print("[raw] list[...] length", len(res))

def main():
    load_dotenv_if_present()
    cid = os.getenv("ASKNEWS_CLIENT_ID")
    sec = os.getenv("ASKNEWS_SECRET")
    print("ASKNEWS_CLIENT_ID present:", bool(cid))
    print("ASKNEWS_SECRET present:", bool(sec))
    show_ver()
    if not cid or not sec:
        print("Missing creds; add them to .env and retry.")
        sys.exit(1)
    try:
        from asknews_sdk import AskNewsSDK
    except Exception as e:
        print("Import failed. Install/update with: poetry add -U asknews-sdk")
        print("Import error:", repr(e))
        sys.exit(1)

    try:
        sdk = AskNewsSDK(client_id=cid, client_secret=sec)
        print("SDK constructed ✅")
    except Exception as e:
        print("SDK construction failed ❌:", repr(e))
        sys.exit(1)

    # Minimal, no filters. We try both n_articles and limit param names.
    base_queries = [
        ("latest", "ukraine"),
        ("relevance", "artificial intelligence"),
        ("top", "election"),
    ]
    for strat, q in base_queries:
        try_search(sdk, f"news.search n_articles (strat={strat})",
                   dict(query=q, n_articles=5, return_type="both", strategy=strat))
        try_search(sdk, f"news.search limit (strat={strat})",
                   dict(query=q, limit=5, return_type="both", strategy=strat))

    # Also try with no return_type/strategy (SDK defaults)
    try_search(sdk, "defaults (only query)", dict(query="ukraine"))

if __name__ == "__main__":
    main()
