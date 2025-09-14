# asknews_rateprobe.py — print AskNews rate-limit headers and behavior for your account.
import os, time, json
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

def headers_of(obj):
    # Try common places the SDK stores headers
    for attr in ("headers",):
        if hasattr(obj, attr):
            try:
                h = getattr(obj, attr)
                if isinstance(h, dict):
                    return h
            except Exception:
                pass
    # Some exceptions carry a .response object
    if hasattr(obj, "response"):
        r = getattr(obj, "response")
        for attr in ("headers",):
            if hasattr(r, attr):
                try:
                    h = getattr(r, attr)
                    if isinstance(h, dict):
                        return h
                except Exception:
                    pass
    return {}

def main():
    load_dotenv_if_present()
    cid = os.getenv("ASKNEWS_CLIENT_ID"); sec = os.getenv("ASKNEWS_SECRET") or os.getenv("ASKNEWS_CLIENT_SECRET")
    from asknews_sdk import AskNewsSDK
    sdk = AskNewsSDK(client_id=cid, client_secret=sec, scopes=["news"])
    q = os.getenv("ASKNEWS_PROBE_QUERY", "ukraine")
    n = int(os.getenv("ASKNEWS_PROBE_N", "10"))
    strategy = os.getenv("ASKNEWS_PROBE_STRATEGY", "news knowledge")
    tries = int(os.getenv("ASKNEWS_PROBE_TRIES", "5"))
    wait = int(os.getenv("ASKNEWS_PROBE_WAIT_SEC", "2"))
    print(f"Query={q!r}, n_articles={n}, strategy={strategy!r}, tries={tries}, wait={wait}s")

    for i in range(1, tries+1):
        print(f"\n[probe] attempt {i}")
        try:
            resp = sdk.news.search_news(query=q, n_articles=n, strategy=strategy, return_type="both")
            # Prefer headers on success too (SDK may expose them)
            hs = headers_of(resp)
            if hs:
                print("  headers:", {k: hs.get(k) for k in ["X-RateLimit-Limit","X-RateLimit-Remaining","X-RateLimit-Reset","Retry-After"]})
            # Pull items count
            data = getattr(resp, "as_dict", None)
            if data and isinstance(data, dict):
                articles = data.get("articles") or data.get("results") or data.get("data") or []
                print("  items:", len(articles))
            else:
                print("  items: n/a (no as_dict)")
        except Exception as e:
            print("  exception:", type(e).__name__, "-", getattr(e, "message", repr(e)))
            hs = headers_of(e)
            if hs:
                print("  headers:", {k: hs.get(k) for k in ["X-RateLimit-Limit","X-RateLimit-Remaining","X-RateLimit-Reset","Retry-After"]})
        time.sleep(wait)

if __name__ == "__main__":
    main()