#!/usr/bin/env python3
"""
meta_category_probe.py
======================

Purpose
-------
A tiny diagnostic script to check whether Metaculus exposes "classification"
information (categories / tags / domains / subdomains) for questions via the API.

What it does
------------
- Loads your .env to get METACULUS_TOKEN (and optional TOURNAMENT_ID).
- Calls /api/posts/ with a limit/offset and (optionally) a tournament filter.
- Prints out any "classification-ish" fields found for each post/question:
  * post-level keys like: categories, tags, domains, subdomains, topics
  * nested question-level keys like: categories, tags, category_ids, topic_names
- Saves a raw JSON snapshot to ./debug/meta_categories_probe_<timestamp>.json
  so you can open it in VS Code and inspect the schema comfortably.

Why this helps
--------------
If categories/tags are present, we can use them in Spagbot directly and avoid
building our own taxonomy. If not (or if inconsistent), we’ll propose and attach
a stable custom taxonomy and a lightweight classifier.

Notes
-----
- Requires: requests, python-dotenv
- Auth header must be: {"Authorization": f"Token <METACULUS_TOKEN>"}
- Endpoint: https://www.metaculus.com/api/posts/
- Parameters used: limit, offset, include_description=true, statuses=open,
  forecast_type in [binary, multiple_choice, numeric, discrete]
- If --post-ids is provided, we’ll fetch details per ID in addition to the list call.

"""

from __future__ import annotations
import os, sys, json, argparse, datetime, pathlib, time
from typing import Any, Dict, List, Tuple, Optional

# 1) Load environment (so your METACULUS_TOKEN and optional TOURNAMENT_ID are available)
try:
    import dotenv
    dotenv.load_dotenv()
except Exception:
    # Non-fatal: the script will still attempt os.getenv if .env isn't present
    pass

import requests

API_BASE_URL = "https://www.metaculus.com/api"
DEBUG_DIR = pathlib.Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

# A list of keys we’ll scan on both the post object and its nested "question" object.
# We include several plausible names because Metaculus has evolved its taxonomy over time.
LIKELY_CLASS_KEYS = [
    "categories", "category", "category_id", "category_ids", "category_names",
    "tags", "tag_names",
    "domains", "subdomains",
    "topics", "topic_names",
    "labels", "label_names",
]

def get_auth_header() -> Dict[str, str]:
    """
    Builds the 'Authorization: Token ...' header from METACULUS_TOKEN in .env.
    """
    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        print("ERROR: METACULUS_TOKEN is not set in your environment / .env.", file=sys.stderr)
        sys.exit(2)
    return {"Authorization": f"Token {token}"}

def fetch_posts(limit: int = 25,
                offset: int = 0,
                tournament: Optional[str] = None,
                statuses: str = "open") -> Dict[str, Any]:
    """
    Hits /api/posts/ to fetch a page of posts (which include questions).

    We set include_description=true to get more context (sometimes schemas vary).
    """
    url = f"{API_BASE_URL}/posts/"
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order_by": "-hotness",
        "include_description": "true",
        "statuses": statuses,  # "open", "resolved", "all"
        # We'll request common forecast types to see a variety of shapes:
        "forecast_type": ",".join(["binary", "multiple_choice", "numeric", "discrete"]),
    }
    if tournament:
        # Metaculus accepts tournaments as a list in the params
        params["tournaments"] = [tournament]

    r = requests.get(url, headers=get_auth_header(), params=params, timeout=30)
    if not r.ok:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text}")
    return r.json()

def fetch_post_detail(post_id: int) -> Dict[str, Any]:
    """
    Hits /api/posts/<id>/ to fetch a single post in more detail (if accessible).
    Useful in case the list call omits some fields.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    r = requests.get(url, headers=get_auth_header(), timeout=30)
    if not r.ok:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text}")
    return r.json()

def pull_classification_fields(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility: Given a JSON dict (post or question), return any keys that look like classification.
    """
    found: Dict[str, Any] = {}
    for k in LIKELY_CLASS_KEYS:
        if k in obj and obj[k] not in (None, [], {}):
            found[k] = obj[k]
    return found

def summarize_one_post(post: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts a friendly summary for the console:
    - id, title, URL if present
    - question.type (binary / multiple_choice / numeric / discrete)
    - “classification” fields found at the post and at the nested question level
    """
    post_id = post.get("id")
    title = post.get("title") or post.get("name") or "(no title)"
    url = post.get("page_url") or post.get("url") or post.get("absolute_url")

    q = post.get("question") or {}
    q_type = q.get("type") or q.get("question_type")
    post_class = pull_classification_fields(post)
    q_class = pull_classification_fields(q)

    out = {
        "post_id": post_id,
        "title": title,
        "url": url,
        "question_type": q_type,
        "post_classification_fields": post_class,
        "question_classification_fields": q_class,
    }
    return out

def save_snapshot(name_prefix: str, payload: Any) -> pathlib.Path:
    """
    Writes a JSON snapshot (pretty-printed) to ./debug/ for post-run inspection.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DEBUG_DIR / f"{name_prefix}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path

def main():
    parser = argparse.ArgumentParser(description="Probe Metaculus API for categories/tags/domains.")
    parser.add_argument("--limit", type=int, default=25, help="Number of posts to request (default 25).")
    parser.add_argument("--offset", type=int, default=0, help="Offset for pagination (default 0).")
    parser.add_argument("--tournament", type=str, default=os.getenv("TOURNAMENT_ID", "").strip(),
                        help="Tournament ID/slug to filter by (default comes from .env TOURNAMENT_ID if present).")
    parser.add_argument("--statuses", type=str, default="open", choices=["open", "resolved", "all"],
                        help="Filter posts by status (default: open).")
    parser.add_argument("--post-ids", type=str, default="",
                        help="Comma-separated post IDs to fetch detail for (optional).")
    args = parser.parse_args()

    print("\n=== Metaculus Classification Probe ===")
    print(f"Using API base: {API_BASE_URL}")
    print(f"Tournament filter: {args.tournament or '(none)'} | Statuses: {args.statuses}")
    print(f"Limit: {args.limit} | Offset: {args.offset}")

    # 1) Page of posts (fast way to check fields)
    posts_page = fetch_posts(limit=args.limit, offset=args.offset,
                             tournament=(args.tournament or None),
                             statuses=args.statuses)
    page_path = save_snapshot("meta_categories_posts_page", posts_page)
    print(f"\nSaved raw page snapshot → {page_path}")

    results = posts_page.get("results") or posts_page.get("data") or []
    if not results:
        print("No posts returned. Try increasing --limit or removing tournament filter.")
        return

    # 2) Summarize classification fields that appear
    print("\n--- Detected classification fields per post ---")
    any_found = False
    for post in results:
        summary = summarize_one_post(post)
        print(f"\n[Post {summary['post_id']}] {summary['title']}")
        print(f"  URL: {summary['url']}")
        print(f"  Question type: {summary['question_type']}")
        # Print post-level fields
        if summary["post_classification_fields"]:
            any_found = True
            print("  Post-level classification fields:")
            for k, v in summary["post_classification_fields"].items():
                print(f"    - {k}: {v}")
        # Print question-level fields
        if summary["question_classification_fields"]:
            any_found = True
            print("  Question-level classification fields:")
            for k, v in summary["question_classification_fields"].items():
                print(f"    - {k}: {v}")

    if not any_found:
        print("\n(No obvious category/tag fields detected on this page. We’ll try per-ID detail next.)")

    # 3) Optionally pull a few posts by ID for detail (some fields only show on detail endpoints)
    ids: List[int] = []
    if args.post_ids.strip():
        try:
            ids = [int(x.strip()) for x in args.post_ids.strip().split(",") if x.strip()]
        except ValueError:
            print("WARN: --post-ids must be integers separated by commas. Ignoring.", file=sys.stderr)

    detail_blobs: Dict[int, Dict[str, Any]] = {}
    for pid in ids:
        try:
            blob = fetch_post_detail(pid)
            detail_blobs[pid] = blob
        except Exception as e:
            print(f"  (Failed to fetch post {pid}: {e})")
            continue

    if detail_blobs:
        detail_path = save_snapshot("meta_categories_post_details", detail_blobs)
        print(f"\nSaved per-ID detail snapshot → {detail_path}")
        print("\n--- Detected classification fields in per-ID details ---")
        for pid, post in detail_blobs.items():
            summary = summarize_one_post(post)
            print(f"\n[Post {pid}] {summary['title']}")
            if summary["post_classification_fields"]:
                print("  Post-level classification fields:")
                for k, v in summary["post_classification_fields"].items():
                    print(f"    - {k}: {v}")
            if summary["question_classification_fields"]:
                print("  Question-level classification fields:")
                for k, v in summary["question_classification_fields"].items():
                    print(f"    - {k}: {v}")

    print("\nDone.\n")

if __name__ == "__main__":
    main()