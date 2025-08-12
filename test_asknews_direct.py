#!/usr/bin/env python3
"""
Direct test of AskNews API to isolate the 429 issue.
This bypasses forecasting-tools entirely to test raw API access.
"""

import asyncio
import os

from dotenv import load_dotenv


async def test_asknews_direct():
    load_dotenv()

    client_id = os.getenv("ASKNEWS_CLIENT_ID")
    secret = os.getenv("ASKNEWS_SECRET")

    print(f"Testing AskNews API directly...")
    print(f"Client ID: {client_id[:8]}... (length: {len(client_id)})")
    print(f"Secret: {secret[:8]}... (length: {len(secret)})")

    try:
        from asknews_sdk import AsyncAskNewsSDK

        print("✓ Successfully imported AsyncAskNewsSDK")

        async with AsyncAskNewsSDK(
            client_id=client_id,
            client_secret=secret,
            scopes=set(["news"]),
        ) as sdk:
            print("✓ Successfully created AsyncAskNewsSDK with 'news' scope")

            print("Testing latest news search (like forecasting-tools)...")
            hot_response = await sdk.news.search_news(
                query="Will AI",
                n_articles=6,
                return_type="both",
                strategy="latest news",
            )
            print(f"✓ Latest news: Got {len(hot_response.as_dicts)} articles")

            print("Testing historical news search (like forecasting-tools)...")
            historical_response = await sdk.news.search_news(
                query="Will AI",
                n_articles=10,
                return_type="both",
                strategy="news knowledge",
            )
            print(f"✓ Historical news: Got {len(historical_response.as_dicts)} articles")

            print("✓ Both calls succeeded - same as forecasting-tools pattern!")

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_asknews_direct())
