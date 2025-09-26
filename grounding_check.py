"""
grounding_check.py

Purpose:
  Simple one-file diagnostic to verify that:
    1) Your Gemini API key is working, and
    2) Google Search grounding is active (the response includes grounding metadata).

How to run:
  - Ensure your environment has GEMINI_API_KEY (or GOOGLE_API_KEY) set.
  - python grounding_check.py   (or `poetry run python grounding_check.py`)

Interpreting results:
  - If grounding is active, you'll see:
      - A normal text answer, and
      - In the printed JSON, a 'groundingMetadata' block (e.g., webSearchQueries, groundingSupports).
  - If grounding is inactive, you may see no 'groundingMetadata' in the JSON.
"""

import os
import json
import sys
import textwrap

try:
    import google.generativeai as genai
except Exception as e:
    print("❌ Could not import google.generativeai. Install with:")
    print("   pip install google-generativeai   (or)   poetry add google-generativeai")
    print("Error:", repr(e))
    sys.exit(1)

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("❌ No GEMINI_API_KEY / GOOGLE_API_KEY found in environment.")
    print("   Set one, e.g.:  setx GEMINI_API_KEY \"YOUR_KEY\"  then restart the terminal.")
    sys.exit(1)

# 1) Configure client
genai.configure(api_key=API_KEY)

# 2) Build a prompt that should require current web info (to trigger grounding)
prompt = (
    "What is the current population of Istanbul? Provide a short answer and cite the source."
)

# 3) Ask Gemini 2.5 Pro with the Google Search grounding tool enabled
model = genai.GenerativeModel("gemini-2.5-pro")
try:
    resp = model.generate_content(
        prompt,
        tools=[{"google_search": {}}]  # this is the switch that enables grounding via Google Search
    )
except Exception as e:
    print("❌ Gemini API call failed:", repr(e))
    sys.exit(1)

# 4) Show the model text
print("\n=== Model text ===")
print(textwrap.fill(resp.text or "(no text)", width=100))

# 5) Dump the raw response as a dict and check grounding fields
resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp
print("\n=== Grounding metadata present? ===")
gm = None
try:
    # candidates[0].groundingMetadata is where Google puts it
    candidates = resp_dict.get("candidates") or []
    if candidates:
        gm = candidates[0].get("groundingMetadata")
except Exception:
    gm = None

if gm:
    print("✅ groundingMetadata FOUND")
    # Print a compact summary
    print(json.dumps({
        "webSearchQueries": gm.get("webSearchQueries"),
        "groundingSupports_count": len(gm.get("groundingSupports", [])),
        "groundingChunks_count": len(gm.get("groundingChunks", [])),
    }, indent=2))
else:
    print("⚠️  groundingMetadata not found in response. Grounding may be disabled or not triggered.")

# 6) (Optional) print the full JSON for deep inspection
# WARNING: can be large
# print("\n=== Full response JSON ===")
# print(json.dumps(resp_dict, indent=2))
