# ANCHOR: config 
from __future__ import annotations
import os, json, math
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
    IST_TZ = ZoneInfo("Europe/Istanbul")
except Exception:
    IST_TZ = timezone(timedelta(hours=3))

# Load .env early if present
try:
    import dotenv; dotenv.load_dotenv()
except Exception:
    pass

# --- Toggles and API keys ---
SUBMIT_PREDICTION = (os.getenv("SUBMIT_PREDICTION", "0") == "1")
USE_OPENROUTER    = (os.getenv("USE_OPENROUTER", "1") == "1")
USE_GOOGLE        = (os.getenv("USE_GOOGLE", "1") == "1")
ENABLE_GROK       = (os.getenv("ENABLE_GROK", "1") == "1")

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")           # e.g., https://openrouter.ai/api/v1
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
XAI_API_KEY     = os.getenv("XAI_API_KEY")
XAI_BASE_URL    = "https://api.x.ai/v1/chat/completions"
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET    = os.getenv("ASKNEWS_SECRET")

# --- Models & timeouts ---
OPENROUTER_GPT5_ID        = os.getenv("OPENROUTER_GPT5_ID", "openai/gpt-4o")
OPENROUTER_GPT5_THINK_ID  = os.getenv("OPENROUTER_GPT5_THINK_ID", "openai/gpt-4o")
OPENROUTER_FALLBACK_ID    = os.getenv("OPENROUTER_FALLBACK_ID", "openai/gpt-4o")
OPENROUTER_CLAUDE37_ID    = os.getenv("OPENROUTER_CLAUDE37_ID", "anthropic/claude-3.7-sonnet")
GEMINI_MODEL              = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
XAI_GROK_ID               = os.getenv("XAI_GROK_ID", "grok-4")

GPT5_CALL_TIMEOUT_SEC   = float(os.getenv("GPT5_CALL_TIMEOUT_SEC", 300))
GEMINI_CALL_TIMEOUT_SEC = float(os.getenv("GEMINI_CALL_TIMEOUT_SEC", 300))
GROK_CALL_TIMEOUT_SEC   = float(os.getenv("GROK_CALL_TIMEOUT_SEC", 300))

# --- Decoding knobs (separate for research vs. forecasting) ---
FORECAST_TEMP = float(os.getenv("FORECAST_TEMP", "0.00"))
RESEARCH_TEMP = float(os.getenv("RESEARCH_TEMP", "0.20"))
FORECAST_TOP_P = float(os.getenv("FORECAST_TOP_P", "0.20"))
RESEARCH_TOP_P = float(os.getenv("RESEARCH_TOP_P", "1.00"))

# --- Concurrency & cache ---
CONCURRENT_REQUESTS_LIMIT = 5
DISABLE_RESEARCH_CACHE = os.getenv("SPAGBOT_DISABLE_RESEARCH_CACHE", "0").lower() in ("1","true","yes")

# --- Markets & tournament ---
ENABLE_MARKET_SNAPSHOT = os.getenv("ENABLE_MARKET_SNAPSHOT", "1").lower() in ("1","true","yes")
MARKET_SNAPSHOT_MAX_MATCHES = int(os.getenv("MARKET_SNAPSHOT_MAX_MATCHES", 3))
METACULUS_INCLUDE_RESOLVED  = os.getenv("METACULUS_INCLUDE_RESOLVED", "1").lower() in ("1","true","yes")
TOURNAMENT_ID = os.getenv("TOURNAMENT_ID", "fall-aib-2025")

API_BASE_URL  = "https://www.metaculus.com/api"
METACULUS_HTTP_TIMEOUT = float(os.getenv("METACULUS_HTTP_TIMEOUT", "30"))
AUTH_HEADERS  = {"headers": {"Authorization": f"Token {METACULUS_TOKEN}"}}

TEST_POSTS_FILE = os.getenv("TEST_POSTS_FILE", "data/test_questions.json")
TEST_POST_IDS_ENV = os.getenv("TEST_POST_IDS", "").strip()

# --- Files & dirs ---
FORECASTS_CSV      = "forecasts.csv"
FORECASTS_BY_MODEL = "forecasts_by_model.csv"
FORECAST_LOG_DIR   = "forecast_logs"
RUN_LOG_DIR        = "logs"
CACHE_DIR          = "cache"
MCQ_WIDE_CSV       = "forecasts_mcq_wide.csv"
MAX_MCQ_OPTIONS    = 20

# --- Calibration note path ---
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "data/calibration_advice.txt")

# --- Time helpers ---
def ist_stamp(fmt: str = "%Y%m%d-%H%M%S") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

def ist_iso(fmt: str = "%Y-%m-%d %H:%M:%S %z") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

def ist_date(fmt: str = "%Y-%m-%d") -> str:
    from datetime import datetime
    return datetime.now(IST_TZ).strftime(fmt)

# --- Cache helpers (JSON) ---
import os, json
def cache_path(kind: str, slug: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{kind}__{slug}.json")

def read_cache(kind: str, slug: str) -> dict | None:
    p = cache_path(kind, slug)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def write_cache(kind: str, slug: str, data: dict) -> None:
    try:
        with open(cache_path(kind, slug), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Misc small utils ---
def clip01(x: float) -> float:
    return max(0.01, min(0.99, float(x)))

def fmt_float_or_blank(x) -> str:
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return ""
        return f"{xf:.6f}"
    except Exception:
        return ""
