"""
Runtime helpers for the template bot: environment validation, startup/result
banners, and suppression of noisy upstream warnings.

Kept separate from main.py so that file can focus on the bot's forecasting
logic. main_with_no_framework.py keeps its own inline copies on purpose --
it's meant to be a single-file reference implementation.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import traceback
import unicodedata
import warnings
from typing import Any, Sequence


# Placeholder values shipped in .env.template. If a real env var still equals
# one of these the user forgot to replace it; we'd rather fail loudly here than
# inside the SDK three layers down.
_PLACEHOLDER_ENV_VALUES = {
    "1234567890",
    "REPLACE_ME",
    "your-token-here",
    "your-api-key-here",
}


def _is_real_env(name: str) -> bool:
    val = os.getenv(name)
    return bool(val and val.strip() and val.strip() not in _PLACEHOLDER_ENV_VALUES)


def silence_noisy_dependencies() -> None:
    """
    Quiet warnings from transitive deps that fire on import and confuse new
    users. Must be called *before* importing forecasting_tools.
    """
    warnings.filterwarnings(
        "ignore", message=r".*does not support cost tracking.*"
    )
    logging.getLogger("forecasting_tools.ai_models.model_tracker").setLevel(
        logging.ERROR
    )
    # Streamlit installs its own logger hierarchy; suppress via its own API.
    try:
        from streamlit.logger import set_log_level

        set_log_level("error")
    except ImportError:
        pass
    # LiteLLM is verbose at INFO; its WARNING level is enough for us.
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False


# Prices litellm may not know, per million tokens, as (input, output) in USD.
# Read from OpenRouter's /models API on 2026-10-02; keep in step with the
# comment above backtest.pin_models.DEFAULTS.
#
# Why this exists: litellm loads its cost map by downloading it at import
# time and falls back to the copy bundled in the wheel only when that
# download fails. The locked wheel (1.80.10) predates claude-opus-5.5, so on
# a run whose download failed the forecaster's calls are priced at $0 and the
# per-question cost in the logs is silently wrong. The call itself does not
# break: forecasting_tools' ModelTracker only warns, and litellm swallows the
# cost-calculation error inside its logging callback. Measured by reading the
# installed code, not assumed (general_llm.py, model_tracker.py,
# litellm_logging.py::_response_cost_calculator).
#
# The other three were added on 2026-10-22 (same API): the spend ledger
# (spend.py) showed GPT-5.6 Sol, Haiku 4.5 and gpt-4o-mini priced at $0 on
# Test Bot run 39578224912 -- 97 of the run's 128 OpenRouter calls -- and the
# SDK's own per-question cost with them.
KNOWN_MODEL_PRICES_PER_MTOK = {
    "openrouter/anthropic/claude-opus-5.5": (4.0, 20.0),
    "openrouter/openai/gpt-5.6-sol": (4.0, 20.0),
    "openrouter/anthropic/claude-haiku-4.5": (1.0, 5.0),
    "openrouter/openai/gpt-4o-mini": (0.15, 0.6),
}


def register_model_prices() -> list[str]:
    """Teach litellm the prices in KNOWN_MODEL_PRICES_PER_MTOK it is missing.

    Fills gaps only. A model litellm already prices is left alone, so a newer
    downloaded map always wins over this table and a price change upstream is
    never masked by it; an entry with no input price counts as a gap. Returns the model ids registered, for the log line.
    Must run before any GeneralLlm is built: ModelTracker decides whether to
    warn in GeneralLlm.__init__.
    """
    import litellm

    registered = []
    for model, (input_per_mtok, output_per_mtok) in KNOWN_MODEL_PRICES_PER_MTOK.items():
        if (litellm.model_cost.get(model) or {}).get("input_cost_per_token"):
            continue
        litellm.register_model(
            {
                model: {
                    "input_cost_per_token": input_per_mtok / 1_000_000,
                    "output_cost_per_token": output_per_mtok / 1_000_000,
                    "litellm_provider": model.split("/", 1)[0],
                    "mode": "chat",
                }
            }
        )
        registered.append(model)
    return registered


# Log records whose CONTENT is a forecast on a live tournament question.
# Everything else -- discovery counts, provider/bucket lines, rate-limit waits,
# "Posted prediction/comment" -- carries no forecast content and is left alone,
# because that is what every audit of this bot has relied on.
#
# LAYER 2 of two. Layer 1 is log_forecast_content(), which stops the content
# reaching logging at all for the sites main.py owns. These patterns exist so a
# revert of layer 1, or an upstream forecasting_tools record we do not own,
# still cannot put forecast content in a public log.
_FORECAST_CONTENT_PATTERNS = (
    # main.py: the full research body, logged with the question URL
    re.compile(r"^(Found Research for URL \S+)", re.S),
    # main.py: the individual prediction value, for every question type
    re.compile(r"^(Forecasted URL \S+) with prediction: .*", re.S),
    # main.py: the model's full rationale, for every question type. The
    # forecasting prompts end with 'Probability: ZZ%', so this record carried
    # both the reasoning and the number. Missing from the original filter --
    # production runs 32239144510 / 32268972325 / 32366841649 emitted it whole.
    re.compile(r"^(Reasoning for URL \S+)", re.S),
)
# forecast_bot.log_report_summary: summaries and first rationales, in one record
_SUMMARY_MARKERS = ("<<<<<<<<<<<<<<<<<<<< Summary", "First Rationale")

# LAYER 3. Pattern matching on a known headline only protects records whose
# headline we already know about. These match the forecast VALUE itself,
# wherever it appears -- an exception message, a traceback tail, a
# forecasting_tools record nobody has read yet. A record carrying any of these
# is redacted whole, because there is no safe prefix to keep when the number
# can be anywhere in it.
#
# Deliberately narrow, so operational lines survive: `rpm=15.0`,
# `latency_s=1.05`, `wait_ms=0` and `Posted prediction on question 45375` all
# contain digits and none of them match.
#
# NO \b ANCHORS, and matched case-insensitively against a NORMALISED copy.
# `\bProbability:` looks tighter but is trivially evaded: in
# "\x1b[31mProbability: 12%" the escape ends in `m`, so `m` meets `P` and
# there is no word boundary at all. The adversarial suite caught that; it was
# not visible by reading the pattern. Anything that can prefix a word
# character to the token defeats a leading \b, so the anchors are gone.
_FORECAST_VALUE_SIGNALS = (
    re.compile(r"Probability\s*:\s*\d", re.I),   # the binary prompt's own format
    re.compile(r"probability\s*=\s*\d", re.I),   # PredictedOption repr
    re.compile(r"PredictedOption\s*\(", re.I),   # multiple-choice repr
    re.compile(r"predicted_options\s*=", re.I),  # PredictedOptionList repr
    re.compile(r"declared_percentiles\s*=", re.I),  # NumericDistribution repr
    re.compile(r"Percentile\s*\(", re.I),        # numeric/date percentile repr
    re.compile(r"prediction_in_decimal", re.I),  # BinaryPrediction field name
    # The comment body's OWN forecast display, from
    # forecast_bot._create_comment: "*Final Prediction*: 42%". A Metaculus
    # error response can echo the submitted comment back, which puts the
    # forecast inside an exception message -- caught by tests/test_publication
    # ::test_error_details_are_bounded_and_value_free, not by inspection.
    re.compile(r"\*\s*Final Prediction\s*\*", re.I),
)

# Upstream records that carry a payload of provider/model text rather than a
# forecast value. The headline is operationally useful; the payload is not
# ours to publish. Keyed on the literal prefix forecasting_tools emits.
_UPSTREAM_PAYLOAD_PREFIXES = (
    "Encountered errors while researching:",
    "Encountered errors while predicting:",
    "Could not summarize research.",
    "Exception occurred during forecasting:",
    "Error while processing question url:",
)

# Anything longer than this in a sanitised error string is provider prose, not
# a diagnostic. 200 was the pre-existing banner budget; keeping it means the
# banner's shape does not change.
_ERROR_TEXT_LIMIT = 200


# Control and escape sequences a terminal would render but a matcher would
# trip over. Stripped before matching, never from what is emitted.
_ANSI_AND_CONTROL = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]")

# The interpreter-generated locator line of a traceback frame:
#     File "/path/to/file.py", line 137, in some_function
# Path, line number and symbol name only -- never a runtime value. The frame's
# SOURCE line, which follows it, is not covered and is scrubbed.
_TRACEBACK_LOCATOR = re.compile(r'\s*File "[^"]*", line \d+(, in .*)?$')


def _normalise_for_matching(text: str) -> str:
    """A canonical copy to run the signals against.

    Matching happens on this; redaction always applies to the ORIGINAL record,
    so normalisation can be as aggressive as it likes without altering output.

      * NFKC folds fullwidth and compatibility forms, so a model answering
        with fullwidth characters cannot slip a value past a plain match.
      * ANSI/control sequences are removed, because they let arbitrary bytes
        sit between or beside the characters of a token.
    """
    return _ANSI_AND_CONTROL.sub("", unicodedata.normalize("NFKC", text))


def carries_forecast_value(text: str) -> bool:
    """Whether this text contains a forecast value in any recognised shape."""
    candidate = _normalise_for_matching(text)
    return any(signal.search(candidate) for signal in _FORECAST_VALUE_SIGNALS)


# "module.path.ExceptionType" at the head of a traceback's final line. Symbol
# names only, so this half is safe to keep -- and it is the half an operator
# actually triages on.
_EXCEPTION_TYPE = re.compile(r"^(\s*)([A-Za-z_][\w.]*)(:\s)(.*)$", re.S)


def _scrub_exception_line(line: str) -> str:
    """Keep the exception TYPE, scrub the message after it.

    `ValueError: <anything the model produced>` becomes
    `ValueError: [redacted...]`. Dropping the whole line loses the single most
    useful triage signal in a traceback, which the adversarial suite caught.
    """
    match = _EXCEPTION_TYPE.match(line)
    if not match:
        return scrub(line)
    indent, exception_type, separator, detail = match.groups()
    return indent + exception_type + separator + scrub(detail)


def scrub(text: object, limit: int = _ERROR_TEXT_LIMIT) -> str:
    """A log-safe rendering of arbitrary error text.

    Used by BOTH the logging filter and print_run_summary_banner, so the
    stdout path and the stderr path cannot drift apart -- the banner used to
    truncate to 200 characters with no content check at all, which let a
    provider body or a parsed probability through whenever an exception
    message happened to carry one.

    Collapses newlines (a traceback in a banner is unreadable anyway), drops
    the record entirely if it carries a forecast value, and truncates.
    """
    rendered = str(text).replace("\n", " ").replace("\r", " ")
    if carries_forecast_value(rendered):
        return "[redacted: carried a forecast value]"
    if len(rendered) > limit:
        rendered = rendered[:limit] + "..."
    return rendered


class RedactForecastContent(logging.Filter):
    """Keep forecast reasoning out of the process's log stream.

    Why this exists
    ---------------
    This repository is a PUBLIC fork, so GitHub Actions logs are world
    readable. A tournament run used to emit, in the clear: the full research
    body, the report summary and first rationale, and each of the five
    individual predictions -- for a question that was still OPEN.

    Two distinct problems, one cause:

      * Metaculus requires bots to leave PRIVATE comments, published later at
        Metaculus' own intervals. Publishing the same reasoning immediately in
        a public log defeats that, whatever the log's medium.
      * FutureEval forbids a bot maker from previewing their bot's forecasts on
        open questions and then updating the bot. Printing the predictions puts
        that preview in front of the maker on every run. Removing it makes the
        guarantee structural instead of a promise -- the same reasoning that
        keeps research/ limited to closed questions.

    It redacts CONTENT, never the fact that work happened: the question URL,
    the call counts, the provider and bucket lines and the publication
    confirmations all survive, so operational auditing is unaffected.

    Not applied in test_questions mode: bot-testing-area is an unscored
    practice area, and full output there is what makes it useful.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not kill the run
            message = ""

        # ALWAYS, and before any message check can return early. exc_info and
        # stack_info are rendered by the HANDLER, after every filter has run,
        # and getMessage() does not include them -- so a record whose message
        # is perfectly operational can still print an exception carrying the
        # model's output. litellm alone raises from 144 logger.exception()
        # sites and its exceptions embed the provider response body.
        self._sanitise_attached_traceback(record)

        if any(marker in message for marker in _SUMMARY_MARKERS):
            return self._replace(
                record,
                "[redacted: per-question summary and rationale withheld from a "
                "public log while the question may still be open]",
            )

        for pattern in _FORECAST_CONTENT_PATTERNS:
            match = pattern.match(message)
            if match:
                return self._replace(record, match.group(1) + " [content redacted]")

        # Upstream payload records: keep the headline, drop what follows. The
        # headline is what tells an operator that a question errored at all.
        for prefix in _UPSTREAM_PAYLOAD_PREFIXES:
            if message.startswith(prefix):
                return self._replace(
                    record, prefix + " [detail redacted from a public log]"
                )

        # Last line of defence: the value itself, wherever it sits. Nothing is
        # preserved here on purpose -- if a probability can appear anywhere in
        # the record, no prefix of it is provably safe to keep.
        if carries_forecast_value(message):
            return self._replace(
                record,
                "[redacted: record carried a forecast value; see the Metaculus "
                "comment for the forecast itself]",
            )
        return True

    @staticmethod
    def _sanitise_attached_traceback(record: logging.LogRecord) -> None:
        """Scrub the exception text a handler would append to this record.

        Kept verbatim: the `File "...", line N, in func` locator lines and the
        structural markers. Those are the operationally useful half of a
        traceback -- they say WHERE it failed -- and they are generated by the
        interpreter from path, line number and symbol names, so no runtime
        value can reach them.

        Scrubbed: everything else. That means the exception message lines,
        which is where a provider response body or a parser complaint quoting
        the model lands, AND the frames' source-text lines. The source line
        was originally preserved on the argument that it shows the literal
        (`f"...{reasoning}"`) rather than the value -- true for this
        repository's own code, but it is still arbitrary text from an
        arbitrary dependency, and the adversarial suite demonstrated a frame
        whose literal carried the value outright.
        """
        if not record.exc_info and not record.exc_text and not record.stack_info:
            return

        if record.stack_info and carries_forecast_value(record.stack_info):
            record.stack_info = "[stack redacted: carried a forecast value]"

        rendered = record.exc_text
        if rendered is None and record.exc_info:
            try:
                rendered = "".join(traceback.format_exception(*record.exc_info))
            except Exception:  # noqa: BLE001 - a broken record must not kill the run
                rendered = "[exception could not be formatted]"
        if not rendered:
            return

        safe_lines = []
        for line in rendered.splitlines():
            if _TRACEBACK_LOCATOR.match(line) or line.startswith(
                ("Traceback", "During handling", "The above exception")
            ):
                safe_lines.append(line)
            else:
                safe_lines.append(_scrub_exception_line(line))
        # exc_info must be cleared: Formatter only uses exc_text when
        # exc_info is falsy or exc_text is already set, and leaving the live
        # exception around invites a second formatter re-rendering it raw.
        record.exc_text = "\n".join(safe_lines)
        record.exc_info = None

    @staticmethod
    def _replace(record: logging.LogRecord, text: str) -> bool:
        """Overwrite a record in place. args must be cleared too, or getMessage
        re-applies %-formatting to the replacement and raises or re-expands."""
        record.msg = text
        record.args = ()
        # The message was bad enough to replace wholesale, so the attached
        # exception goes too -- _sanitise_attached_traceback has already run,
        # but a scrubbed traceback beside a fully redacted message tells an
        # operator nothing it does not already know.
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


# Whether this process is running against a SCORED tournament. Defaults to
# True -- fail closed. A caller that never declares a mode gets the safe
# behaviour, so forgetting to call install_forecast_redaction() cannot be the
# thing that publishes a rationale.
_withhold_forecast_content = True


def install_forecast_redaction(run_mode: str) -> bool:
    """Attach the redaction filter for scored modes. Returns whether it was.

    Also sets the process-wide flag log_forecast_content() reads, so layer 1
    (suppress at the source) and layer 2 (filter the stream) are switched by
    one decision and cannot disagree about which mode this is.
    """
    global _withhold_forecast_content
    if run_mode == "test_questions":
        # bot-testing-area is unscored; full output is what makes it useful.
        _withhold_forecast_content = False
        return False
    _withhold_forecast_content = True
    root = logging.getLogger()
    _add_filter_once(root)
    # Filters on the root logger are not consulted for records emitted by child
    # loggers, so the handlers get it too - that is the path every record takes.
    for handler in root.handlers:
        _add_filter_once(handler)
    # ...and every handler that already exists anywhere else, plus every
    # handler added from now on. A handler attached to a CHILD logger is
    # called BEFORE the root handlers and would otherwise never see this
    # filter: `logging.getLogger("some.library").addHandler(...)` was measured
    # emitting an unredacted rationale while the root handler emitted the
    # redacted one. Nothing in the current pipeline does that, but
    # silence_noisy_dependencies() exists precisely because dependencies
    # rearrange their own loggers, and Streamlit installs a whole hierarchy.
    for existing in list(logging.Logger.manager.loggerDict.values()):
        for handler in getattr(existing, "handlers", []) or []:
            _add_filter_once(handler)
    _install_handler_hook()
    return True


#: ONE filter object for the whole process. Not one per call: every record is
#: run through every filter on its handler, so a fresh instance per call makes
#: log handling O(number of calls) per record. Repeated installation is
#: normal -- tests do it constantly, and nothing stops a caller doing it twice.
_REDACTION_FILTER = RedactForecastContent()


def _add_filter_once(target: Any) -> None:
    """Attach the redaction filter unless this target already has one.

    Idempotent by identity AND by type, so a target that somehow acquired a
    different RedactForecastContent instance is not given a second one.
    """
    try:
        for existing in getattr(target, "filters", []) or []:
            if isinstance(existing, RedactForecastContent):
                return
        target.addFilter(_REDACTION_FILTER)
    except Exception:  # noqa: BLE001 - logging setup must never kill a run
        pass


def _install_handler_hook() -> None:
    """Make every future addHandler() carry the redaction filter.

    Patched once and idempotently. Deliberately narrow: it adds a filter and
    changes nothing else about logging, so a dependency that installs its own
    handler still gets its handler -- it just cannot get an unfiltered one.
    """
    if getattr(logging.Logger.addHandler, "_redaction_hook", False):
        return
    original = logging.Logger.addHandler

    def add_handler_with_redaction(self, hdlr):  # type: ignore[no-untyped-def]
        _add_filter_once(hdlr)
        return original(self, hdlr)

    add_handler_with_redaction._redaction_hook = True  # type: ignore[attr-defined]
    logging.Logger.addHandler = add_handler_with_redaction  # type: ignore[assignment]


def forecast_content_is_withheld() -> bool:
    """Whether forecast content is being kept out of this process's output."""
    return _withhold_forecast_content


def log_forecast_content(
    target_logger: logging.Logger, headline: str, content: object
) -> None:
    """Log `headline`, and `content` only where publishing it is safe.

    LAYER 1, and the one that actually matters: on a scored run the content is
    never handed to logging at all, so no handler, formatter, third-party
    filter or future refactor can put it on a public stream. The filter in this
    module stays as layer 2 for the records this function does not own.

    `headline` must be free of forecast content by construction -- it is a URL
    and a fixed label. It is deliberately kept, because every audit of this bot
    reads which questions were worked on out of these lines.
    """
    if _withhold_forecast_content:
        target_logger.info("%s [content withheld from a public log]", headline)
    else:
        target_logger.info("%s: %s", headline, content)


def check_environment(strict: bool = True) -> None:
    """
    Verify METACULUS_TOKEN is set; warn if no LLM key is configured. On
    failure with strict=True, exits the process with a non-zero status.
    """
    problems: list[str] = []

    if not _is_real_env("METACULUS_TOKEN"):
        problems.append(
            "METACULUS_TOKEN is missing or still a placeholder. "
            "Get one at https://www.metaculus.com/futureeval/participate/"
        )

    has_llm_key = any(
        _is_real_env(k)
        for k in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    )
    if not has_llm_key:
        print(
            "⚠️  No LLM key set (OPENROUTER/OPENAI/ANTHROPIC). The bot will fall back\n"
            "    to the Metaculus LLM proxy. Free OpenRouter credits: "
            "https://forms.gle/aQdYMq9Pisrf1v7d8\n"
        )

    if problems:
        print("❌  Setup problems:")
        for p in problems:
            print(f"    • {p}")
        if strict:
            sys.exit(1)


def print_startup_banner(run_mode: str, will_publish: bool) -> None:
    publish = "publish=yes" if will_publish else "publish=no (dry run)"
    print(f"🤖  Running mode={run_mode}, {publish}\n")


def print_run_summary_banner(
    forecast_reports: Sequence[Any],
    will_publish: bool,
    tournament_url: str | None = None,
) -> None:
    """
    End-of-run summary printed via print() (not logger) so it survives log
    filtering. Shows count, per-question URLs, and any failure tracebacks.
    If tournament_url is given, it's included as a footer link.
    """
    # Lazy AND defensive. This banner is the last thing a run does, after the
    # forecasts are already published, so it must never be the thing that
    # turns a successful run into a failed one. If the symbol is unavailable
    # the partition degrades to "anything that is not an exception is a
    # report", which is exactly what forecast_questions returns anyway.
    try:
        from forecasting_tools import ForecastReport

        def _is_report(item: Any) -> bool:
            return isinstance(item, ForecastReport)

    except Exception:  # noqa: BLE001 - any import problem must degrade, not raise

        def _is_report(item: Any) -> bool:
            return not isinstance(item, BaseException)

    valid = [r for r in forecast_reports if _is_report(r)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]
    banner = "=" * 80

    print()
    print(banner)

    if not forecast_reports:
        print("ℹ️   No new questions to forecast on this run.")
        print(banner)
        print()
        return

    if valid and not exceptions:
        verb = "submitted" if will_publish else "produced (dry run)"
        print(f"🎉  Bot {verb} {len(valid)} forecast(s).")
    elif valid and exceptions:
        print(
            f"⚠️   Partial — {len(valid)} succeeded, {len(exceptions)} failed."
        )
    else:
        print(f"❌  All {len(exceptions)} attempt(s) failed.")

    if valid:
        print()
        for r in valid:
            note = f"  (with {len(r.errors)} minor error(s))" if r.errors else ""
            print(f"  ✅ {r.question.page_url}{note}")
        if will_publish and tournament_url:
            print(f"\n  Tournament: {tournament_url}")

    if exceptions:
        print()
        for exc in exceptions:
            # scrub(), not a bare truncation: this is print(), so no logging
            # filter sees it. A provider error body or a parsed probability
            # inside an exception message used to reach stdout untouched.
            print(f"  ❌ {type(exc).__name__}: {scrub(exc)}")

    print(banner)
    print()
