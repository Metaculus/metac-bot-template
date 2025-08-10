from __future__ import annotations

from typing import Sequence

from forecasting_tools import ForecastBot, ForecastReport


def compact_log_report_summary(
    forecast_reports: Sequence[ForecastReport | BaseException],
    raise_errors: bool = True,
) -> None:
    """Print exactly one line per forecast and summarise exceptions.

    This mirrors the behaviour previously monkey-patched in *main.py* but
    lives in a dedicated module so we avoid global patching side-effects.
    """

    valid_reports = [r for r in forecast_reports if isinstance(r, ForecastReport)]
    exceptions = [r for r in forecast_reports if isinstance(r, BaseException)]

    def _line(r: ForecastReport) -> str:
        readable = type(r).make_readable_prediction(r.prediction).strip()
        return f"✅ {r.question.page_url} | Prediction: {readable} | Minor Errors: {len(r.errors)}"

    import logging  # postponed import to keep module load light

    summary_lines = "\n".join(_line(r) for r in valid_reports)

    for exc in exceptions:
        msg = str(exc)
        if len(msg) > 300:
            msg = msg[:297] + "…"
        summary_lines += f"\n❌ Exception: {exc.__class__.__name__} | {msg}"

    logger = logging.getLogger(__name__)
    logger.info(summary_lines + "\n")

    minor_lists = [r.errors for r in valid_reports if r.errors]
    if minor_lists:
        logger.error(f"{len(minor_lists)} minor error groups occurred while forecasting: {minor_lists}")

    if exceptions and raise_errors:
        raise RuntimeError(f"{len(exceptions)} errors occurred while forecasting: {exceptions}")


class CompactLoggingForecastBot(ForecastBot):
    """ForecastBot subclass that uses the compact report summary."""

    # Override class attribute with compact logger
    log_report_summary = staticmethod(compact_log_report_summary)  # type: ignore[misc]
