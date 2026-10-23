"""An uncaught exception must not print a forecast to a public log.

The log redaction (bot_helpers.RedactForecastContent) only sees what goes
through logging. Python prints an uncaught exception straight to stderr, so
when every question of a step failed, the RuntimeError the SDK's
log_report_summary raises escaped main.py and printed its message -- each
failed forecaster's pydantic ValidationError, declared percentiles and all --
in the public logs of runs 39453516101 and 39965881683, for questions still
open. Two layers now: main.summarise_reports exits 1 with a fixed line
instead of raising, and scored runs install an excepthook that scrubs any
traceback the way the log filter does.

Every value here is synthetic: the repository is public.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import subprocess
import sys
import threading
import unittest

import bot_helpers
from tests._real_forecasting_tools import real_forecasting_tools

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#: Synthetic declared percentiles, in the shape the SDK's ValidationError
#: quotes them.
SECRET = "4.24242"
VALIDATION = (
    "ValidationError: 1 validation error for NumericDistribution\n  Value error, "
    "No declared percentiles are within the range of the question +/- 25.0%. "
    f"Percentiles: [Percentile(percentile=0.1, value={SECRET}), "
    "Percentile(percentile=0.9, value=4.75757)]"
)
SUMMARY = ("1 errors occurred while forecasting: [RuntimeError(\"Error while processing "
           f"question url: 'https://www.metaculus.com/questions/900001': {VALIDATION}\")]")


def load_main():
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "main_uncaught", os.path.join(ROOT, "main.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class FakeBot:
    def log_report_summary(self, reports):
        if any(isinstance(r, BaseException) for r in reports):
            raise RuntimeError(SUMMARY)


class RestoreRedactionMixin:
    def setUp(self):
        super().setUp()
        saved = (bot_helpers._withhold_forecast_content, sys.excepthook, threading.excepthook)

        def restore():
            (bot_helpers._withhold_forecast_content, sys.excepthook,
             threading.excepthook) = saved

        self.addCleanup(restore)


class SummariseReportsTests(RestoreRedactionMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main()

    def summarise(self, reports):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), \
                self.assertLogs(self.main.logger, level="WARNING") as logs:
            try:
                self.main.summarise_reports(FakeBot(), reports)
                code = None
            except SystemExit as exc:
                code = exc.code
        return code, stderr.getvalue(), logs.output

    def test_a_step_where_everything_failed_exits_1_without_the_message(self):
        bot_helpers.install_forecast_redaction("market_pulse")
        code, stderr, logs = self.summarise([ValueError(VALIDATION), ValueError(VALIDATION)])
        self.assertEqual(code, 1)
        self.assertEqual(logs, [f"ERROR:{self.main.logger.name}:all_failed count=2"])
        for text in [stderr] + logs:
            self.assertNotIn(SECRET, text)
            self.assertNotIn("Percentile(", text)

    def test_a_partial_failure_only_warns(self):
        bot_helpers.install_forecast_redaction("tournament")
        code, _stderr, logs = self.summarise([ValueError(VALIDATION), object()])
        self.assertIsNone(code)
        self.assertEqual(len(logs), 1)
        self.assertIn("partial_failure failed=1 of=2", logs[0])

    def test_the_unscored_practice_mode_still_raises(self):
        bot_helpers.install_forecast_redaction("test_questions")
        with self.assertRaises(RuntimeError):
            self.main.summarise_reports(FakeBot(), [ValueError(VALIDATION)])

    def test_main_calls_it_instead_of_the_sdk_summary(self):
        with io.open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            src = handle.read()
        block = src[src.index('if __name__ == "__main__":'):]
        self.assertIn("summarise_reports(template_bot, forecast_reports)", block)
        self.assertNotIn("log_report_summary", block)


class ExcepthookTests(RestoreRedactionMixin, unittest.TestCase):
    def raised(self):
        try:
            try:
                raise ValueError(VALIDATION)
            except ValueError as inner:
                raise RuntimeError(SUMMARY) from inner
        except RuntimeError as exc:
            return type(exc), exc, exc.__traceback__

    def test_scored_runs_install_it_and_practice_runs_do_not(self):
        bot_helpers.install_forecast_redaction("market_pulse")
        self.assertIs(sys.excepthook, bot_helpers._redacting_excepthook)
        self.assertIs(threading.excepthook, bot_helpers._redacting_thread_excepthook)
        bot_helpers.install_forecast_redaction("test_questions")
        self.assertIs(sys.excepthook, sys.__excepthook__)
        self.assertIs(threading.excepthook, threading.__excepthook__)

    def test_the_traceback_keeps_its_shape_but_not_the_values(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            bot_helpers._redacting_excepthook(*self.raised())
        out = stderr.getvalue()
        self.assertNotIn(SECRET, out)
        self.assertNotIn("Percentile(", out)
        self.assertIn("Traceback (most recent call last):", out)
        self.assertIn("\nRuntimeError: ", out)
        self.assertIn("[redacted: carried a forecast value]", out)
        self.assertIn(", in raised", out)

    def test_an_uncaught_exception_in_a_scored_process(self):
        """End to end, in a fresh interpreter on the real SDK: an uncaught
        RuntimeError wrapping a real ValidationError."""
        code = (
            "import bot_helpers\n"
            "from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile\n"
            "bot_helpers.install_forecast_redaction('market_pulse')\n"
            "try:\n"
            "    NumericDistribution(declared_percentiles=[\n"
            f"        Percentile(percentile=0.1, value={SECRET}),\n"
            "        Percentile(percentile=0.9, value=4.75757)],\n"
            "        open_upper_bound=True, open_lower_bound=True,\n"
            "        upper_bound=14.0, lower_bound=10.0, zero_point=None)\n"
            "except Exception as exc:\n"
            "    raise RuntimeError(f'1 errors occurred while forecasting: [{exc!r}]')\n"
        )
        done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(done.returncode, 1)
        self.assertIn("\nRuntimeError: ", done.stderr)
        self.assertIn("[redacted: carried a forecast value]", done.stderr)
        self.assertNotIn(SECRET, done.stderr + done.stdout)


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
