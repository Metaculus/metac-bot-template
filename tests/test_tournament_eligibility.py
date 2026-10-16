"""Static guards on the invariants that keep the bot eligible.

These are not unit tests of behaviour; they are assertions about configuration
that, if silently changed, would put tournament eligibility at risk without
breaking anything visible. Each one names the rule it protects.

Source: FutureEval Bot Tournament Resources Page.
"""

from __future__ import annotations

import io
import os.path
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    return io.open(os.path.join(ROOT, *parts), encoding="utf-8").read()


class PrivateCommentTests(unittest.TestCase):
    """RULE: "Bots are required to leave comments on questions they forecast,
    though bots should only leave private comments"."""

    def test_publishing_is_enabled_so_comments_are_actually_posted(self):
        self.assertIn("publish_reports_to_metaculus=publish_to_metaculus", read("main.py"))
        self.assertIn("publish_to_metaculus = True", read("main.py"))

    def test_we_never_force_a_comment_public(self):
        """The SDK defaults is_private=True; overriding it would be the only
        way to make a bot comment public."""
        for path in ("main.py", "bot_helpers.py"):
            self.assertNotIn("is_private=False", read(path))
            self.assertNotIn("is_private = False", read(path))


class NoPreviewTests(unittest.TestCase):
    """RULE: "A bot maker cannot preview how their bot forecasts on open or
    upcoming questions and then update their bot based on that preview"."""

    def test_the_research_lab_defaults_to_closed_questions_only(self):
        src = read("research", "fetch_own_track_record.py")
        self.assertIn('CLOSED_STATUSES = ["resolved", "closed"]', src)

    def test_open_questions_require_an_explicit_opt_in(self):
        src = read("research", "fetch_own_track_record.py")
        self.assertIn('"--include-open"', src)
        self.assertIn('statuses.append("open")', src)

    def test_no_workflow_ever_passes_include_open(self):
        """Comment lines are excluded: research_track_record.yaml documents the
        flag precisely to record that it is NOT passed."""
        workflows = os.path.join(ROOT, ".github", "workflows")
        for name in os.listdir(workflows):
            for line in read(".github", "workflows", name).splitlines():
                if line.strip().startswith("#"):
                    continue
                self.assertNotIn(
                    "--include-open", line,
                    "{0} would preview forecasts on open questions".format(name),
                )

    def test_forecast_content_is_redacted_on_scored_modes(self):
        """Printing predictions on open questions is itself a preview."""
        self.assertIn("install_forecast_redaction(run_mode)", read("main.py"))


class NoReforecastTests(unittest.TestCase):
    """RULE: "A bot maker cannot decide that they don't like their bot's
    forecast and rerun the bot on that question"."""

    def test_the_tournament_path_skips_already_forecasted_questions(self):
        self.assertIn("skip_previously_forecasted_questions=True", read("main.py"))

    def test_the_only_override_is_the_practice_area(self):
        """Three branches set it False: the unscored bot-testing-area; Market
        Pulse, whose market_pulse.select already did the skipping (new
        questions plus a refresh before its spot-scored close); and the Cup,
        whose market_pulse.select_stale did (new questions plus week-old
        forecasts). Never the scored FutureEval/MiniBench branch."""
        src = read("main.py")
        overrides = [
            i for i, line in enumerate(src.splitlines())
            if "skip_previously_forecasted_questions = False" in line
        ]
        self.assertEqual(len(overrides), 3, "unexpected number of overrides")
        lines = src.splitlines()
        for index in overrides:
            context = "\n".join(lines[max(0, index - 12):index + 3])
            self.assertTrue(
                'run_mode == "test_questions"' in context
                or 'run_mode == "market_pulse"' in context
                or 'run_mode == "metaculus_cup"' in context,
                "an override sits outside test_questions, market_pulse and the Cup",
            )

    def test_the_tournament_branch_does_not_disable_dedup(self):
        src = read("main.py")
        start = src.index('if run_mode == "tournament":')
        end = src.index('elif run_mode == "metaculus_cup":')
        self.assertNotIn("skip_previously_forecasted_questions", src[start:end])


class AlibabaStaysOutTests(unittest.TestCase):
    """Alibaba was probed and its credential rejected (401, both regions). It
    must not be in any production path."""

    def test_it_is_absent_from_the_fallback_chain(self):
        src = read("backtest", "pin_models.py")
        chain = src[src.index("FALLBACK_CHAIN = ["):src.index("]", src.index("FALLBACK_CHAIN = ["))]
        for marker in ("dashscope", "alibaba", "qwen"):
            self.assertNotIn(marker, chain.lower())

    def test_the_production_workflow_does_not_carry_its_credential(self):
        self.assertNotIn(
            "ALIBABA_API_KEY",
            read(".github", "workflows", "run_bot_on_tournament.yaml"),
        )

    def test_it_lives_only_in_the_analysis_only_probe(self):
        self.assertIn("dashscope", read("research", "smoke_test_providers.py"))
        for path in ("main.py", "bot_helpers.py"):
            self.assertNotIn("dashscope", read(path).lower())


class TestsNeverPublishTests(unittest.TestCase):
    """A test that publishes a forecast would be a rule violation committed by
    the test suite itself."""

    #: Names that would actually write to Metaculus if CALLED.
    WRITE_CALLS = frozenset({
        "publish_report_to_metaculus",
        "post_question_comment",
        "post_binary_question_prediction",
        "post_numeric_question_prediction",
        "post_multiple_choice_question_prediction",
    })

    #: Assignments that replace the real transport on the real client class.
    #: A file containing BOTH has made it impossible for any code in it to
    #: reach Metaculus, whatever it calls afterwards.
    TRANSPORT_NEUTRALISERS = (
        "MetaculusClient.post_question_comment =",
        "MetaculusClient._post_question_prediction =",
    )

    def _test_sources(self):
        for name in sorted(os.listdir(os.path.join(ROOT, "tests"))):
            # This file names the forbidden calls in order to forbid them.
            if not name.endswith(".py") or name == os.path.basename(__file__):
                continue
            yield name, read("tests", name)

    @classmethod
    def _neutralises_the_transport(cls, src):
        return all(marker in src for marker in cls.TRANSPORT_NEUTRALISERS)

    def test_no_test_enables_publishing(self):
        """Same exemption as the write-endpoint guard below, for the same
        reason: a file that has provably replaced BOTH network methods on the
        real MetaculusClient cannot publish, whatever it enables. The end-to-end
        pipeline test has to enable publishing to exercise publication at all."""
        for name, src in self._test_sources():
            if self._neutralises_the_transport(src):
                continue
            self.assertNotIn("publish_reports_to_metaculus=True", src, name)

    def test_no_test_calls_a_metaculus_write_endpoint(self):
        """AST, not text: a docstring that MENTIONS a write helper is fine --
        test_read_api.py names them to assert the read-only client has none.
        What must not exist is a call expression.

        A file that has provably replaced BOTH network methods on the real
        MetaculusClient is exempt, because the property this guard exists to
        protect -- "no test can reach Metaculus" -- is then guaranteed by
        construction rather than by a naming convention. tests/test_publication
        does exactly that: the publication state machine cannot be tested
        without calling the two methods it wraps.

        This is deliberately stricter than an exemption list keyed on
        filenames, which a new file could join without proving anything.
        """
        import ast

        for name, src in self._test_sources():
            if self._neutralises_the_transport(src):
                continue
            for node in ast.walk(ast.parse(src)):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = getattr(func, "attr", None) or getattr(func, "id", None)
                self.assertNotIn(
                    called, self.WRITE_CALLS,
                    "{0} CALLS {1}()".format(name, called),
                )

    def test_every_exempt_file_also_restores_the_real_transport(self):
        """An exemption that leaked would poison every later test module."""
        exempt = [
            name for name, src in self._test_sources()
            if self._neutralises_the_transport(src)
        ]
        self.assertTrue(exempt, "the exemption path must stay exercised")
        for name in exempt:
            src = read("tests", name)
            self.assertIn("addCleanup(setattr, MetaculusClient", src, name)

    def test_the_neutralisation_check_is_not_trivially_satisfiable(self):
        """Guards the guard: a file that only mentions one half is not exempt."""
        self.assertFalse(self._neutralises_the_transport(
            "MetaculusClient.post_question_comment = fake"
        ))
        self.assertFalse(self._neutralises_the_transport("nothing here"))
        self.assertTrue(self._neutralises_the_transport(
            "MetaculusClient.post_question_comment = a\n"
            "MetaculusClient._post_question_prediction = b\n"
        ))

    def test_the_read_only_api_client_refuses_non_get(self):
        """research/ must stay incapable of writing to Metaculus."""
        self.assertIn("GET", read("research", "metaculus_read_api.py"))


class ProductionTargetTests(unittest.TestCase):
    def test_the_tournament_workflow_runs_the_unmodified_entry_point(self):
        """The scored poll loop uses the default mode. The only --mode flags
        in the file are the steps after the loop: the Cup (practice) and
        Market Pulse (prize, refreshed before close)."""
        src = read(".github", "workflows", "run_bot_on_tournament.yaml")
        self.assertIn("poetry run python main.py 2>&1 | tee", src)
        self.assertEqual(
            re.findall(r"--mode\s+(\S+)", src), ["metaculus_cup", "market_pulse"],
            "production must use the default mode for the scored tournament",
        )


if __name__ == "__main__":
    unittest.main()
