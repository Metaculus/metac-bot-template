"""Guarantees this bot makes to production, asserted as guarantees.

Every other test file checks that a component behaves. This one checks that
the SYSTEM still promises what an operator was told it promises. The tests are
deliberately written against properties rather than implementations, so a
refactor that preserves the guarantee passes and a refactor that quietly drops
it fails.

Grouped by the promise, not by the module:

  Security      no secret, rationale, probability or provider body in a log
  Integrity     five predictions stay five; nothing partial is ever published
  Publication   one prediction, one comment, orphans named out loud
  Concurrency   loop-safe semaphore, bounded research, atomic chain selection
  Dead code     no statement that can never run, so nothing reads as live
  Discovery     every page read, no question processed twice
  Workflow      scored tournaments only from the one production workflow
  Deployment    the code CI generates still keeps all of the above
"""

from __future__ import annotations

import ast
import io
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with io.open(os.path.join(ROOT, *parts), encoding="utf-8") as handle:
        return handle.read()


def parse(*parts: str) -> ast.Module:
    return ast.parse(read(*parts))


def yaml_without_comments(*parts: str) -> str:
    """Workflow text with `#` comment lines removed.

    Necessary because these workflows document what they deliberately do NOT
    contain -- ci.yaml literally says a future edit adding `env: secrets.*`
    must be caught -- so a plain substring search finds the prose rather than
    the key.
    """
    kept = []
    for line in read(*parts).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        kept.append(line.split(" #", 1)[0])
    return "\n".join(kept)


def find_function(tree: ast.AST, name: str) -> ast.AST | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


PRODUCTION_SOURCES = (
    ("main.py",),
    ("bot_helpers.py",),
    ("publication.py",),
    ("discovery.py",),
    ("backtest", "fallback_llm.py"),
    ("backtest", "balanced_llm.py"),
    ("backtest", "rate_limiter.py"),
    ("backtest", "pin_models.py"),
)


# ===========================================================  SECURITY


class NoCredentialMaterialInProductionCode(unittest.TestCase):
    """Credentials are referenced by env var NAME, never by value."""

    #: Shapes of real provider credentials. Deliberately requires enough
    #: trailing characters that a prose mention ("an AIza key") cannot match.
    CREDENTIAL_SHAPES = (
        re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
        re.compile(r"\bgsk_[0-9A-Za-z]{20,}"),
        re.compile(r"\bsk-or-v1-[0-9A-Za-z]{20,}"),
        re.compile(r"\bsk-[0-9A-Za-z]{32,}"),
    )

    def test_no_production_file_contains_credential_shaped_text(self):
        for parts in PRODUCTION_SOURCES:
            src = read(*parts)
            for shape in self.CREDENTIAL_SHAPES:
                self.assertIsNone(
                    shape.search(src),
                    "{0} contains credential-shaped text".format("/".join(parts)),
                )


class NoForecastContentReachesLogs(unittest.TestCase):
    """The full proof lives in test_forecast_content_leakage; these are the
    structural preconditions that make it hold."""

    def test_both_redaction_layers_exist(self):
        src = read("bot_helpers.py")
        self.assertIn("def log_forecast_content", src, "layer 1 (source)")
        self.assertIn("class RedactForecastContent", src, "layer 2 (filter)")

    def test_the_source_layer_fails_closed(self):
        src = read("bot_helpers.py")
        self.assertIn("_withhold_forecast_content = True", src)

    def test_the_banner_scrubs_rather_than_truncates(self):
        src = read("bot_helpers.py")
        self.assertIn("scrub(exc)", src)
        self.assertNotIn("msg = msg[:200]", src,
                         "the unchecked truncation must not come back")

    def test_provider_errors_are_bounded_everywhere_they_are_logged(self):
        self.assertIn("def _reason", read("backtest", "fallback_llm.py"))
        self.assertIn("def _safe_error", read("publication.py"))


# ===========================================================  INTEGRITY


class ForecastIntegrityInvariants(unittest.TestCase):
    def test_five_predictions_per_research_report(self):
        self.assertIn("predictions_per_research_report=5,", read("main.py"))

    def test_one_research_report_per_question(self):
        self.assertIn("research_reports_per_question=1,", read("main.py"))

    def test_publishing_is_enabled(self):
        self.assertIn("publish_to_metaculus = True", read("main.py"))

    def test_the_success_threshold_is_not_overridden(self):
        """required_successful_predictions defaults to 0.5 upstream, so 5
        predictions need at least 3 successes or nothing is published at all.
        Overriding it downward would let a thinner forecast reach Metaculus."""
        self.assertNotIn("required_successful_predictions", read("main.py"))

    def test_the_fallback_layers_return_one_string_per_call(self):
        """Neither wrapper may turn one logical call into two predictions."""
        for module in (("backtest", "fallback_llm.py"), ("backtest", "balanced_llm.py")):
            tree = parse(*module)
            invoke = find_function(tree, "invoke")
            self.assertIsNotNone(invoke, module)
            returns = [n for n in ast.walk(invoke) if isinstance(n, ast.Return)]
            self.assertTrue(returns, module)
            for node in returns:
                self.assertIsInstance(
                    node.value, ast.Name,
                    "{0}.invoke must return the single backend result".format(module),
                )

    def test_neither_wrapper_knows_what_a_forecast_is(self):
        for module in (("backtest", "fallback_llm.py"), ("backtest", "balanced_llm.py"),
                       ("backtest", "rate_limiter.py")):
            src = read(*module)
            for forbidden in ("post_question_comment", "post_binary_question_prediction",
                              "publish_report_to_metaculus", "aggregate_predictions"):
                self.assertNotIn(forbidden, src, "/".join(module))


# ===========================================================  PUBLICATION


class PublicationInvariants(unittest.TestCase):
    def test_the_bot_uses_the_publishing_client(self):
        src = read("main.py")
        self.assertIn("client = PublishingClient()", src)
        self.assertIn("metaculus_client=client", src)

    def test_duplicate_guards_exist_on_both_write_paths(self):
        src = read("publication.py")
        self.assertIn("_commented_posts", src)
        self.assertIn("_predicted_questions", src)

    def test_the_orphan_marker_is_a_stable_token(self):
        from publication import ORPHAN_MARKER

        self.assertEqual(ORPHAN_MARKER, "PUBLICATION_ORPHAN")

    def test_the_comment_gets_more_attempts_than_the_sdk_alone(self):
        from publication import COMMENT_ATTEMPTS

        self.assertGreater(COMMENT_ATTEMPTS, 1)

    def test_comments_are_never_forced_public(self):
        """post_question_comment defaults is_private=True; production must not
        override it. Metaculus publishes bot comments on its own schedule."""
        for parts in PRODUCTION_SOURCES:
            self.assertNotIn("is_private=False", read(*parts), "/".join(parts))

    def test_the_orphan_report_runs_at_the_end_of_every_run(self):
        self.assertIn("print_publication_report(client)", read("main.py"))


# ===========================================================  CONCURRENCY


class ConcurrencyInvariants(unittest.TestCase):
    """R5's intended properties, asserted structurally.

    The behavioural proof is tests/test_concurrency_limiter_loops.py. These
    catch the two mistakes that would reintroduce the bug while still passing
    a behavioural test on a single event loop.
    """

    def _limiter_property(self) -> ast.AST:
        tree = parse("main.py")
        node = find_function(tree, "_concurrency_limiter")
        self.assertIsNotNone(node, "the limiter property is gone")
        return node

    def test_the_property_contains_no_await_or_yield(self):
        """An await inside would let two tasks interleave between the check
        and the assignment, producing two semaphores for one loop."""
        node = self._limiter_property()
        for child in ast.walk(node):
            self.assertNotIsInstance(child, ast.Await)
            self.assertNotIsInstance(child, ast.Yield)
            self.assertNotIsInstance(child, ast.YieldFrom)

    def test_no_semaphore_is_constructed_at_import_time(self):
        """A Semaphore built at class-definition time binds to whichever loop
        first waits on it, which is the original R5 bug."""
        tree = parse("main.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    continue
                # A CALL, not a mention. The declaration carries the quoted
                # annotation "asyncio.Semaphore | None", which is exactly what
                # the lazy version is supposed to look like.
                for child in ast.walk(statement):
                    if not isinstance(child, ast.Call):
                        continue
                    called = getattr(child.func, "attr", None) or getattr(
                        child.func, "id", None
                    )
                    self.assertNotEqual(
                        called, "Semaphore",
                        "class-level Semaphore() rebinds the original R5 bug",
                    )

    def test_the_property_binds_to_the_running_loop(self):
        src = ast.unparse(self._limiter_property())
        self.assertIn("get_running_loop", src)

    def test_it_does_not_replace_a_live_semaphore_outside_a_loop(self):
        src = ast.unparse(self._limiter_property())
        self.assertIn("_concurrency_limiter_instance is not None", src)

    def test_research_concurrency_stays_at_one(self):
        self.assertIn("_max_concurrent_questions = (\n        1", read("main.py"))

    def test_load_order_contains_no_await_or_yield(self):
        """R9: chain selection must be atomic under asyncio. A suspension
        point between reading the loads and using them would let every
        concurrent caller pick the same chain."""
        node = find_function(parse("backtest", "balanced_llm.py"), "_load_order")
        self.assertIsNotNone(node)
        for child in ast.walk(node):
            self.assertNotIsInstance(child, ast.Await)
            self.assertNotIsInstance(child, ast.Yield)


# ===========================================================  DEAD CODE


def unreachable_statements(tree: ast.AST) -> list:
    """Line numbers of statements that follow a return, raise, continue or
    break in the same block, and so can never run."""
    terminal = (ast.Return, ast.Raise, ast.Continue, ast.Break)
    hits = []

    def walk_block(statements):
        for position, statement in enumerate(statements):
            if isinstance(statement, terminal) and position + 1 < len(statements):
                hits.append(statements[position + 1].lineno)
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(statement, field, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                    walk_block(inner)
            for handler in getattr(statement, "handlers", None) or []:
                walk_block(handler.body)
            for case in getattr(statement, "cases", None) or []:
                walk_block(case.body)

    walk_block(getattr(tree, "body", []))
    return sorted(hits)


class NoUnreachableCode(unittest.TestCase):
    """pin_models._parser_primary once ended `return configured` and then went
    on to read GROQ_API_KEY and return a Groq model. Nothing after the first
    return could run, but it read like live behaviour -- as if an environment
    variable could reroute the parser to Groq -- and a reader had no way to
    tell from the source that it could not."""

    SKIP = {"__pycache__", "node_modules", "site-packages"}

    def _sources(self):
        for directory, subdirs, files in os.walk(ROOT):
            subdirs[:] = sorted(
                d for d in subdirs if not d.startswith(".") and d not in self.SKIP
            )
            for name in sorted(files):
                if name.endswith(".py"):
                    yield os.path.join(directory, name)

    def test_no_python_file_has_statements_after_a_return_or_raise(self):
        checked = 0
        for path in self._sources():
            with io.open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            checked += 1
            self.assertEqual(
                unreachable_statements(tree), [],
                "{0}: statements that can never run".format(
                    os.path.relpath(path, ROOT)),
            )
        self.assertGreater(checked, 20, "the scan found almost no source files")

    def test_the_scanner_finds_what_it_is_meant_to_find(self):
        """Guards the guard, including the nested shapes."""
        straight = "def f(x):\n    return x\n    y = 1\n"
        nested = (
            "def f(items):\n"
            "    for item in items:\n"
            "        if item:\n"
            "            continue\n"
            "            print(item)\n"
            "    try:\n"
            "        pass\n"
            "    except ValueError:\n"
            "        raise\n"
            "        cleanup()\n"
        )
        clean = (
            "def f(x):\n"
            "    if x:\n"
            "        return 1\n"
            "    return 2\n"
        )
        self.assertEqual(unreachable_statements(ast.parse(straight)), [3])
        self.assertEqual(unreachable_statements(ast.parse(nested)), [5, 10])
        self.assertEqual(unreachable_statements(ast.parse(clean)), [])


# ===========================================================  DISCOVERY


class DiscoveryInvariants(unittest.TestCase):
    def test_the_bot_does_not_use_the_truncating_upstream_discovery(self):
        """PublishingClient must override the single-page method."""
        src = read("publication.py")
        self.assertIn("def get_all_open_questions_from_tournament", src)
        self.assertIn("fetch_all_open_questions", src)

    def test_truncation_is_reported_and_never_silent(self):
        src = read("discovery.py")
        self.assertIn("discovery_page_limit_reached", src)
        self.assertIn("logger.error", src)

    def test_the_page_ceiling_is_explicit(self):
        from discovery import MAX_PAGES

        self.assertIsInstance(MAX_PAGES, int)
        self.assertGreater(MAX_PAGES, 1)

    def test_deduplication_is_on_the_question_not_the_post(self):
        """Group subquestions share a post id; deduping on it loses forecasts."""
        src = read("discovery.py")
        self.assertIn("seen_question_ids", src)
        self.assertNotIn("seen_post_ids", src)


# ===========================================================  WORKFLOW


class WorkflowInvariants(unittest.TestCase):
    PRODUCTION = (".github", "workflows", "run_bot_on_tournament.yaml")
    CUP = (".github", "workflows", "run_bot_on_metaculus_cup.yaml")
    TEST_BOT = (".github", "workflows", "test_bot.yaml")

    def test_production_runs_the_scored_tournament_mode(self):
        """The poll loop runs the default mode (FutureEval + MiniBench). The
        Cup has its own step after it; see the Cup tests below."""
        src = read(*self.PRODUCTION)
        self.assertIn("python main.py", src)
        self.assertNotIn("--mode test_questions", src)
        self.assertIn("poetry run python main.py 2>&1 | tee", src)

    # ------------------------------------------------- the Cup, in production

    CUP_STEP = "Forecast new Metaculus Cup questions (practice)"

    def _production(self) -> dict:
        import yaml

        return yaml.safe_load(read(*self.PRODUCTION))

    def _production_steps(self) -> dict:
        return {
            step["name"]: (index, step)
            for index, step in enumerate(self._production()["jobs"]["forecast_job"]["steps"])
            if "name" in step
        }

    def test_the_cup_runs_after_the_pin_step_and_the_poll_window(self):
        """After the pin step, or the Cup gets the SDK's default models (one of
        which OpenRouter no longer serves). After the window, or its early
        exit never fires: the Cup almost always has open questions."""
        steps = self._production_steps()
        cup_index, cup = steps[self.CUP_STEP]
        self.assertLess(steps["Pin LLM models"][0], cup_index)
        self.assertLess(steps["Run bot"][0], cup_index)
        self.assertEqual(cup["run"].strip(), "poetry run python main.py --mode metaculus_cup")
        self.assertIn("timeout-minutes", cup)

    def test_the_cup_step_gets_the_same_credentials_as_run_bot(self):
        """A key missing from one step shortens that step's fallback chain
        with no error. The Cup drops only the poll settings and WhatsApp."""
        steps = self._production_steps()
        run_env = set(steps["Run bot"][1]["env"])
        cup_env = set(steps[self.CUP_STEP][1]["env"])
        expected = {
            name for name in run_env
            if not name.startswith(("POLL_", "CALLMEBOT_"))
        }
        self.assertEqual(cup_env, expected)
        self.assertNotIn("GEMINI2_API_KEY", cup_env)

    def test_cup_mode_refreshes_only_through_the_weekly_selection(self):
        """Production runs the Cup every 30 minutes; re-forecasting every open
        question each time would multiply spend on an unpaid tournament. The
        bot's own skip is lifted only because market_pulse.select_stale
        already chose new questions plus week-old forecasts."""
        src = read("main.py")
        start = src.index('elif run_mode == "metaculus_cup":')
        end = src.index('elif run_mode == "market_pulse":')
        cup_branch = "\n".join(
            line for line in src[start:end].splitlines()
            if not line.strip().startswith("#")
        )
        self.assertIn("market_pulse.select_stale(", cup_branch)
        self.assertIn("forecast_questions(", cup_branch)
        self.assertNotIn("forecast_on_tournament(", cup_branch)

    def test_the_manual_cup_workflow_shares_the_production_concurrency_group(self):
        """Two processes deciding "not forecast yet" at the same moment would
        both forecast the same Cup question."""
        import yaml

        production = self._production()
        self.assertEqual(production["concurrency"]["group"], "${{ github.workflow }}")
        cup = yaml.safe_load(read(*self.CUP))
        self.assertEqual(cup["concurrency"]["group"], production["name"])
        self.assertFalse(cup["concurrency"]["cancel-in-progress"])

    def test_production_is_least_privilege(self):
        self.assertIn("permissions:\n  contents: read", read(*self.PRODUCTION))
        # actions: read since 2026-10-23, for the health check's look at this
        # workflow's own runs. Nothing more, and no job widens it.
        production = self._production()
        self.assertEqual(production["permissions"], {"contents": "read", "actions": "read"})
        for name, job in production["jobs"].items():
            self.assertNotIn("permissions", job, name)

    def test_only_production_reads_the_actions_api(self):
        """The health check is the one reader; every other workflow keeps
        contents: read alone."""
        import yaml

        for name in sorted(os.listdir(os.path.join(ROOT, ".github", "workflows"))):
            if not name.endswith((".yaml", ".yml")) or name == self.PRODUCTION[-1]:
                continue
            flow = yaml.safe_load(read(".github", "workflows", name))
            grants = [flow.get("permissions") or {}]
            grants += [job.get("permissions") or {} for job in (flow.get("jobs") or {}).values()]
            for grant in grants:
                if isinstance(grant, dict):
                    self.assertNotIn("actions", grant, name)

    # ------------------------------------------- after the poll window

    MARKET_PULSE_STEP = "Forecast Market Pulse questions (prize)"

    def test_market_pulse_runs_before_the_cup_and_neither_skips_the_other(self):
        """The step that pays goes first (2026-10-23; Cup steps took up to
        13.5 min), and `!cancelled()` on both keeps a failed poll window or
        a failed step of either kind from skipping the other."""
        steps = self._production_steps()
        self.assertLess(steps[self.MARKET_PULSE_STEP][0], steps[self.CUP_STEP][0])
        for name in (self.MARKET_PULSE_STEP, self.CUP_STEP):
            self.assertEqual(steps[name][1].get("if"), "${{ !cancelled() }}", name)

    def test_the_cp_pool_weight_is_a_variable_that_defaults_to_off(self):
        env = self._production()["jobs"]["forecast_job"]["env"]
        self.assertEqual(env["MP_CP_POOL_WEIGHT"], "${{ vars.MP_CP_POOL_WEIGHT || '0' }}")

    HEALTH_STEPS = ("Restore the health-check marker", "Health check",
                    "Save the health-check marker")

    def test_the_health_check_runs_after_the_spend_step_and_cannot_fail_the_job(self):
        steps = self._production_steps()
        indexes = [steps[name][0] for name in self.HEALTH_STEPS]
        self.assertEqual(indexes, sorted(indexes))
        self.assertLess(steps["Spend alerts and run summary"][0], indexes[0])
        for name in self.HEALTH_STEPS:
            step = steps[name][1]
            self.assertTrue(step.get("continue-on-error"), name)
            self.assertIn("!cancelled()", step.get("if", ""), name)

    def test_the_health_check_is_daily_through_a_dated_cache_key(self):
        steps = self._production_steps()
        restore = steps["Restore the health-check marker"][1]
        save = steps["Save the health-check marker"][1]
        self.assertEqual(restore["uses"], "actions/cache/restore@v4")
        self.assertEqual(restore["with"]["restore-keys"].strip(), "health-")
        self.assertEqual(save["uses"], "actions/cache/save@v4")
        self.assertEqual(save["with"]["key"], "${{ steps.health.outputs.daily_key }}")
        self.assertIn("steps.health.outputs.daily_key != ''", save["if"])
        self.assertEqual(restore["with"]["path"], save["with"]["path"])
        self.assertIn('f"daily_key=health-{today}', read("research", "health_check.py"))

    def test_the_health_check_never_runs_the_bot(self):
        """Plain python, not poetry: research/health_check.py is standard
        library only and reads Metaculus through the GET-only client. It gets
        no provider key, so it could not forecast even by mistake."""
        step = self._production_steps()["Health check"][1]
        self.assertEqual(step["run"].strip(), "python research/health_check.py --marker "
                         + self._production_steps()["Restore the health-check marker"][1]["with"]["path"])
        self.assertEqual(set(step["env"]), {"METACULUS_TOKEN", "GITHUB_TOKEN",
                                            "CALLMEBOT_APIKEY", "CALLMEBOT_PHONE_NUMBER"})
        self.assertIn("timeout-minutes", step)

    def test_production_serialises_itself(self):
        src = read(*self.PRODUCTION)
        self.assertIn("concurrency:", src)
        self.assertIn("cancel-in-progress: false", src)

    def test_the_unscored_workflow_targets_only_the_practice_area(self):
        src = read(*self.TEST_BOT)
        self.assertIn("--mode test_questions", src)

    def test_no_unscored_workflow_is_scheduled(self):
        src = yaml_without_comments(*self.TEST_BOT)
        self.assertNotIn("schedule:", src)
        self.assertNotIn("cron:", src)

    def test_exactly_one_workflow_publishes_to_a_scored_tournament(self):
        """The Cup workflow also publishes, so it must stay off the schedule
        of anything CI can start by itself. It is disabled at the GitHub level
        (disabled_fork); this asserts the repository-side half."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        scored = []
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            runs_bot = "python main.py" in src
            unscored = "--mode test_questions" in src
            scheduled = "cron:" in src
            if runs_bot and not unscored and scheduled:
                scored.append(name)
        self.assertEqual(
            scored, ["run_bot_on_tournament.yaml"],
            "exactly one scheduled workflow may publish to a scored tournament",
        )

    def test_the_cup_workflow_is_manual_only(self):
        """It publishes under the same token as production, which already
        forecasts the Cup on its schedule; a second schedule would be a second
        unattended publisher."""
        src = yaml_without_comments(*self.CUP)
        self.assertIn("--mode metaculus_cup", src)
        self.assertNotIn("schedule:", src)
        self.assertNotIn("cron:", src)

    def test_every_workflow_declares_least_privilege(self):
        """Without an explicit grant a job inherits the repository-wide
        default, which is broader than any job here needs."""
        for name in sorted(os.listdir(os.path.join(ROOT, ".github", "workflows"))):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            self.assertIn("permissions:", src, name)
            self.assertIn("contents: read", src, name)

    def test_a_ci_gate_runs_the_suite_on_push_and_pull_request(self):
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertIn("push:", src)
        self.assertIn("pull_request:", src)
        self.assertIn("unittest discover -s tests -t .", src)

    def test_the_ci_gate_covers_both_supported_python_versions(self):
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertIn('"3.11"', src)
        self.assertIn('"3.12"', src)

    def test_the_ci_gate_holds_no_credentials_and_runs_no_bot(self):
        """A test job that could authenticate could also publish.

        Comment-stripped: ci.yaml documents in prose that a future edit adding
        `env: secrets.*` must be caught, and that mention is not a grant."""
        src = yaml_without_comments(".github", "workflows", "ci.yaml")
        self.assertNotIn("secrets.", src)
        self.assertNotIn("python main.py", src)

    def test_production_uses_exactly_one_gemini_credential(self):
        """R11. Four credentials from four Google projects give one
        application 60 RPM where one project allows 15. Google documents that
        limits are per project, and the APIs ToS 2.d forbids attempting to
        circumvent them, but no Google source says whether one workload across
        several projects is circumvention. Unresolved, so production carries
        one credential. The code that could spread calls over several has been
        removed, so this keeps the workflow from suggesting otherwise. See the
        note in the workflow."""
        src = yaml_without_comments(*self.PRODUCTION)
        self.assertIn("GEMINI_API_KEY:", src, "the fallback leg must stay")
        for extra in ("GEMINI2_API_KEY:", "GEMINI3_API_KEY:", "GEMINI4_API_KEY:"):
            self.assertNotIn(
                extra, src,
                "additional Gemini projects must not reach production while "
                "the quota-circumvention question is unresolved",
            )

    def test_extra_gemini_credentials_generate_no_per_credential_wiring(self):
        """The R11 reduction must be real, not cosmetic: a second Gemini
        credential in the environment must not be wired into main.py.

        The code that once did so (bucket_backend, limiter_key, one chain per
        credential) is gone, so this asserts the absence in the output rather
        than in a flag: no bucket_backend call, no limiter_key, and no
        GEMINI2/3/4 anywhere in the generated source, even when those
        variables ARE set.
        """
        import importlib
        import sys

        names = ("GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY",
                 "GEMINI4_API_KEY", "GROQ_API_KEY")
        saved = {key: os.environ.get(key) for key in names}
        for key in saved:
            os.environ.pop(key, None)
        os.environ.update(GEMINI_API_KEY="one", GEMINI2_API_KEY="two",
                          GEMINI3_API_KEY="three", GEMINI4_API_KEY="four",
                          GROQ_API_KEY="g")
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            generated = pin_models.patch(read("main.py"), pin_models.DEFAULTS)
            self.assertNotIn("bucket_backend", generated)
            self.assertNotIn("limiter_key", generated)
            for extra in ("GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY"):
                self.assertNotIn(extra, generated)
            self.assertIn("FallbackLlm([", generated)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    #: The tournament workflow polls inside one run instead of relying on the
    #: schedule event, because the event is dropped. See docs/cadence.md.
    TOURNAMENT = (".github", "workflows", "run_bot_on_tournament.yaml")

    def _poll_settings(self) -> dict:
        text = read(*self.TOURNAMENT)
        out = {}
        for key in ("POLL_WINDOW_SECONDS", "POLL_INTERVAL_SECONDS"):
            match = re.search(key + r':\s*"(\d+)"', text)
            self.assertIsNotNone(match, key + " is gone from the workflow")
            out[key] = int(match.group(1))
        idle = re.search(r'POLL_IDLE_LIMIT:\s*"(\d+)"', text)
        if idle:
            out["POLL_IDLE_LIMIT"] = int(idle.group(1))
        timeout = re.search(r"timeout-minutes:\s*(\d+)", text)
        self.assertIsNotNone(timeout, "the Run bot step lost its timeout")
        out["timeout_seconds"] = int(timeout.group(1)) * 60
        return out

    def test_the_poll_window_fits_inside_the_step_timeout(self):
        """A window longer than the timeout would be cut off mid-poll every
        single run, which looks like a flaky bot rather than a misconfigured
        one."""
        s = self._poll_settings()
        self.assertGreater(
            s["timeout_seconds"], s["POLL_WINDOW_SECONDS"],
            "the timeout must leave room for the window plus the last poll",
        )

    def test_the_step_timeout_stays_under_githubs_job_ceiling(self):
        """GitHub kills a job at 6 hours. A step timeout above that is a
        promise the platform will not keep."""
        self.assertLess(self._poll_settings()["timeout_seconds"], 6 * 60 * 60)

    def test_the_window_cannot_occupy_a_runner_continuously(self):
        """The first version of this loop used a 3h window and, with
        cancel-in-progress: false queueing the scheduled events that arrive
        during a run, occupied a runner ~24h/day and exhausted the account's
        Actions allowance in two days.

        The bound that prevents it: a window must be a small fraction of the
        gap between DELIVERED events, which docs/cadence.md measures in hours.
        One hour is the ceiling; anything near the 6h job limit is how the
        first version looked."""
        window = self._poll_settings()["POLL_WINDOW_SECONDS"]
        self.assertLessEqual(
            window, 3600,
            "a window over an hour makes runs queue back-to-back and occupy "
            "a runner continuously",
        )

    def test_the_window_ends_early_when_nothing_is_open(self):
        """The change that actually cut the cost. Tournament 33022 has had no
        open questions for weeks and MiniBench opens a batch roughly every two
        weeks, so nearly every poll discovers zero questions and there is
        nothing to stay awake for."""
        settings = self._poll_settings()
        self.assertIn("POLL_IDLE_LIMIT", settings)
        self.assertGreaterEqual(settings["POLL_IDLE_LIMIT"], 1)
        text = read(*self.TOURNAMENT)
        self.assertIn("poll_window_early_exit", text)

    def test_the_early_exit_reads_a_marker_discovery_actually_emits(self):
        """The loop decides whether to keep polling by counting
        `questions=<n>` on discovery_complete lines. If discovery.py ever
        renames that marker the loop would stop being able to tell a quiet
        tournament from a broken run -- silently, and in the direction of
        burning minutes again. Pin both halves."""
        self.assertIn("discovery_complete tournament=%s pages=%d questions=%d",
                      read("discovery.py"))
        self.assertIn("discovery_complete", read(*self.TOURNAMENT))

    def test_a_poll_with_no_discovery_line_does_not_end_the_window(self):
        """A run that died before discovery says nothing about whether the
        tournaments are quiet. Treating that as 'idle' would end the window on
        exactly the runs that need another attempt."""
        text = read(*self.TOURNAMENT)
        self.assertIn('if [ "${marker}" -eq 0 ]; then', text)

    def test_the_poll_interval_is_the_schedule_granularity(self):
        """5 minutes is the shortest interval GitHub's scheduler accepts, and
        the whole point of polling is to get that granularity for real."""
        self.assertEqual(self._poll_settings()["POLL_INTERVAL_SECONDS"], 300)

    def test_a_failed_poll_does_not_abort_the_remaining_window(self):
        """One bad question or one provider blip must not blind the bot for
        the rest of the window -- that would be strictly worse than the single
        shot this replaced."""
        text = read(*self.TOURNAMENT)
        self.assertNotIn(
            "set -euo pipefail", text,
            "-e would abort the whole window on the first non-zero poll",
        )
        self.assertIn("set -uo pipefail", text)

    def test_a_window_where_every_poll_failed_still_fails_the_job(self):
        """The other half: swallowing every error would leave a systemic
        break (dead token, dead providers) reported as a green run, and the
        WhatsApp failure notifier keys off the job result."""
        text = read(*self.TOURNAMENT)
        self.assertIn('if [ "${failures}" -eq "${polls}" ]; then', text)
        self.assertIn("exit 1", text)

    def test_the_loop_still_runs_the_real_bot(self):
        text = read(*self.TOURNAMENT)
        self.assertIn("poetry run python main.py", text)

    def test_long_runs_hand_over_rather_than_cancel_each_other(self):
        """With multi-hour runs, cancel-in-progress would kill a run
        mid-forecast whenever a new schedule event landed. Queueing instead
        makes the next run start the moment this one ends, which is the
        continuous coverage the whole change is for."""
        text = yaml_without_comments(*self.TOURNAMENT)
        self.assertIn("cancel-in-progress: false", text)

    def test_the_pin_step_and_the_run_step_get_the_same_provider_keys(self):
        """A fallback chain is configured in TWO places and both must agree.

        pin_models reads os.environ at PATCH time to decide which backends to
        write into the generated llms= block; litellm reads it again at RUN
        time to authenticate them. Give the keys to only one step and the
        failure is silent in both directions:

          pin only -> the block names Gemini and Groq, litellm has no
                      credential for either, and every fallback leg dies. This
                      was live on run_bot_on_metaculus_cup.yaml, which
                      publishes to a SCORED tournament.
          run only -> pin_models sees no keys, emits a single GeneralLlm per
                      role, and the chain silently does not exist at all. This
                      was live on test_bot.yaml, the pre-flight smoke test,
                      so the check that is supposed to catch problems before a
                      real run was testing a configuration production never
                      uses.

        Asserted per workflow that runs main.py, over the provider keys that
        actually gate the chain.
        """
        # DERIVED from the code, not typed here. The hardcoded pair this
        # replaced said ("GEMINI_API_KEY", "GROQ_API_KEY"), which was complete
        # when it was written and silently stopped being so the moment
        # claude-haiku-4.5 joined FALLBACK_CHAIN behind OPENROUTER_API_KEY.
        # Production then generated a chain without Haiku, in both the ensemble
        # and the parser, while every test and every E2E passed -- because
        # test_bot.yaml happened to pass that key and the two scoring workflows
        # did not. A test that must be edited whenever the thing it guards
        # changes is a test that will eventually be wrong.
        import backtest.pin_models as pin

        chain_keys = sorted({
            env for _model, env in pin.FALLBACK_CHAIN + pin.PARSER_EXTRA_CHAIN
        })
        # The AskNews credential gates the researcher's chain at patch time
        # in exactly the same way (pin_models.ACTIVE_ASKNEWS), so it is held
        # to the same rule: present in both steps or in neither.
        chain_keys += sorted(pin.ASKNEWS_ENV_VARS)
        self.assertTrue(chain_keys, "no chain-gating credentials found")
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        checked = 0
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            text = yaml_without_comments(".github", "workflows", name)
            # Must EXECUTE main.py, not merely mention it: ci.yaml names the
            # file in a test command without ever running the bot.
            if "python main.py" not in text:
                continue
            steps = text.split("- name:")
            pin = [b for b in steps if "pin_models.py" in b]
            run = [b for b in steps if "python main.py" in b]
            self.assertTrue(pin, "{0}: no pin_models step".format(name))
            self.assertTrue(run, "{0}: no main.py step".format(name))
            for key in chain_keys:
                for label, blocks in (("pin", pin), ("run", run)):
                    self.assertTrue(
                        any(key in b for b in blocks),
                        "{0}: {1} step is missing {2}; the fallback chain "
                        "would be half-configured".format(name, label, key),
                    )
            checked += 1
        self.assertGreaterEqual(
            checked, 3, "expected tournament, cup and test_bot to be covered"
        )

    def test_asknews_credentials_travel_together(self):
        """AskNewsSearcher takes ASKNEWS_API_KEY or ASKNEWS_CLIENT_ID plus
        ASKNEWS_SECRET. Which one the repository secret will be is not
        decided (none exists yet), so every env block that names one AskNews
        variable must name all three: a block carrying only the OAuth pair
        would silently leave an API-key secret unused -- the researcher would
        run without news while the credential sat in Settings."""
        import backtest.pin_models as pin

        names = set(pin.ASKNEWS_ENV_VARS)
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        checked = 0
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            text = yaml_without_comments(".github", "workflows", name)
            for block in text.split("- name:"):
                present = {var for var in names if var + ":" in block}
                if not present:
                    continue
                self.assertEqual(
                    present, names,
                    "{0}: a step names {1} but not {2}".format(
                        name, sorted(present), sorted(names - present)),
                )
                for var in names:
                    self.assertIn(
                        "{0}: ${{{{ secrets.{0} }}}}".format(var), block,
                        "{0}: {1} must come from the repository secret".format(
                            name, var),
                    )
                checked += 1
        # Pin + Run bot in test_bot and the Cup workflow, Pin + Run bot + Cup
        # in the tournament workflow.
        self.assertGreaterEqual(checked, 7, "expected every pin and run step")

    def test_every_workflow_that_can_publish_to_a_scored_tournament_pins_models(self):
        """Without pin_models, forecasting-tools assigns the researcher role to
        a model OpenRouter no longer serves and every question 404s -- after
        the run has already been dispatched against real questions under the
        real token. The Cup is scored, so it needs the same protection as the
        tournament even though it is manual-only."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            if "python main.py" not in src:
                continue
            if "--mode test_questions" in src:
                continue  # bot-testing-area is unscored
            self.assertIn("pin_models.py", src,
                          name + " can publish to a scored tournament without "
                          "pinning models")

    def test_no_scored_publisher_carries_extra_gemini_credentials(self):
        """R11 applies to every workflow that can publish, not just the
        scheduled one."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = yaml_without_comments(".github", "workflows", name)
            if "python main.py" not in src or "--mode test_questions" in src:
                continue
            for extra in ("GEMINI2_API_KEY:", "GEMINI3_API_KEY:", "GEMINI4_API_KEY:"):
                self.assertNotIn(extra, src, name)

    #: Workflow artifacts are world-readable, because this repository is
    #: public. Metaculus' Terms of Use restrict redistributing its API data
    #: (.gitignore, research/export_safe_report.py), and a dataset of question
    #: titles, URLs, resolutions and community predictions is exactly that. So
    #: what a workflow may upload is an ALLOWLIST, not "everything except the
    #: dataset": adding an upload means adding it here, on purpose, in review.
    #: A denylist fails open the day someone uploads a different file.
    ALLOWED_ARTIFACT_UPLOADS = {
        "research_track_record.yaml": {
            "milestone2_summary.json",
            "milestone2_report.txt",
        },
    }

    @staticmethod
    def _artifact_paths_in(workflow_text: str) -> set:
        """Every `path:` entry of every actions/upload-artifact step in one
        workflow's text (comments already removed). A step with no `path:` at
        all is reported as "<none>", since an upload whose contents cannot be
        read cannot be allowed."""
        lines = workflow_text.splitlines()

        def indent(text: str) -> int:
            return len(text) - len(text.lstrip())

        found = set()
        for start, line in enumerate(lines):
            if "actions/upload-artifact" not in line:
                continue
            # The step's own keys sit at the column of `uses:`, or two columns
            # right of the dash when the step starts with `- uses:`.
            key_indent = indent(line) + (2 if line.lstrip().startswith("- ") else 0)
            paths = set()
            index = start + 1
            while index < len(lines):
                text = lines[index]
                if text.strip() and indent(text) < key_indent:
                    break
                match = re.match(r"\s*path:\s*(.*)$", text)
                if match and indent(text) == key_indent + 2:
                    value = match.group(1).strip()
                    if value in ("|", "|-", ">", ">-"):
                        index += 1
                        while index < len(lines) and (
                            not lines[index].strip()
                            or indent(lines[index]) > indent(text)
                        ):
                            if lines[index].strip():
                                paths.add(lines[index].strip())
                            index += 1
                        continue
                    if value:
                        paths.add(value.strip("\"'"))
                index += 1
            found |= paths or {"<none>"}
        return found

    def _uploaded_artifact_paths(self, name: str) -> set:
        return self._artifact_paths_in(
            yaml_without_comments(".github", "workflows", name))

    def test_no_workflow_uploads_anything_outside_the_allowlist(self):
        """backtest.yaml used to upload dataset.json -- Metaculus question
        titles, URLs, resolutions and community predictions -- as a public
        artifact, against the Terms-of-Use note in .gitignore."""
        workflow_dir = os.path.join(ROOT, ".github", "workflows")
        for name in sorted(os.listdir(workflow_dir)):
            if not name.endswith((".yaml", ".yml")):
                continue
            uploaded = self._uploaded_artifact_paths(name)
            allowed = self.ALLOWED_ARTIFACT_UPLOADS.get(name, set())
            self.assertLessEqual(
                uploaded, allowed,
                "{0} uploads {1}, which is not on the allowlist; workflow "
                "artifacts of a public repository are world-readable".format(
                    name, sorted(uploaded - allowed)),
            )

    def test_the_artifact_allowlist_is_not_vacuous(self):
        """Guards the guard: the one permitted upload must still be found by
        the same parser, or an unreadable step layout would pass silently."""
        for name, allowed in self.ALLOWED_ARTIFACT_UPLOADS.items():
            self.assertEqual(self._uploaded_artifact_paths(name), allowed, name)

    def test_the_artifact_parser_reads_inline_and_block_paths(self):
        """The two YAML spellings of `path:` must both be seen, a `path:` that
        belongs to some other step must not be, and a step with no path must
        not be mistaken for a harmless one."""
        inline = (
            "jobs:\n  a:\n    steps:\n"
            "      - name: Upload\n"
            "        uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: x\n"
            "          path: dataset.json\n"
            "      - name: Next\n"
            "        run: echo\n"
            "        env:\n"
            "          path: not-an-upload\n"
        )
        block = (
            "jobs:\n  a:\n    steps:\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: x\n"
            "          path: |\n"
            "            one.json\n"
            "            two.txt\n"
        )
        nopath = (
            "jobs:\n  a:\n    steps:\n"
            "      - uses: actions/upload-artifact@v4\n"
            "        with:\n"
            "          name: x\n"
        )
        self.assertEqual(self._artifact_paths_in(inline), {"dataset.json"})
        self.assertEqual(self._artifact_paths_in(block), {"one.json", "two.txt"})
        self.assertEqual(self._artifact_paths_in(nopath), {"<none>"})
        self.assertEqual(self._artifact_paths_in("jobs:\n  a:\n    steps: []\n"),
                         set())

    def test_no_workflow_grants_write_permissions_to_the_bot(self):
        for name in sorted(os.listdir(os.path.join(ROOT, ".github", "workflows"))):
            if not name.endswith((".yaml", ".yml")):
                continue
            src = read(".github", "workflows", name)
            self.assertNotIn("contents: write", src, name)
            self.assertNotIn("permissions: write-all", src, name)


# ===========================================================  DEPLOYMENT


class GeneratedCodeInvariants(unittest.TestCase):
    """pin_models.py rewrites main.py inside the runner, before the bot runs.

    That makes it the closest thing this repo has to a supply-chain step: the
    code that executes in production is not the code in git. Everything the
    other tests assert about main.py has to survive the rewrite.
    """

    def _generate(self, **env):
        """Patch a real main.py with a real environment and return the result."""
        import importlib
        import sys

        saved = {k: os.environ.get(k) for k in (
            "GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY",
            "GEMINI4_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY",
            "ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET",
        )}
        for key in saved:
            os.environ.pop(key, None)
        os.environ.update(env)
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            return pin_models.patch(read("main.py"), pin_models.DEFAULTS)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    def test_generated_code_is_valid_python(self):
        ast.parse(self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g"))

    def test_generation_is_deterministic(self):
        env = dict(OPENROUTER_API_KEY="o", GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertEqual(self._generate(**env), self._generate(**env))

    def test_generation_is_idempotent(self):
        import importlib
        import sys

        once = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        saved = {k: os.environ.get(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY")}
        os.environ.update(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        try:
            sys.modules.pop("backtest.pin_models", None)
            pin_models = importlib.import_module("backtest.pin_models")
            self.assertEqual(pin_models.patch(once, pin_models.DEFAULTS), once)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value
            sys.modules.pop("backtest.pin_models", None)

    def test_generated_code_contains_no_secret_value(self):
        generated = self._generate(
            OPENROUTER_API_KEY="secret-openrouter", GEMINI_API_KEY="secret-one",
            GEMINI2_API_KEY="secret-two", GROQ_API_KEY="secret-groq",
        )
        for value in ("secret-openrouter", "secret-one", "secret-two",
                      "secret-groq"):
            self.assertNotIn(value, generated,
                             "a credential VALUE was written into main.py")

    def test_generated_code_keeps_the_redaction_wiring(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("install_forecast_redaction(run_mode)", generated)
        self.assertEqual(generated.count("log_forecast_content("), 9)

    def test_generated_code_keeps_the_publishing_client(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("client = PublishingClient()", generated)
        self.assertIn("metaculus_client=client", generated)
        self.assertIn("print_publication_report(client)", generated)

    def test_generated_code_keeps_the_event_loop_safe_limiter(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("_concurrency_limiter_instance", generated)
        self.assertIn("get_running_loop", generated)

    def test_generated_code_keeps_the_forecast_integrity_settings(self):
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        self.assertIn("predictions_per_research_report=5,", generated)
        self.assertIn("skip_previously_forecasted_questions=True,", generated)

    def test_the_generator_only_touches_the_llms_block(self):
        """Everything outside llms={...} must be byte-identical."""
        original = read("main.py")
        generated = self._generate(GEMINI_API_KEY="a", GROQ_API_KEY="g")

        def strip_llms(src: str) -> str:
            """Remove the bot's llms= block, whether commented or generated.

            Anchored PAST the bot instantiation on purpose: main.py's module
            docstring contains an uncommented `llms={...}` EXAMPLE at the same
            indentation, and an unanchored search strips that instead -- the
            exact trap pin_models._llms_block_span documents.
            """
            init_at = src.find("template_bot = SummerTemplateBot2026(")
            if init_at == -1:
                return src
            commented = src.find("        # llms={", init_at)
            if commented != -1:
                end = src.find("        # },\n", commented)
                return src[:commented] + src[end + len("        # },\n"):]
            start = src.find("        llms={", init_at)
            if start == -1:
                return src
            depth, index = 0, src.index("{", start)
            for position in range(index, len(src)):
                if src[position] == "{":
                    depth += 1
                elif src[position] == "}":
                    depth -= 1
                    if depth == 0:
                        end = position + 1
                        if src[end:end + 2] == ",\n":
                            end += 2
                        return src[:start] + src[end:]
            return src

        # The generator also inserts its own imports, which are the only other
        # legitimate difference.
        cleaned = strip_llms(generated)
        for import_line in ("from backtest.fallback_llm import FallbackLlm\n",
                            "from backtest.balanced_llm import BalancedLlm\n"):
            cleaned = cleaned.replace(import_line, "")
        self.assertEqual(cleaned, strip_llms(original))


if __name__ == "__main__":
    unittest.main()


# ===========================================================  TEST HARNESS


class TestHarnessSafetyInvariants(unittest.TestCase):
    """The suite must be incapable of touching production.

    These guard the harness itself. A module-caching bug in
    _real_forecasting_tools once produced two MetaculusClient classes -- the
    fake transport went on one, publication.PublishingClient was bound to the
    other, and _post_question_prediction fell through to the real
    implementation and began issuing live requests. It showed up only as a
    hang. Both properties below would have named it immediately.
    """

    def test_there_is_exactly_one_metaculus_client_class(self):
        from tests._real_forecasting_tools import (
            real_forecasting_tools,
            the_real_metaculus_client,
        )

        canonical = the_real_metaculus_client()
        with real_forecasting_tools():
            import publication

            self.assertIs(publication.MetaculusClient, canonical,
                          "publication is bound to a different MetaculusClient "
                          "class than the tests patch")
        # And re-entering must not mint a third one.
        self.assertIs(the_real_metaculus_client(), canonical)

    def test_the_network_kill_switch_is_armed(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests):
            requests.post("https://www.metaculus.com/api/questions/forecast/",
                          json=[{"question": 1}], timeout=1)

    def test_the_kill_switch_covers_plain_get_too(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests):
            requests.get("https://example.com", timeout=1)

    def test_the_kill_switch_names_the_target(self):
        import requests

        from tests import NetworkAccessDeniedInTests

        with self.assertRaises(NetworkAccessDeniedInTests) as caught:
            requests.post("https://www.metaculus.com/api/comments/create/", timeout=1)
        self.assertIn("metaculus.com", str(caught.exception))
