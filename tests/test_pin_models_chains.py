"""Chain and ensemble wiring in pin_models: what gets generated, and what must
not change.

pin_models reads os.environ at import time (deliberately: the generated main.py
is meant to be an honest record of what ran), so these tests reload it under a
controlled environment rather than mutating module state in place.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import os
import unittest

# Every credential pin_models reads at import time. Held out of the real
# environment for the duration of a simulated set, so a developer's own keys
# -- including an AskNews one, which changes the researcher's chain -- cannot
# leak into what a test believes it is exercising.
MANAGED_ENV = (
    "GEMINI_API_KEY", "GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY",
    "GROQ_API_KEY", "OPENROUTER_API_KEY",
    "ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET",
    # Not a credential, but read at import time the same way: a developer's
    # own setting must not change the effort a test believes it generates.
    "FORECASTER_REASONING_OPUS",
)


@contextlib.contextmanager
def pinned_env(**env):
    """Reload pin_models AND hold the credential environment for the whole
    block, so call-time os.environ reads see the simulated set too."""
    managed = list(MANAGED_ENV)
    saved = {name: os.environ.pop(name, None) for name in managed}
    os.environ.update(env)
    try:
        import backtest.pin_models as pm

        yield importlib.reload(pm)
    finally:
        for name in managed:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]
        import backtest.pin_models as pm

        importlib.reload(pm)


def load(**env):
    """Reload pin_models with exactly the given credential environment.

    HAZARD, and it has bitten once. The environment is applied only for the
    duration of the IMPORT and restored before this function returns. Anything
    the module resolves at import time -- ACTIVE_FALLBACKS, ACTIVE_PARSER_EXTRA
    -- is captured correctly. Anything that reads os.environ at CALL time sees
    the real environment instead, so a test using `load` to simulate a
    credential set silently exercises a different one.

    That is how the crash of 2026-09-10 reached production: _parser_primary
    read os.environ when called, the selftest test for this combination (then
    `test_one_credential`, now `test_the_production_pin_step_combination`)
    restored the env before calling selftest(), the selftest passed in CI, and
    the identical combination failed on the first scheduled run. Use
    `pinned_env` for anything that must hold the environment across a CALL.
    """
    saved = {}
    managed = list(MANAGED_ENV)
    for name in managed:
        saved[name] = os.environ.pop(name, None)
    os.environ.update(env)
    try:
        import backtest.pin_models as pm

        return importlib.reload(pm)
    finally:
        for name in managed:
            os.environ.pop(name, None)
            if saved[name] is not None:
                os.environ[name] = saved[name]


class GeneratedBlockTests(unittest.TestCase):
    def test_single_key_output_contains_no_per_credential_machinery(self):
        """One Gemini credential produces no PER-CREDENTIAL wiring: no
        bucket_backend, no limiter_key, and no GEMINI2/3/4 env var name.

        BalancedLlm is not evidence of any such wiring: the forecaster
        ensemble uses that class to spread the five forecast calls over the
        distinct primary models, which needs exactly one credential each.
        """
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        self.assertNotIn("bucket_backend", block)
        self.assertNotIn("limiter_key", block)
        for env_var in ("GEMINI2_API_KEY", "GEMINI3_API_KEY", "GEMINI4_API_KEY"):
            self.assertNotIn(env_var, block)
        self.assertIn("FallbackLlm", block)

    def test_a_forecast_grade_fallback_still_ensembles_the_forecaster(self):
        """The ensemble machinery survives FORECAST_GRADE being empty today:
        list a fallback as forecast grade and the default role ensembles
        across it again, while the once-per-question roles stay single."""
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        pm.FORECAST_GRADE = frozenset({GEMINI})
        block = pm.build_block(pm.DEFAULTS)
        default_entry = block[block.index('"default"'):block.index('"summarizer"')]
        self.assertIn("BalancedLlm", default_entry)
        for primary in pm._ensemble_primaries(pm.DEFAULTS["default"]):
            self.assertIn('"{0}"'.format(primary), default_entry)
        # Roles called once per question gain nothing from a lottery over
        # models, so they stay single chains.
        rest = block[block.index('"summarizer"'):]
        self.assertNotIn("BalancedLlm", rest)

    def test_the_parser_keeps_gemini_and_never_gains_gpt_oss(self):
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS)
        parser = block[block.index('"parser"'):]
        self.assertIn("gemini/gemini-3.5-flash-lite", parser)
        self.assertNotIn("gpt-oss-120b", parser,
                         "the parser must not gain a backend that answers in prose")

    def test_no_credential_value_is_ever_written_into_the_block(self):
        """Only variable NAMES may appear; a secret must not reach main.py."""
        pm = load(OPENROUTER_API_KEY="SECRET-O", GEMINI_API_KEY="SECRET-A",
                  GROQ_API_KEY="SECRET-G")
        block = pm.build_block(pm.DEFAULTS)
        for secret in ("SECRET-O", "SECRET-A", "SECRET-G"):
            self.assertNotIn(secret, block)

    def test_generated_code_is_valid_python(self):
        import ast

        pm = load(OPENROUTER_API_KEY="o", GEMINI_API_KEY="a", GROQ_API_KEY="g")
        block = pm.build_block(pm.DEFAULTS).strip().rstrip(",")
        # The block is `llms={...}`; parse the dict literal it assigns.
        self.assertTrue(block.startswith("llms="))
        tree = ast.parse("x = " + block[len("llms="):])
        assigned = tree.body[0].value
        self.assertIsInstance(assigned, ast.Dict,
                              "llms= must be a dict literal, not a set")
        keys = [k.value for k in assigned.keys]
        self.assertEqual(sorted(keys),
                         ["default", "parser", "researcher", "summarizer"])


class ExtraGeminiCredentialsAreInertTests(unittest.TestCase):
    """GEMINI2/3/4_API_KEY used to add one Gemini chain per credential. That
    path is gone (decision R11: production carries one Gemini credential), so
    a stray secret of that name must change nothing, not half-activate."""

    PRODUCTION = dict(OPENROUTER_API_KEY="o", GEMINI_API_KEY="g", GROQ_API_KEY="q")
    EXTRA = dict(GEMINI2_API_KEY="g2", GEMINI3_API_KEY="g3", GEMINI4_API_KEY="g4")

    def test_the_generated_block_is_the_same_with_or_without_them(self):
        for label, env in (
            ("production", self.PRODUCTION),
            ("gemini and groq", dict(GEMINI_API_KEY="g", GROQ_API_KEY="q")),
            ("gemini only", dict(GEMINI_API_KEY="g")),
        ):
            with self.subTest(env=label):
                pm = load(**env)
                without = pm.build_block(pm.DEFAULTS)
                pm = load(**dict(env, **self.EXTRA))
                with_extra = pm.build_block(pm.DEFAULTS)
                self.assertEqual(with_extra, without)
                for name in self.EXTRA:
                    self.assertNotIn(name, with_extra)


HAIKU = "openrouter/anthropic/claude-haiku-4.5"
OPUS = "openrouter/anthropic/claude-opus-5.5"
GPT_4O_MINI = "openrouter/openai/gpt-4o-mini"
GEMINI = "gemini/gemini-3.5-flash-lite"
GROQ_OSS = "groq/openai/gpt-oss-120b"
GROQ_QWEN = "groq/qwen/qwen3.8-27b"
GPT_SOL = "openrouter/openai/gpt-5.6-sol"


def chains_by_role(pm, models):
    """{role: [chain, ...]} where a chain is the list of model names one
    FallbackLlm tries, in order.

    Read out of the GENERATED SOURCE with ast rather than through pin_models'
    own evaluator. The evaluator computes its expectation with the same helpers
    the generator uses, so a mistake shared by both passes its own check; this
    looks at what would actually be written into main.py. A role that is a
    bare model string or a single GeneralLlm is a one-element chain, and a
    BalancedLlm contributes one chain per member.
    """
    block = pm.build_block(models).strip().rstrip(",")
    tree = ast.parse("x = " + block[len("llms="):])

    def model_of(node):
        if isinstance(node, ast.Constant):
            return node.value
        assert isinstance(node, ast.Call), ast.dump(node)
        name = getattr(node.func, "id", None)
        if name == "GeneralLlm":
            return next(k.value.value for k in node.keywords if k.arg == "model")
        raise AssertionError("unexpected backend expression: " + ast.dump(node))

    def chains_of(node):
        name = getattr(getattr(node, "func", None), "id", None)
        if name == "FallbackLlm":
            return [[model_of(b) for b in node.args[0].elts]]
        if name == "BalancedLlm":
            return [c for member in node.args[0].elts for c in chains_of(member)]
        return [[model_of(node)]]

    out = {}
    for key, value in zip(tree.body[0].value.keys, tree.body[0].value.values):
        out[key.value] = chains_of(value)
    return out


class ChainsNameEachBackendOnceTests(unittest.TestCase):
    """A fallback chain tries each backend once.

    The researcher and summarizer use claude-haiku-4.5 as their PRIMARY, and
    FALLBACK_CHAIN also begins with claude-haiku-4.5, so the chain generated
    for those two roles was [haiku, haiku, gemini, groq]: a repeat call to the
    backend that had just failed, which TRIES_IN_CHAIN = 1 exists to prevent.
    """

    PRODUCTION = dict(OPENROUTER_API_KEY="o", GEMINI_API_KEY="g", GROQ_API_KEY="q")

    def test_no_generated_chain_repeats_a_backend(self):
        for label, env in (
            ("production", self.PRODUCTION),
            ("gemini and groq only", dict(GEMINI_API_KEY="g", GROQ_API_KEY="q")),
            ("openrouter only", dict(OPENROUTER_API_KEY="o")),
            ("groq only", dict(GROQ_API_KEY="q")),
        ):
            pm = load(**env)
            for role, chains in chains_by_role(pm, pm.DEFAULTS).items():
                for chain in chains:
                    with self.subTest(env=label, role=role):
                        self.assertEqual(
                            len(chain), len(set(chain)),
                            "{0} chain names a backend twice: {1}".format(role, chain),
                        )

    def test_the_production_chains_are_exactly_these_and_in_this_order(self):
        """Pins the shape production runs, so de-duplicating cannot reorder or
        drop a link: the primary leads, the rest keep their configured order."""
        pm = load(**self.PRODUCTION)
        chains = chains_by_role(pm, pm.DEFAULTS)
        self.assertEqual(chains["researcher"], [[HAIKU, GEMINI, GROQ_OSS]])
        self.assertEqual(chains["summarizer"], [[HAIKU, GEMINI, GROQ_OSS]])
        # Claude parses since 2026-10-22; gpt-4o-mini only catches an outage.
        self.assertEqual(
            chains["parser"], [[HAIKU, GEMINI, GPT_4O_MINI, GROQ_QWEN]])
        # The forecaster: Opus and GPT-5.6 Sol share the five calls (one chain
        # each, see FORECAST_VOTERS); the cheap models only catch a failure.
        # They used to be ensemble voters, which put three or four of every
        # five forecasts on them (see FORECAST_GRADE).
        self.assertEqual(chains["default"], [
            [OPUS, GPT_SOL, HAIKU, GEMINI, GROQ_OSS],
            [GPT_SOL, OPUS, HAIKU, GEMINI, GROQ_OSS],
        ])

    def test_cheap_fallbacks_never_vote_in_the_forecaster_ensemble(self):
        pm = load(**self.PRODUCTION)
        self.assertEqual(pm._ensemble_primaries(pm.DEFAULTS["default"]), [OPUS, GPT_SOL])
        for cheap in (HAIKU, GEMINI, GROQ_OSS):
            self.assertNotIn(cheap, pm.FORECAST_GRADE)
            self.assertNotIn(cheap, [m for m, _ in pm.FORECAST_VOTERS])

    def test_voters_were_smoke_tested_and_cost_no_more_than_opus(self):
        """Provenance, like test_fallback_llm's verified set: a voter answers
        every question, so it must have passed a real completion first."""
        import backtest.pin_models as pin
        verified = {GPT_SOL}  # smoke-test run 39073967369: http/litellm/structured OK
        for model, _env in pin.FORECAST_VOTERS:
            self.assertIn(model, verified, model + " was never smoke-tested")

    def test_voters_reach_only_the_forecaster(self):
        pm = load(**self.PRODUCTION)
        chains = chains_by_role(pm, pm.DEFAULTS)
        for role in ("researcher", "summarizer", "parser"):
            for chain in chains[role]:
                self.assertNotIn(GPT_SOL, chain, role)

    def test_a_forecast_grade_fallback_gets_its_own_chain(self):
        """How a second frontier voter would be wired: one chain per primary,
        that primary first and the others behind it in primary order."""
        pm = load(**self.PRODUCTION)
        pm.FORECAST_GRADE = frozenset({GEMINI})
        chains = chains_by_role(pm, pm.DEFAULTS)
        self.assertEqual(chains["default"], [
            [OPUS, GEMINI, GPT_SOL, HAIKU, GROQ_OSS],
            [GEMINI, OPUS, GPT_SOL, HAIKU, GROQ_OSS],
            [GPT_SOL, OPUS, GEMINI, HAIKU, GROQ_OSS],
        ])

    def test_an_override_naming_a_fallback_model_does_not_repeat_it(self):
        """models.txt can set any role to any model, including one that is also
        a fallback. The first occurrence wins and the order is otherwise kept."""
        pm = load(**self.PRODUCTION)
        chains = chains_by_role(pm, dict(pm.DEFAULTS, researcher=GEMINI))
        self.assertEqual(chains["researcher"], [[GEMINI, HAIKU, GROQ_OSS]])
        # The parser excluded DEFAULTS["parser"] rather than the configured
        # primary, so overriding it with a model that is also a fallback
        # repeated that model.
        chains = chains_by_role(pm, dict(pm.DEFAULTS, parser=GPT_4O_MINI))
        self.assertEqual(chains["parser"], [[GPT_4O_MINI, GEMINI, GROQ_QWEN]])


class ImportWiringTests(unittest.TestCase):
    def _main_src(self):
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return io.open(os.path.join(root, "main.py"), encoding="utf-8").read()

    def test_balanced_import_is_added_exactly_once_and_is_idempotent(self):
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        pm.FORECAST_GRADE = frozenset({GEMINI})
        once = pm.patch(self._main_src(), pm.DEFAULTS)
        self.assertIn("from backtest.balanced_llm import", once)
        self.assertEqual(once.count("from backtest.balanced_llm import"), 1)
        self.assertEqual(pm.patch(once, pm.DEFAULTS), once, "patch must be idempotent")

    def test_the_balanced_import_tracks_whether_the_block_uses_it(self):
        """The import must be present exactly when the generated block names
        BalancedLlm, and absent otherwise -- disagreeing either way is the one
        thing this generator can do that makes main.py raise at startup.

        Both directions are checked against the block itself rather than
        against a re-derived condition, so this stays true whatever the reason
        the generator has for emitting BalancedLlm.
        """
        for env in (
            {"GEMINI_API_KEY": "a", "GROQ_API_KEY": "g"},
            {"OPENROUTER_API_KEY": "o", "GEMINI_API_KEY": "a", "GROQ_API_KEY": "g"},
            {},  # no fallback keys at all: single GeneralLlm per role
        ):
            with self.subTest(env=sorted(env)):
                pm = load(**env)
                once = pm.patch(self._main_src(), pm.DEFAULTS)
                uses = "BalancedLlm(" in once[once.index("llms={"):]
                imported = "from backtest.balanced_llm import" in once
                self.assertEqual(
                    uses, imported,
                    "block uses BalancedLlm={0} but import present={1}".format(
                        uses, imported),
                )

    def test_dropping_every_fallback_key_removes_the_balanced_import_again(self):
        """Converge-either-way: a source patched WITH the import must lose it
        when the keys that justified it are gone."""
        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        pm.FORECAST_GRADE = frozenset({GEMINI})
        balanced = pm.patch(self._main_src(), pm.DEFAULTS)
        self.assertIn("from backtest.balanced_llm import", balanced)
        pm = load()  # no fallback keys: nothing to balance or fall back to
        reverted = pm.patch(balanced, pm.DEFAULTS)
        self.assertNotIn("from backtest.balanced_llm import", reverted)


    def test_every_name_the_generated_imports_ask_for_exists(self):
        """main.py imports what the generator tells it to, at startup, before
        a single question is read. A name that has been removed from the module
        but is still imported turns the whole run into an ImportError, and
        nothing else in the suite executes that line. This reads the modules'
        own definitions with ast, so it needs neither forecasting_tools nor the
        modules' imports to succeed."""
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def defined_in(module):
            path = os.path.join(root, *module.split(".")) + ".py"
            with io.open(path, encoding="utf-8") as handle:
                tree = ast.parse(handle.read())
            names = set()
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names.update(t.id for t in node.targets
                                 if isinstance(t, ast.Name))
            return names

        pm = load(OPENROUTER_API_KEY="o", GEMINI_API_KEY="a", GROQ_API_KEY="g")
        generated = pm.patch(self._main_src(), pm.DEFAULTS)
        imports = [node for node in ast.parse(generated).body
                   if isinstance(node, ast.ImportFrom)
                   and (node.module or "").startswith("backtest.")]
        self.assertTrue(imports, "the generator added no backtest import at all")
        for node in imports:
            available = defined_in(node.module)
            for alias in node.names:
                self.assertIn(alias.name, available,
                              "{0} imports {1}, which {0} does not define".format(
                                  node.module, alias.name))


class SelftestAcrossCredentialSetsTests(unittest.TestCase):
    """pin_models runs its own selftest on every invocation, before touching
    main.py. What it generates depends on which provider credentials are
    present, and an assertion written for one combination can fail only when
    another appears -- which is how a selftest failure once escaped local
    checks and broke CI (run 32385415823) -- so every combination the
    workflows can produce is exercised here.

    selftest() operates on synthetic sources; it never writes main.py.
    """

    def _run_selftest(self, **env):
        # pinned_env, not load: selftest() is a CALL, and the environment has
        # to still be in place while it runs. With `load` the env was restored
        # first, which is exactly how a real crash passed CI here.
        with pinned_env(**env) as pm:
            pm.selftest()  # raises AssertionError on failure

    def test_no_credentials(self):
        self._run_selftest()

    def test_the_production_pin_step_combination(self):
        """The exact credential set the scoring workflows give the pin step.

        Named explicitly because it is the one that broke: on 2026-09-10 three
        consecutive scheduled runs died here (34417432835, 34425442320,
        34445843668) in a combination the matrix already covered on paper --
        but `load` restored the environment before selftest() ran, so the
        simulated set was never the one under test. With pinned_env it is."""
        self._run_selftest(GEMINI_API_KEY="a", GROQ_API_KEY="g")

    def test_the_production_run_step_combination(self):
        """Every provider key present, as the Run bot step sees them."""
        self._run_selftest(GEMINI_API_KEY="a", GROQ_API_KEY="g",
                           OPENROUTER_API_KEY="o")

    def test_openrouter_only(self):
        self._run_selftest(OPENROUTER_API_KEY="o")

    def test_leftover_extra_gemini_secrets_are_ignored(self):
        """The repository may still hold GEMINI2/3/4_API_KEY as secrets. Even
        if a workflow passed them, they must not change what the selftest
        expects."""
        self._run_selftest(OPENROUTER_API_KEY="o", GEMINI_API_KEY="a",
                           GEMINI2_API_KEY="b", GEMINI3_API_KEY="c",
                           GEMINI4_API_KEY="d", GROQ_API_KEY="g")

    def test_groq_only(self):
        self._run_selftest(GROQ_API_KEY="g")


class DriftTests(unittest.TestCase):
    """pin_models duplicates the set of rate-limited models because it runs as
    a bare script and cannot import backtest.*. That duplication must never
    drift."""

    def test_rate_limited_models_matches_the_real_registry(self):
        """pin_models duplicates the set of rate-limiter-registered model
        strings because it cannot import backtest.* (see the test below). This
        pins the copy to the original.

        It is a safety property, not tidiness: DEFAULT_LIMITS.get(model,
        ProviderLimits()) hands back an UNTHROTTLED limiter for an unknown key,
        so an ensemble bucket key that drifted out of the registry would look
        controlled and enforce nothing -- silently removing Groq's measured
        8000 TPM cap.
        """
        from backtest import rate_limiter

        pm = load()
        self.assertEqual(
            set(pm.RATE_LIMITED_MODELS),
            set(rate_limiter.DEFAULT_LIMITS),
            "pin_models.RATE_LIMITED_MODELS has drifted from "
            "rate_limiter.DEFAULT_LIMITS",
        )

    def test_every_ensemble_bucket_key_is_rate_limited(self):
        """The keys the generator actually emits must all be registered, for
        the same reason. Checked against the real registry, not the copy."""
        from backtest import rate_limiter

        pm = load(GEMINI_API_KEY="a", GROQ_API_KEY="g")
        for primary in pm._ensemble_primaries(pm.DEFAULTS["default"]):
            self.assertIn(primary, rate_limiter.DEFAULT_LIMITS, primary)

    def test_pin_models_imports_nothing_from_backtest(self):
        """It runs as `python backtest/pin_models.py`, with the repo root off
        sys.path. A backtest.* import raises before the bot ever starts."""
        import io
        import os.path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = io.open(os.path.join(root, "backtest", "pin_models.py"),
                      encoding="utf-8").read()
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import backtest", "from backtest")):
                self.fail("pin_models must not import backtest.*: " + stripped)


if __name__ == "__main__":
    unittest.main()


class ForecasterReasoningTests(unittest.TestCase):
    """The forecast call asks Opus for high reasoning effort, and only it."""

    PRODUCTION = dict(OPENROUTER_API_KEY="o", GEMINI_API_KEY="g", GROQ_API_KEY="q")

    def _role_entry(self, block, role, next_role):
        return block[block.index('"{0}"'.format(role)):block.index('"{0}"'.format(next_role))]

    def test_opus_forecasts_with_high_effort_and_a_longer_timeout(self):
        pm = load(**self.PRODUCTION)
        default = self._role_entry(pm.build_block(pm.DEFAULTS), "default", "summarizer")
        opus_line = next(l for l in default.splitlines() if OPUS in l)
        self.assertIn('extra_body={"reasoning": {"effort": "high"}}', opus_line)
        self.assertIn("timeout={0}".format(pm.FORECASTER_TIMEOUT_SECONDS), opus_line)

    def test_the_fallbacks_and_other_roles_are_unchanged(self):
        pm = load(**self.PRODUCTION)
        block = pm.build_block(pm.DEFAULTS)
        default = self._role_entry(block, "default", "summarizer")
        for line in default.splitlines():
            if "GeneralLlm(" in line and OPUS not in line and GPT_SOL not in line:
                self.assertNotIn("extra_body", line)
        gpt_line = next(l for l in default.splitlines() if GPT_SOL in l)
        self.assertIn('"effort": "high"', gpt_line)
        rest = block[block.index('"summarizer"'):]
        self.assertNotIn("extra_body", rest)

    def test_without_fallback_keys_opus_still_gets_it(self):
        pm = load()
        default = self._role_entry(pm.build_block(pm.DEFAULTS), "default", "summarizer")
        self.assertIn('"effort": "high"', default)

    def test_never_litellms_reasoning_effort(self):
        """litellm 1.80.10 raises UnsupportedParamsError for reasoning_effort on
        openrouter/ models, which would fail every Opus call over to Haiku."""
        pm = load(**self.PRODUCTION)
        self.assertNotIn("reasoning_effort", pm.build_block(pm.DEFAULTS))

    def test_the_generated_block_still_parses(self):
        pm = load(**self.PRODUCTION)
        block = pm.build_block(pm.DEFAULTS).strip().rstrip(",")
        ast.parse("x = " + block[len("llms="):])


class OpusEffortOverrideTests(unittest.TestCase):
    """FORECASTER_REASONING_OPUS (2026-10-23): one Test Bot run may ask Opus
    for another effort, to measure whether it reasons more. Production never
    sets it, so the default must not move."""

    PRODUCTION = ForecasterReasoningTests.PRODUCTION

    def tearDown(self):
        # load() leaves the module as the last reload made it; an xhigh left
        # behind would leak into any later test that builds a block.
        load()

    def _lines(self, pm):
        block = pm.build_block(pm.DEFAULTS)
        default = block[block.index('"default"'):block.index('"summarizer"')]
        opus = next(l for l in default.splitlines() if OPUS in l)
        sol = next(l for l in default.splitlines() if GPT_SOL in l)
        return opus, sol

    def test_unset_keeps_high(self):
        opus, _sol = self._lines(load(**self.PRODUCTION))
        self.assertIn('"effort": "high"', opus)

    def test_xhigh_reaches_opus_only(self):
        pm = load(FORECASTER_REASONING_OPUS="xhigh", **self.PRODUCTION)
        opus, sol = self._lines(pm)
        self.assertIn('extra_body={"reasoning": {"effort": "xhigh"}}', opus)
        self.assertIn('"effort": "high"', sol)
        self.assertNotIn("max_tokens", pm.build_block(pm.DEFAULTS))

    def test_anything_but_an_effort_name_is_ignored(self):
        """A token budget returns 400 on Opus 5.5 and would fail every Opus
        call over to Sol or Haiku in silence."""
        import io

        for value in ("4000", "max", "high; drop", "XHIGH "):
            warned = io.StringIO()
            with contextlib.redirect_stderr(warned):
                opus, _sol = self._lines(load(FORECASTER_REASONING_OPUS=value, **self.PRODUCTION))
            expected = "xhigh" if value.strip().lower() == "xhigh" else "high"
            self.assertIn('"effort": "{0}"'.format(expected), opus, value)
            self.assertEqual("::warning::" in warned.getvalue(), expected == "high", value)

    def test_the_override_names_the_production_forecaster(self):
        pm = load(**self.PRODUCTION)
        self.assertEqual(pm.OPUS_FORECASTER, pm.DEFAULTS["default"])
        self.assertIn(pm.OPUS_FORECASTER, pm.FORECASTER_REASONING)

    def test_only_test_bot_sets_it(self):
        import yaml

        workflows = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 ".github", "workflows")
        for name in sorted(os.listdir(workflows)):
            if not name.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(workflows, name), encoding="utf-8") as handle:
                text = handle.read()
            if name != "test_bot.yaml":
                self.assertNotIn("FORECASTER_REASONING_OPUS", text, name)
                continue
            flow = yaml.safe_load(text)
            # PyYAML reads the bare key `on:` as True.
            choice = (flow.get("on") or flow.get(True))["workflow_dispatch"]["inputs"]["opus_reasoning"]
            self.assertEqual((choice["default"], choice["options"]), ("high", ["high", "xhigh"]))
            steps = flow["jobs"]["forecast_job"]["steps"]
            setters = [s.get("name") for s in steps if "FORECASTER_REASONING_OPUS" in (s.get("env") or {})]
            self.assertEqual(setters, ["Pin and verify LLM models"])
