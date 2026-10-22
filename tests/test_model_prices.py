"""litellm must price the forecaster's model, whatever cost map it loaded.

litellm downloads its cost map at import and falls back to the copy bundled
in the wheel when that fails. The locked wheel (1.80.10) predates
claude-opus-5.5, so without help a run whose download failed prices every
forecaster call at $0 -- silently, because forecasting_tools' ModelTracker
only warns and litellm swallows the cost error in its logging callback.
bot_helpers.register_model_prices fills that gap. These tests prove it does,
and that it never overrides a price litellm already has.
"""

from __future__ import annotations

import unittest

import litellm

import bot_helpers
from backtest import pin_models

OPUS = "openrouter/anthropic/claude-opus-5.5"


class RegisterModelPricesTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            model: dict(litellm.model_cost[model])
            for model in bot_helpers.KNOWN_MODEL_PRICES_PER_MTOK
            if model in litellm.model_cost
        }

    def tearDown(self):
        for model in bot_helpers.KNOWN_MODEL_PRICES_PER_MTOK:
            litellm.model_cost.pop(model, None)
        litellm.model_cost.update(self._saved)

    def test_the_table_covers_the_forecaster(self):
        """The one model this exists for. A future forecaster that litellm's
        bundled map does not know needs its own row, with a verified price."""
        self.assertIn(pin_models.DEFAULTS["default"], bot_helpers.KNOWN_MODEL_PRICES_PER_MTOK)

    def test_a_missing_model_is_registered_at_the_verified_price(self):
        litellm.model_cost.pop(OPUS, None)
        registered = bot_helpers.register_model_prices()
        self.assertIn(OPUS, registered)
        entry = litellm.model_cost[OPUS]
        # $4 / $20 per million tokens, read from OpenRouter on 2026-10-02.
        self.assertAlmostEqual(entry["input_cost_per_token"], 4e-6)
        self.assertAlmostEqual(entry["output_cost_per_token"], 20e-6)
        self.assertEqual(entry["litellm_provider"], "openrouter")

    def test_a_known_model_is_left_alone(self):
        """A newer downloaded map wins over this table; a price change
        upstream must never be masked by it."""
        litellm.model_cost[OPUS] = {
            "input_cost_per_token": 1e-9, "output_cost_per_token": 2e-9,
            "litellm_provider": "openrouter", "mode": "chat",
        }
        registered = bot_helpers.register_model_prices()
        self.assertNotIn(OPUS, registered)
        self.assertEqual(litellm.model_cost[OPUS]["input_cost_per_token"], 1e-9)

    def test_registration_is_idempotent(self):
        litellm.model_cost.pop(OPUS, None)
        self.assertIn(OPUS, bot_helpers.register_model_prices())
        self.assertEqual(bot_helpers.register_model_prices(), [])

    def test_an_entry_without_a_price_is_a_gap(self):
        sol = "openrouter/openai/gpt-5.6-sol"
        litellm.model_cost[sol] = {"litellm_provider": "openrouter", "mode": "chat"}
        self.assertIn(sol, bot_helpers.register_model_prices())
        cost = litellm.cost_per_token(model=sol, prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(sum(cost), 24.0, places=6)

    def test_every_model_the_pinned_chains_call_is_priced(self):
        """The spend ledger found three of them at $0 (Test Bot run
        39578224912)."""
        for model in ("openrouter/openai/gpt-5.6-sol", "openrouter/anthropic/claude-haiku-4.5",
                      "openrouter/openai/gpt-4o-mini"):
            self.assertIn(model, bot_helpers.KNOWN_MODEL_PRICES_PER_MTOK)

    def test_the_registered_price_is_what_litellm_then_charges(self):
        """Through litellm's own calculator, not by reading the dict back:
        the point is that cost tracking works, not that a key exists."""
        litellm.model_cost.pop(OPUS, None)
        bot_helpers.register_model_prices()
        cost = litellm.cost_per_token(
            model=OPUS, prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        self.assertAlmostEqual(sum(cost), 24.0, places=6)

    def test_main_registers_before_building_any_llm(self):
        """ModelTracker decides whether to warn inside GeneralLlm.__init__,
        and the bot is built at the bottom of main.py, so the call must sit
        at module level above the bot class."""
        import io
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = io.open(os.path.join(root, "main.py"), encoding="utf-8").read()
        self.assertLess(src.index("register_model_prices()"),
                        src.index("class SummerTemplateBot2026"))


if __name__ == "__main__":
    unittest.main()
