"""The forecasting prompts and the binary parse, run through main.py itself.

Four changes of 2026-10-23, each checked on the REAL SummerTemplateBot2026
with the model and the network replaced:

* Market Pulse EPS questions that resolve on GAAP get the GAAP note and a
  width line of their own; every other numeric prompt, FutureEval and
  MiniBench above all, is byte for byte what it was.
* A binary forecast is read from its own "Probability: NN%" line, and only
  goes to the LLM parser when that line is missing or unclear.
* The crux pass pools its five forecasts with the first five.
* The binary prompt and the timing block no longer say an outcome cannot
  have happened yet.

Every question here is synthetic: no real company, value or prior.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import unittest
from unittest import mock

from tests._real_forecasting_tools import real_forecasting_tools

with real_forecasting_tools():
    from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
    from forecasting_tools.data_models.questions import (
        BinaryQuestion,
        DiscreteQuestion,
        NumericQuestion,
    )
    from forecasting_tools.data_models.forecast_report import ReasonedPrediction

    import claude_research
    from publication import PublishingClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKET_PULSE = "market-pulse-26q4"
FUTUREEVAL = "fall-futureeval-2026"
MINIBENCH = "minibench"

GAAP_EPS_CRITERIA = (
    "This question resolves as the diluted earnings per share for the quarter, "
    "as first reported by the company in conformity with US GAAP."
)
REVENUE_CRITERIA = (
    "This question resolves as total revenue for the quarter, as first reported "
    "by the company in conformity with US GAAP."
)
ADJUSTED_EPS_CRITERIA = (
    "This question resolves as the adjusted (non-GAAP) earnings per share for the "
    "quarter, as first reported by the company."
)

#: The width line the numeric prompt has always carried.
WIDE = ("You remind yourself that good forecasters are humble and set wide 90/10 "
        "confidence intervals to account for unknown unknowns.")


def load_main():
    """main.py without its __main__ block (see test_pipeline_hostile_model)."""
    with real_forecasting_tools():
        spec = importlib.util.spec_from_file_location(
            "prompt_wiring_main", os.path.join(ROOT, "main.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


BOTMOD = load_main()


class FakeLlm:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    async def invoke(self, prompt, system_prompt=None):  # noqa: ARG002
        self.prompts.append(prompt)
        return self.text


class Bot(BOTMOD.SummerTemplateBot2026):
    """Captures each prompt instead of calling a model."""

    llm = None

    def get_llm(self, purpose="default", guarantee_type=None):  # noqa: ARG002
        return self.llm

    async def _numeric_prompt_to_forecast(self, question, prompt):
        return prompt

    async def _date_prompt_to_forecast(self, question, prompt):
        return prompt


def make_bot():
    with real_forecasting_tools():
        return Bot(
            metaculus_client=PublishingClient(sleep=lambda _s: None),
            research_reports_per_question=1,
            predictions_per_research_report=5,
            publish_reports_to_metaculus=False,
            folder_to_save_reports_to=None,
        )


def run(coroutine):
    with real_forecasting_tools():
        return asyncio.run(coroutine)


def numeric(slugs, criteria=GAAP_EPS_CRITERIA, cls=DiscreteQuestion):
    return cls(
        question_text="What will Q4 EPS be for these companies?",
        group_question_option="Synthetic Co",
        id_of_post=7001,
        id_of_question=7002,
        page_url="https://www.metaculus.com/questions/7001",
        background_info="background",
        resolution_criteria=criteria,
        fine_print="fine print",
        unit_of_measure="USD",
        lower_bound=1.0,
        upper_bound=3.0,
        open_lower_bound=True,
        open_upper_bound=True,
        tournament_slugs=list(slugs),
    )


def binary():
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_post=7101,
        id_of_question=7102,
        page_url="https://www.metaculus.com/questions/7101",
        background_info="background",
        resolution_criteria="criteria",
        fine_print="fine print",
    )


class NumericPromptTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot()

    def prompt(self, question):
        return run(self.bot._run_forecast_on_numeric(question, "research"))

    def test_market_pulse_gaap_eps_gets_the_note_and_its_own_width_line(self):
        with self.assertLogs(BOTMOD.logger, level="INFO") as logs:
            text = self.prompt(numeric([MARKET_PULSE]))
        self.assertIn(BOTMOD.SummerTemplateBot2026._GAAP_EPS_NOTE, text)
        self.assertIn(BOTMOD.SummerTemplateBot2026._GAAP_EPS_WIDTH, text)
        self.assertNotIn(WIDE, text)
        self.assertIn("numeric_prompt gaap_eps=1", "\n".join(logs.output))

    def test_the_note_comes_before_the_width_and_the_answer_format(self):
        text = self.prompt(numeric([MARKET_PULSE]))
        note = text.index(BOTMOD.SummerTemplateBot2026._GAAP_EPS_NOTE)
        self.assertLess(note, text.index(BOTMOD.SummerTemplateBot2026._GAAP_EPS_WIDTH))
        self.assertLess(note, text.index("The last thing you write"))

    def test_futureeval_and_minibench_never_get_it(self):
        for slugs in ([FUTUREEVAL], [MINIBENCH], [], ["bot-testing-area"]):
            for cls in (NumericQuestion, DiscreteQuestion):
                text = self.prompt(numeric(slugs, cls=cls))
                self.assertNotIn("GAAP surprises", text, slugs)
                self.assertNotIn("This resolves on GAAP", text, slugs)
                self.assertIn(WIDE, text, slugs)

    def test_other_prompts_are_unchanged_byte_for_byte(self):
        """A FutureEval prompt is what the old template produced: the old
        width line straight after the conditional disclaimer, no blank line
        where the note would go."""
        text = self.prompt(numeric([FUTUREEVAL]))
        self.assertIn(
            "and is that tail wide enough?\n\n\n" + WIDE + "\n\nThe last thing you write", text
        )

    def test_market_pulse_questions_that_are_not_gaap_eps_keep_the_old_prompt(self):
        for criteria in (REVENUE_CRITERIA, ADJUSTED_EPS_CRITERIA, "", None):
            text = self.prompt(numeric([MARKET_PULSE], criteria=criteria))
            self.assertNotIn("GAAP surprises", text, criteria)
            self.assertIn(WIDE, text, criteria)

    def test_a_broken_question_degrades_to_the_old_prompt(self):
        question = numeric([MARKET_PULSE])
        with mock.patch.object(BOTMOD.tournaments, "MARKET_PULSE", None):
            with self.assertLogs(BOTMOD.logger, level="WARNING") as logs:
                text = self.prompt(question)
        self.assertIn(WIDE, text)
        self.assertIn("gaap_eps_check_failed", "\n".join(logs.output))

    def test_the_date_prompt_is_not_touched(self):
        with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as handle:
            src = handle.read()
        start = src.index("async def _run_forecast_on_date")
        body = src[start:src.index("async def _date_prompt_to_forecast", start)]
        self.assertIn(WIDE, body)
        self.assertNotIn("gaap", body.lower())


class BinaryParseTests(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot()
        self.parser_calls = 0

        async def fake_structure_output(text, output_type, model=None, **kwargs):  # noqa: ARG001
            self.parser_calls += 1
            from forecasting_tools import BinaryPrediction

            return BinaryPrediction(prediction_in_decimal=0.4)

        patcher = mock.patch.object(BOTMOD, "structure_output", fake_structure_output)
        patcher.start()
        self.addCleanup(patcher.stop)

    def forecast(self, reasoning):
        self.bot.llm = FakeLlm(reasoning)
        with self.assertLogs(BOTMOD.logger, level="INFO") as logs:
            result = run(self.bot._binary_prompt_to_forecast(binary(), "prompt"))
        return result.prediction_value, "\n".join(logs.output)

    def test_a_plain_last_line_skips_the_parser(self):
        value, log = self.forecast("(a) 22 days.\nProbability: 15%")
        self.assertAlmostEqual(value, 0.15)
        self.assertEqual(self.parser_calls, 0)
        self.assertIn("binary_parse source=verbatim", log)

    def test_an_unclear_line_falls_back_to_the_parser(self):
        value, log = self.forecast("(a) 22 days.\nProbability: 15-20%")
        self.assertAlmostEqual(value, 0.4)
        self.assertEqual(self.parser_calls, 1)
        self.assertIn("binary_parse source=llm", log)

    def test_the_cap_still_applies_to_a_verbatim_read(self):
        value, _log = self.forecast("Probability: 100%")
        self.assertAlmostEqual(value, 0.97)

    def test_the_log_carries_the_source_never_the_value(self):
        _value, log = self.forecast("Probability: 15%")
        line = next(x for x in log.splitlines() if "binary_parse" in x)
        self.assertNotIn("15", line.replace("7101", ""))

    def test_a_reader_error_degrades_to_the_parser(self):
        with mock.patch.object(BOTMOD.forecast_guards, "read_probability",
                               side_effect=RuntimeError("boom")):
            value, log = self.forecast("Probability: 15%")
        self.assertAlmostEqual(value, 0.4)
        self.assertIn("binary_parse_verbatim_failed reason=RuntimeError", log)


class CruxPoolTests(unittest.TestCase):
    def test_the_crux_pass_keeps_all_ten_forecasts(self):
        bot = make_bot()
        first = [ReasonedPrediction(prediction_value=v, reasoning=f"first {v}")
                 for v in (0.1, 0.2, 0.3, 0.4, 0.6)]
        second = iter((0.3, 0.35, 0.4, 0.45, 0.5))

        async def fake_make_prediction(question, research):  # noqa: ARG001
            v = next(second)
            return ReasonedPrediction(prediction_value=v, reasoning=f"second {v}")

        async def fake_research(prompt, depth):  # noqa: ARG001
            return "crux research"

        result = ResearchWithPredictions(research_report="research", summary_report="summary",
                                         predictions=first)
        with mock.patch.object(claude_research, "research", fake_research), \
                mock.patch.object(bot, "_make_prediction", fake_make_prediction):
            pooled = run(bot._resolve_crux(binary(), result, [p.prediction_value for p in first],
                                           (first[0], first[-1])))
        self.assertEqual(len(pooled.predictions), 10)
        self.assertEqual(pooled.predictions[:5], first)
        self.assertEqual(sorted(p.reasoning.split()[0] for p in pooled.predictions),
                         ["first"] * 5 + ["second"] * 5)
        self.assertIn("crux research", pooled.research_report)


class AlreadyDeterminedWordingTests(unittest.TestCase):
    def test_the_binary_prompt_says_when_the_criteria_were_unmet(self):
        bot = make_bot()
        llm = FakeLlm("Probability: 50%")
        bot.llm = llm
        run(bot._run_forecast_on_binary(binary(), "research"))
        prompt = llm.prompts[0]
        self.assertIn("These criteria had not been satisfied when the question was written:",
                      prompt)
        self.assertNotIn("have not yet been satisfied", prompt)
        self.assertIn("never assume an outcome without a dated, primary source showing it",
                      prompt)


if __name__ == "__main__":
    unittest.main()
