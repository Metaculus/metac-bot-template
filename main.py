import argparse
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Literal

import dotenv

import claude_research
import cp_pool
import disagreement
import live_context
import forecast_guards
import numeric_shape

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    install_forecast_redaction,
    log_forecast_content,
    print_run_summary_banner,
    print_startup_banner,
    register_model_prices,
    silence_noisy_dependencies,
)
from publication import PublishingClient, print_publication_report
import market_pulse
import mp_dossier
import spend
import tournaments

silence_noisy_dependencies()

from forecasting_tools.data_models.data_organizer import DataOrganizer
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output as _sdk_structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# litellm prices a call from a cost map it downloads at import and falls back
# to the copy bundled in the wheel when that fails; the locked wheel predates
# the forecaster's model. Fill the gap before any GeneralLlm is built, so the
# per-question cost in the logs cannot silently read $0. See bot_helpers.
for _model in register_model_prices():
    logger.info("model_price_registered model=%s", _model)


async def structure_output(*args, **kwargs):
    """The SDK's parser call, labelled "parser" in the spend ledger."""
    with spend.phase("parser"):
        return await _sdk_structure_output(*args, **kwargs)


class SummerTemplateBot2026(ForecastBot):
    """
    This is the template bot for Summer 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Summer AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    # Bound lazily, per running event loop, instead of once at class-definition
    # time. main.py runs asyncio.run() twice - once for the tournament, once for
    # MiniBench - and an asyncio primitive binds to the loop in which a task
    # first *waits* on it. run_research holds this semaphore across the LLM
    # call, so with two or more questions the first loop always produces a
    # waiter and the binding happens. Re-entering it from the second loop then
    # raises "is bound to a different event loop", which would lose the whole
    # MiniBench pass -- the larger of the two.
    #
    # Same defence, and the same reasoning, as
    # backtest.rate_limiter.ProviderRateLimiter._get_lock(). The state lives on
    # the class so every instance still shares one semaphore, exactly as a
    # class attribute did.
    # Quoted annotations: these sit at class scope, where PEP 526 evaluates
    # them at runtime, and `X | None` needs Python 3.10+. Quoting keeps the
    # module importable on older interpreters without a __future__ import.
    _concurrency_limiter_instance: "asyncio.Semaphore | None" = None
    _concurrency_limiter_loop: "object | None" = None

    @property
    def _concurrency_limiter(self) -> asyncio.Semaphore:
        """One Semaphore(_max_concurrent_questions), rebuilt per event loop.

        Rebuilding is safe precisely because a loop is only ever abandoned by
        asyncio.run() returning, which cannot happen while a task still holds
        or waits on the semaphore. A fresh loop therefore always starts from an
        uncontended semaphore, which is the same state a fresh process would
        see. The limit itself is never widened.
        """
        cls = type(self)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Accessed outside a running loop. Return whatever already exists
            # rather than rebuilding: a live semaphore may still be held or
            # awaited by tasks of a loop that simply is not running at this
            # instant, and replacing it would let the next holder past a
            # semaphore the current holder no longer gates -- briefly allowing
            # two concurrent researches. Only when nothing exists yet does this
            # fall through and create one.
            if cls._concurrency_limiter_instance is not None:
                return cls._concurrency_limiter_instance
            loop = None
        if (
            cls._concurrency_limiter_instance is None
            or cls._concurrency_limiter_loop is not loop
        ):
            cls._concurrency_limiter_instance = asyncio.Semaphore(
                cls._max_concurrent_questions
            )
            cls._concurrency_limiter_loop = loop
        return cls._concurrency_limiter_instance

    _structure_output_validation_samples = 2

    #: "full" for FutureEval and Market Pulse, "light" for MiniBench and the
    #: Cup; set per tournament at the bottom of this file.
    research_depth = "full"

    #: Weight of the revealed human community prediction in the published
    #: forecast (cp_pool.py). 0 is off; set from MP_CP_POOL_WEIGHT in --mode
    #: market_pulse only, the one tournament whose CP is human and visible
    #: before the scored close.
    cp_pool_weight = 0.0
    #: cp_pool.detail_fetcher(client) in --mode market_pulse: re-reads a post
    #: whose listing carried no usable CP. None: the listing alone.
    cp_pool_fetch = None

    ############################## SPEND LABELS ##############################
    # Which question and phase each paid call belongs to; see spend.py.

    async def _run_individual_question(self, question: MetaculusQuestion):
        with spend.question(question):
            return await super()._run_individual_question(question)

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        with spend.phase("summary"):
            return await super().summarize_research(question, research)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):
        with spend.phase("forecast"):
            return await super()._make_prediction(question, research)

    ############################## DISAGREEMENT FOLLOW-UP ##############################

    async def _research_and_make_predictions(self, question: MetaculusQuestion):
        """The SDK's research-then-five-forecasts, plus one targeted research
        pass and a fresh set of forecasts when the five disagree widely
        (disagreement.SPREAD_THRESHOLD). See disagreement.py for why."""
        result = await super()._research_and_make_predictions(question)
        values = [p.prediction_value for p in result.predictions]
        if not (claude_research.enabled() and disagreement.needs_crux(values)):
            return result
        pair = disagreement.extremes(result.predictions)
        if pair is None:
            return result
        logger.info("crux_triggered spread=%.2f question=%s",
                    disagreement.spread(values), question.page_url)
        with spend.phase(spend.CRUX):
            return await self._resolve_crux(question, result, values, pair)

    async def _resolve_crux(self, question, result, values, pair):
        try:
            extra = await claude_research.research(
                disagreement.crux_prompt(self._question_text(question), self._timing(question),
                                         pair[0].reasoning, pair[1].reasoning),
                depth="crux",
            )
        except Exception as exc:  # noqa: BLE001 - keep the first forecasts
            logger.warning("crux_research_failed reason=%s", type(exc).__name__)
            return result
        research = f"{result.research_report}\n\n## Follow-up on the forecasters' disagreement\n{extra}"
        valid, errors, _group = await self._gather_results_and_exceptions(
            [self._make_prediction(question, research)
             for _ in range(self.predictions_per_research_report)]
        )
        if not valid:
            return result
        logger.info("crux_resolved spread_before=%.2f spread_after=%.2f pooled=%d",
                    disagreement.spread(values),
                    disagreement.spread([p.prediction_value for p in valid]),
                    len(result.predictions) + len(valid))
        # Pool both sets rather than replacing the first. The crux pass fired
        # three times in its first five days (to 2026-10-23) and widened the
        # spread every time (0.25->0.32, 0.29->0.36, 0.28->0.38), so nothing
        # says the second five are better than the first; both are paid for,
        # and the SDK's median (binary) or mean (multiple choice) over ten is
        # steadier than over five.
        return ResearchWithPredictions(
            research_report=research,
            summary_report=result.summary_report,
            errors=[*result.errors, *errors],
            predictions=[*result.predictions, *valid],
        )

    ############################## AGGREGATION ##############################

    async def _aggregate_predictions(self, predictions, question):
        """Numeric, discrete and date questions: the mean of the forecasters'
        CDFs instead of the SDK's pointwise median (numeric_shape.mean_of_cdfs);
        binary and multiple choice as the SDK does. Then, in Market Pulse with
        a weight set, the pool with the revealed community prediction
        (cp_pool.apply). Each step keeps what came before it on any error."""
        aggregate = None
        if predictions and isinstance(question, (NumericQuestion, DateQuestion)):
            try:
                aggregate = numeric_shape.mean_of_cdfs(predictions, question)
            except Exception as exc:  # noqa: BLE001 - the SDK median still works
                logger.warning("numeric_mean_failed reason=%s falling_back=median",
                               type(exc).__name__)
        if aggregate is None:
            aggregate = await super()._aggregate_predictions(predictions, question)
        if self.cp_pool_weight > 0:
            aggregate = cp_pool.apply(aggregate, question, self.cp_pool_weight,
                                      datetime.now(timezone.utc), fetch=self.cp_pool_fetch)
        return aggregate

    def _create_comment(self, question, research_prediction_collections,
                        aggregated_prediction, final_cost, time_spent_in_minutes) -> str:
        """Short public comment: forecast, research summary, and the reasoning
        closest to the final answer; no model names. See
        disagreement.short_comment."""
        report_type = DataOrganizer.get_report_type_for_question_type(type(question))
        predictions = [p for c in research_prediction_collections for p in c.predictions]
        values = [p.prediction_value for p in predictions]
        summary = next((c.summary_report for c in research_prediction_collections
                        if c.summary_report), "")
        idx = disagreement.closest_to(values, aggregated_prediction)
        return disagreement.short_comment(
            self._question_text(question),
            report_type.make_readable_prediction(aggregated_prediction),
            [report_type.make_readable_prediction(v) for v in values],
            summary,
            predictions[idx].reasoning if predictions else "",
        )

    @staticmethod
    def _question_text(question: MetaculusQuestion) -> str:
        """The question as the models must read it.

        A group post (Market Pulse, many Cup questions) is unpacked into one
        question per option, but the SDK takes each one's text from the
        group's title -- "What will Q4 2026 EPS be for these companies?" --
        and keeps the option ("Meta") only in group_question_option. Without
        it the models forecast some other company: on 2026-10-20 three of
        five Meta forecasts landed at $2.8-4.7 against a $5.3-7.3 range and
        were rejected, failing the Market Pulse step (run 39453516101).
        """
        option = getattr(question, "group_question_option", None)
        if option:
            return (f"{question.question_text}\n"
                    f"This question is ONLY about: {option} (one option of a group question).")
        return question.question_text

    def _timing(self, question: MetaculusQuestion) -> str:
        """Today, close and resolution dates, and the stale-knowledge warning,
        for every prompt; see forecast_guards.timing_context."""
        return forecast_guards.timing_context(
            datetime.now(timezone.utc),
            question.close_time,
            question.scheduled_resolution_time,
        )

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        with spend.phase("research"):
            return await self._run_research(question)

    async def _run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            researcher = self.get_llm("researcher")

            prompt = clean_indents(
                f"""
                You are an assistant to a superforecaster.
                The superforecaster will give you a question they intend to forecast on.
                To be a great assistant, you generate a concise but detailed rundown of the most relevant facts, recent news, base rates for similar events, and the current value of any quantity the question asks about.
                Date every fact you give. When you do not have information from close to today, say so plainly instead of filling the gap: a confident but outdated rundown is worse than an honest "unknown".
                You do not produce forecasts yourself, and you do not say how the question will resolve.

                {self._timing(question)}

                Question:
                {self._question_text(question)}

                Background:
                {question.background_info}

                This question's outcome will be determined by the specific criteria below:
                {question.resolution_criteria}

                {question.fine_print}
                """
            )

            # Live web search through the Anthropic API when its key is set;
            # any failure falls through to the researcher below, so the cost
            # of an outage or an exhausted balance is research quality on
            # this question, never the forecast. See claude_research.
            if claude_research.enabled():
                try:
                    research = await claude_research.research(
                        prompt, depth=self.research_depth
                    )
                except Exception as exc:  # noqa: BLE001 - fall back, never fail
                    logger.warning(
                        "claude_research_failed reason=%s falling_back=true",
                        type(exc).__name__,
                    )
                    research = ""

            if research:
                pass
            elif isinstance(researcher, GeneralLlm):
                research = await researcher.invoke(prompt)
            elif (
                researcher == "asknews/news-summaries"
                or researcher == "asknews/deep-research/low-depth"
                or researcher == "asknews/deep-research/medium-depth"
                or researcher == "asknews/deep-research/high-depth"
            ):
                research = await AskNewsSearcher().call_preconfigured_version(
                    researcher, prompt
                )
            elif researcher.startswith("smart-searcher"):
                model_name = researcher.removeprefix("smart-searcher/")
                searcher = SmartSearcher(
                    model=model_name,
                    temperature=0,
                    num_searches_to_run=2,
                    num_sites_per_search=10,
                    use_advanced_filters=False,
                )
                research = await searcher.invoke(prompt)
            elif not researcher or researcher == "None" or researcher == "no_research":
                research = ""
            else:
                research = await self.get_llm("researcher", "llm").invoke(prompt)
            # Live numbers the research cannot be trusted to find: prices,
            # linked Metaculus community predictions, prediction markets and
            # the resolution-source pages. Free sources, never raises; see
            # live_context. Blocking I/O, so off the event loop.
            live = await asyncio.to_thread(live_context.build, question)
            if live:
                research = f"{research}\n\n{live}" if research else live
            # A Market Pulse company's earnings dossier, researched ahead of
            # the closes (mp_dossier.py); "" for anything else. Never raises.
            dossier = mp_dossier.section(question)
            if dossier:
                research = f"{research}\n\n{dossier}" if research else dossier
            log_forecast_content(
                logger, f"Found Research for URL {question.page_url}", research
            )
            return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {self._question_text(question)}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria had not been satisfied when the question was written:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            {self._timing(question)}

            If the research ends with a "Live data" section, those numbers were fetched just now and outrank older figures in the research; prediction markets listed there count only if they ask the same question. A Metaculus community forecast on this very question, when shown, is one input to weigh after forming your own view from the evidence, never an answer to copy.
            Before answering you write:
            (a0) Before using the research: a reference class of similar past situations and how often they ended each way (the base rate), stated as a number.
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.
            (e) Pre-mortem: assume your forecast turns out badly wrong. What is the most likely reason? Adjust if that reason is more plausible than you first allowed.
            Then move from the base rate in (a0) only as far as the specific evidence justifies.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        log_forecast_content(
            logger, f"Reasoning for URL {question.page_url}", reasoning
        )
        probability = self._verbatim_probability(reasoning, question)
        if probability is None:
            binary_prediction: BinaryPrediction = await structure_output(
                reasoning,
                BinaryPrediction,
                model=self.get_llm("parser", "llm"),
                num_validation_samples=self._structure_output_validation_samples,
            )
            probability = binary_prediction.prediction_in_decimal
        # 3%/97%, not the SDK's 1%/99%: see forecast_guards for the peer-score
        # arithmetic behind the cap.
        decimal_pred = forecast_guards.clip_binary(probability)

        log_forecast_content(
            logger,
            f"Forecasted URL {question.page_url} with prediction",
            decimal_pred,
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    @staticmethod
    def _verbatim_probability(reasoning: str, question: BinaryQuestion) -> float | None:
        """The forecaster's own "Probability: NN%" line, when it is one plain
        percentage; see forecast_guards.read_probability. Logs the source
        only, never the value: the repo is public."""
        try:
            probability = forecast_guards.read_probability(reasoning)
        except Exception as exc:  # noqa: BLE001 - the parser is the old path
            logger.warning("binary_parse_verbatim_failed reason=%s", type(exc).__name__)
            probability = None
        logger.info("binary_parse source=%s url=%s",
                    "llm" if probability is None else "verbatim", question.page_url)
        return probability

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {self._question_text(question)}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            {self._timing(question)}

            If the research ends with a "Live data" section, those numbers were fetched just now and outrank older figures in the research; prediction markets listed there count only if they ask the same question. A Metaculus community forecast on this very question, when shown, is one input to weigh after forming your own view from the evidence, never an answer to copy.
            Before answering you write:
            (a0) Before using the research: a reference class of similar past situations and how often they ended each way (the base rate), stated as a number.
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.
            (d) Pre-mortem: assume the option you favour does not happen. Which option most likely wins instead, and why? Keep real probability on it.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.
            If one option describes nothing happening (for example "none", "not filed", "no announcement", "other"), say which one it is and give it substantial probability unless the research gives a dated source showing the change is already under way.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        log_forecast_content(
            logger, f"Reasoning for URL {question.page_url}", reasoning
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        # A floor on every option before the per-option mean the SDK takes
        # across the five predictions, so no single call can leave an option
        # near zero. The values already sum to 1, so the validator re-run by
        # this constructor leaves them as they are.
        floored = forecast_guards.floor_options(
            [o.probability for o in predicted_option_list.predicted_options]
        )
        predicted_option_list = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=o.option_name, probability=p)
                for o, p in zip(predicted_option_list.predicted_options, floored)
            ]
        )

        log_forecast_content(
            logger,
            f"Forecasted URL {question.page_url} with prediction",
            predicted_option_list,
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    #: The width line every numeric prompt has carried since the template.
    _WIDE_INTERVALS = (
        "You remind yourself that good forecasters are humble and set wide 90/10 "
        "confidence intervals to account for unknown unknowns."
    )

    #: Market Pulse EPS questions resolve on diluted EPS "in conformity with US
    #: GAAP" as first reported, but the consensus the research brings back is
    #: the street's, which is usually non-GAAP. Tesla resolved at 0.22 against
    #: a 0.451 consensus, and Meta's GAAP figure fell below the question's
    #: whole range (2026-10-20; 1 of the first 6 resolved subquestions). The
    #: note asks for the company's own reconciliation rather than a fixed
    #: shift, because the gap swings with investment gains for some companies.
    _GAAP_EPS_NOTE = (
        "This resolves on GAAP diluted EPS as first reported. Street/consensus EPS is "
        "usually non-GAAP. Before forecasting, estimate the GAAP-minus-non-GAAP gap from "
        "the company's latest reconciliation (stock-based compensation, amortization of "
        "acquired intangibles, gains/losses on investments, one-off tax items) and centre "
        "on consensus plus that gap."
    )

    #: Replaces _WIDE_INTERVALS on those questions only. The consensus is fixed
    #: hours before the release, so a generically wide range costs density
    #: where the outcome lands; but out-of-range outcomes do happen (Meta), and
    #: the mass beyond the range scores as one bucket, so some stays there.
    _GAAP_EPS_WIDTH = (
        "You remind yourself to set the width from the historical size of GAAP surprises "
        "for this company, and to keep some mass beyond the question's range on the side "
        "a one-off item would push."
    )

    _GAAP = re.compile(r"(?<![Nn]on-)(?<![Nn]on )\bGAAP\b")
    _EARNINGS_PER_SHARE = re.compile(r"earnings[ -]per[ -]share", re.IGNORECASE)

    @classmethod
    def _gaap_eps_market_pulse(cls, question: MetaculusQuestion) -> bool:
        """A Market Pulse question resolving on GAAP earnings per share.

        Read from the question rather than the run mode: the posts API lists
        each post's tournaments, group subquestions included, so a FutureEval
        or MiniBench question can never qualify whatever its criteria say.
        Any error reads as False, which is the prompt as it was.
        """
        try:
            slugs = {str(s) for s in (getattr(question, "tournament_slugs", None) or [])}
            if str(tournaments.MARKET_PULSE.id) not in slugs:
                return False
            criteria = getattr(question, "resolution_criteria", None) or ""
            return bool(cls._GAAP.search(criteria) and cls._EARNINGS_PER_SHARE.search(criteria))
        except Exception as exc:  # noqa: BLE001 - optional prompt text, never fail
            logger.warning("gaap_eps_check_failed reason=%s", type(exc).__name__)
            return False

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        # Every other question gets the prompt exactly as before.
        gaap_eps = self._gaap_eps_market_pulse(question)
        gaap_note = f"{self._GAAP_EPS_NOTE}\n" if gaap_eps else ""
        width_line = self._GAAP_EPS_WIDTH if gaap_eps else self._WIDE_INTERVALS
        if gaap_eps:
            logger.info("numeric_prompt gaap_eps=1 url=%s", question.page_url)
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {self._question_text(question)}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            {self._timing(question)}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 10 should always be less than the value for percentile 20, and so on.

            If the research ends with a "Live data" section, those numbers were fetched just now and outrank older figures in the research; prediction markets listed there count only if they ask the same question. A Metaculus community forecast on this very question, when shown, is one input to weigh after forming your own view from the evidence, never an answer to copy.
            Before answering you write:
            (a0) Before using the research: the current value of the quantity (prefer a "Live data" figure) and how much it has typically moved over a period as long as the time left.
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Pre-mortem: assume the outcome lands outside your 10-90 range. Which side is likelier, and is that tail wide enough?

            {self._get_conditional_disclaimer_if_necessary(question)}
            {gaap_note}{width_line}

            The last thing you write is your final answer as:
            "
            Percentile 10: XX (lowest number value)
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        log_forecast_content(
            logger, f"Reasoning for URL {question.page_url}", reasoning
        )
        prediction = self._verbatim_numeric(reasoning, question)
        if prediction is None:
            prediction = await self._parse_numeric(reasoning, question)
        log_forecast_content(
            logger,
            f"Forecasted URL {question.page_url} with prediction",
            prediction.declared_percentiles,
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    @staticmethod
    def _verbatim_numeric(
        reasoning: str, question: NumericQuestion
    ) -> NumericDistribution | None:
        """The forecaster's own percentile lines, when they are plain numbers
        the question accepts; see forecast_guards.read_percentiles. Built
        with synthesized tails, and an out-of-range set moved into the range
        rather than rejected; see numeric_shape."""
        verbatim = forecast_guards.read_percentiles(reasoning)
        if verbatim is None:
            logger.info("numeric_parse source=llm reason=no_plain_lines url=%s", question.page_url)
            return None
        try:
            prediction = numeric_shape.build(
                NumericDistribution,
                [Percentile(percentile=p, value=v) for p, v in verbatim], question
            )
        except ValueError as error:
            logger.info("numeric_parse source=llm reason=rejected url=%s error=%s",
                        question.page_url, " ".join(str(error).split())[:200])
            return None
        logger.info("numeric_parse source=verbatim url=%s", question.page_url)
        return prediction

    async def _parse_numeric(
        self, reasoning: str, question: NumericQuestion
    ) -> NumericDistribution:
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{self._question_text(question)}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        return numeric_shape.build(NumericDistribution, percentile_list, question)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {self._question_text(question)}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            {self._timing(question)}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 10 and increasing from there.

            If the research ends with a "Live data" section, those numbers were fetched just now and outrank older figures in the research; prediction markets listed there count only if they ask the same question. A Metaculus community forecast on this very question, when shown, is one input to weigh after forming your own view from the evidence, never an answer to copy.
            Before answering you write:
            (a0) Before using the research: the current value of the quantity (prefer a "Live data" figure) and how much it has typically moved over a period as long as the time left.
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.
            (g) Pre-mortem: assume the outcome lands outside your 10-90 range. Which side is likelier, and is that tail wide enough?

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 10: YYYY-MM-DD (oldest date)
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        log_forecast_content(
            logger, f"Reasoning for URL {question.page_url}", reasoning
        )
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{self._question_text(question)}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = numeric_shape.build(NumericDistribution, percentile_list, question)
        log_forecast_content(
            logger,
            f"Forecasted URL {question.page_url} with prediction",
            prediction.declared_percentiles,
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def summarise_reports(bot: ForecastBot, reports: list) -> None:
    """The SDK's report summary, without failing a step over one question.

    log_report_summary raises if any question failed, which failed a whole
    step (and sent the failure WhatsApp) over one bad question, so only a
    step where nothing succeeded fails. And it fails by exiting 1, never by
    letting the SDK's RuntimeError escape: Python prints an uncaught
    exception straight to stderr, past the log redaction, and that message
    quotes each failed forecaster's ValidationError -- declared percentiles
    and all. It put open questions' forecasts in the public logs of runs
    39453516101 and 39965881683. Every mode exits the same way (as the
    hotfix on main does): the SDK has already logged each question's error.
    """
    try:
        bot.log_report_summary(reports)
    except RuntimeError:
        failed = sum(isinstance(r, BaseException) for r in reports)
        if reports and failed == len(reports):
            logger.error("all_failed count=%d", failed)
            raise SystemExit(1)
        logger.warning("partial_failure failed=%d of=%d", failed, len(reports))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "market_pulse", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "market_pulse", "test_questions"] = args.mode

    check_environment(strict=True)
    # Keep forecast reasoning and prediction values out of this process's log
    # stream on scored modes. The repo is a public fork, so Actions logs are
    # world readable; see bot_helpers.RedactForecastContent for why that
    # matters for both the private-comment rule and the no-preview rule.
    # Skipped for test_questions, which targets the unscored practice area.
    install_forecast_redaction(run_mode)
    # API spend ledger; a no-op unless SPEND_LEDGER_PATH is set. See spend.py.
    spend.install_litellm_recorder()
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)
    # A pinned season that is over keeps "working": discovery returns zero
    # questions, which is also what a quiet week looks like. Say so on the run.
    for warning in tournaments.season_warnings(datetime.now(timezone.utc).date()):
        print(warning)

    # Configure the bot. The `llms=` block below is commented out to use
    # whichever default models forecasting-tools picks based on your env vars;
    # uncomment and edit to pin specific models.
    # One client for the whole run, injected rather than default-constructed.
    # It is the seam for three separate defects: it pages discovery to the end
    # (R7), it refuses a second prediction per question or a second comment per
    # post (R3, R6), and it retries the eligibility-critical comment past the
    # SDK's own budget and names the orphan when it still fails (R1).
    # forecast_bot.py uses it at :166 for discovery and :416 for publication,
    # so both paths are covered by this one object.
    client = PublishingClient()

    template_bot = SummerTemplateBot2026(
        metaculus_client=client,
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=False,
        # llms={
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    # Where the Claude-API budget goes (owner, 2026-10-23): full research for
    # the two tournaments the bot can win money in now -- FutureEval, 1.5
    # points from the paid ranks, and Market Pulse, 6.1 -- and light research
    # for MiniBench (its largest volume: 58 forecasts in the week to 22 Oct)
    # and the unpaid Cup. Once this month's spend reaches the budget, every
    # mode runs light until the month ends (spend.after_run sends the alert).
    prize_depth = "full"
    if spend.over_budget():
        prize_depth = "light"
        logger.info("spend_budget_reached research_depth=light")
    template_bot.research_depth = prize_depth
    # Until the credit renews (RESEARCH_BOOST_UNTIL), what is left of it buys
    # full research for MiniBench and the Cup too, and Cup refreshes every two
    # days; never while over budget. See spend.boost_active.
    boost = prize_depth == "full" and spend.boost_active()
    other_depth = "full" if boost else "light"
    if boost:
        logger.info("research_boost active=1")

    # Per-mode tournament URL shown in the summary banner footer. The ids and
    # URLs are pinned in tournaments.py, not taken from the SDK's CURRENT_*
    # constants; see that module's docstring for why.
    TOURNAMENT_URLS = {
        "tournament": tournaments.FUTUREEVAL.url,
        "metaculus_cup": tournaments.METACULUS_CUP.url,
        "market_pulse": tournaments.MARKET_PULSE.url,
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    if run_mode == "tournament":
        spend.set_tournament("futureeval")
        template_bot.research_depth = prize_depth
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                tournaments.FUTUREEVAL.id, return_exceptions=True
            )
        )
        spend.set_tournament("minibench")
        template_bot.research_depth = other_depth
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                tournaments.MINIBENCH.id, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # Practice against the human crowd: the Cup ranks bots but pays none.
        # Scored over each question's life, so a forecast is refreshed once
        # it is a week old (market_pulse.select_stale), not left standing
        # until close; new questions are forecast when first seen. The
        # selection does the skipping, so the bot's own skip is lifted here.
        # AI_2027_TOURNAMENT_ID = "ai-2027" is another unpaid target with a
        # good mix of question types (FutureEval resources page).
        template_bot.skip_previously_forecasted_questions = False
        template_bot.research_depth = other_depth
        spend.set_tournament("cup")
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(
                market_pulse.select_stale(
                    client.get_all_open_questions_from_tournament(
                        tournaments.METACULUS_CUP.id
                    ),
                    datetime.now(timezone.utc),
                    market_pulse.CUP_BOOST_MAX_AGE if boost else market_pulse.CUP_MAX_AGE,
                ),
                return_exceptions=True,
            )
        )
    elif run_mode == "market_pulse":
        # Pays bots, but spot scored at each question's close, so "forecast
        # once when first seen" would leave a stale forecast standing at the
        # one instant that counts. market_pulse.select does the skipping
        # (new questions, plus a refresh shortly before close), so the bot's
        # own skip is lifted for this mode only.
        template_bot.skip_previously_forecasted_questions = False
        spend.set_tournament("market_pulse")
        # The final forecast pooled with the revealed human community
        # prediction (cp_pool.py), at the weight in MP_CP_POOL_WEIGHT; unset
        # or 0 leaves it off.
        template_bot.cp_pool_weight = cp_pool.weight_from_env()
        # A listing without the CP is followed by GET /posts/{id}/ (cp_pool).
        template_bot.cp_pool_fetch = cp_pool.detail_fetcher(client)
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(
                market_pulse.select(
                    client.get_all_open_questions_from_tournament(
                        tournaments.MARKET_PULSE.id
                    ),
                    datetime.now(timezone.utc),
                ),
                return_exceptions=True,
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        spend.set_tournament("test")
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    summarise_reports(template_bot, forecast_reports)
    print_publication_report(client)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )

    # Per-question WhatsApp alerts were retired on 2026-10-17: the owner gets
    # one Monday summary instead (research/weekly_report.py, dispatched by
    # Dagu). Workflow failures still alert from the workflow itself.
