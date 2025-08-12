from __future__ import annotations

from datetime import datetime

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

__all__ = [
    "binary_prompt",
    "multiple_choice_prompt",
    "numeric_prompt",
]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def binary_prompt(question: BinaryQuestion, research: str) -> str:
    """Return the forecasting prompt for binary questions.

    The body is copied verbatim from the original TemplateForecaster implementation
    to ensure behaviour is unchanged.
    """

    from forecasting_tools import clean_indents  # local import to avoid heavy deps at module import time

    return clean_indents(
        f"""
            You are a senior forecaster preparing a public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.
            Historically, LLMs like you have overestimated probabilities, and the base rate for positive resolutions on Metaculus is 35%. (This should slightly influence your calibration, but it is NOT a base rate.)

            Your Metaculus question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {_today_str()}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) The historical base rate or plausible base rates with weighting for each.
            (d) The Strongest Bear Case (FOR 'No'): Construct the most compelling, evidence-based argument for a 'No' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (e) The Strongest Bull Case (FOR 'Yes'): Construct the most compelling, evidence-based argument for a 'Yes' outcome. Your argument must be powerful enough to convince a skeptic. Cite specific facts, data points, or causal chains from the Intelligence Briefing.
            (f) Red team critique of the Strongest Bull Case and Strongest Bear Case.
            (g) Final Rationale: Synthesize the above points into a concise, final rationale. Explain how you are balancing the base rate, the strength of the competing arguments, and the severity of their respective flaws to arrive at your final estimate. Also consider that you will be judged on your Metaculus peer score (log score) and that calibration matters.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"
            An example response is: "Probability: 50%"
            """
    )


def multiple_choice_prompt(question: MultipleChoiceQuestion, research: str) -> str:
    from forecasting_tools import clean_indents

    return clean_indents(
        f"""
        You are a **senior forecaster** preparing a rigorous public report for expert peers.
        Your accuracy and *calibration* will be scored with Metaculus' log-score, so avoid
        over-confidence and make sure your probabilities sum to **100 %**.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}

        • Options (in resolution order): {question.options}

        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today's date: {_today_str()}

        ── Write your analysis in the following numbered sections ────────────
        (1) **Time to resolution**: how long until the panel can decide.

        (2) **Status-quo outcome**: if present trends simply continue, which
            option is most plausible and why?

        (3) **Base-rate & expert priors**: assemble a table like:
            Option | Historical / analogous base-rate | Expert / market signal
            -------|-----------------------------------|-----------------------
            A      | …                                 | …
            …      | …                                 | …

        (4) **Strongest pro case** for the *currently most-likely* option
            (use evidence & causal chains from the briefing).

        (5) **Red-team critique**: attack the argument in (4); highlight
            hidden assumptions or data that could flip the conclusion.

        (6) **Unexpected scenario**: outline a plausible but overlooked
            pathway that would make a different option win.

        (7) **Final rationale**: reconcile everything above into calibrated
            probabilities.  Remember:
            • Good forecasters leave a little probability on most options.
            • Use integers 1-99 (no 0 % or 100 %).
            • They must sum to 100 %.

        ── OUTPUT FORMAT (must be last lines, nothing after) ────────────────
        Option_A: NN%
        Option_B: NN%
        …
        Option_N: NN%
        """
    )


def numeric_prompt(
    question: NumericQuestion,
    research: str,
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    from forecasting_tools import clean_indents

    return clean_indents(
        f"""
        You are a **senior forecaster** writing a public report for expert peers.
        You will be scored with Metaculus’ log-score, so accuracy **and** calibration
        (especially the width of your 90 / 10 interval) are critical.
        Please consider news, research, and prediction markets, but you are not beholden to them.

        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}

        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}

        {question.resolution_criteria}
        {question.fine_print}

        Units: {question.unit_of_measure or "Not stated: infer if possible"}

        ── Intelligence Briefing (assistant research) ────────────────────────
        {research}

        Today’s date: {_today_str()}

        {lower_bound_message}
        {upper_bound_message}

        ── Write your analysis in the following numbered sections ────────────
        (1) **Time to resolution**: how long until we know the answer.

        (2) **Status-quo outcome**: what value is implied if current
            conditions simply persist?

        (3) **Trend continuation**: extrapolate historical data to 
            the closing date.

        (4) **Expert & market priors**: cite ranges or point forecasts from
            specialists, prediction markets, or peer forecasts.

        (5) **Unexpected low scenario**: describe a coherent pathway that
            would push the result into an unusually *low* tail.

        (6) **Unexpected high scenario**: analogous pathway for an unusually
            *high* tail.

        (7) **Red-team critique & final rationale**: challenge your own
            assumptions, then state how you weight everything to set each
            percentile.  Good forecasters:
            • keep 10 / 90 far apart (unknown unknowns)  
            • ensure strictly increasing values  
            • avoid scientific notation  
            • respect the explicit bounds above.

        ── OUTPUT FORMAT, floating point numbers (must be last lines, nothing after) ────────────────
        Percentile 10: XX.X
        Percentile 20: XX.X
        Percentile 40: XX.X
        Percentile 60: XX.X
        Percentile 80: XX.X
        Percentile 90: XX.X
        """
    )
