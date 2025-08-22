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
    """Return the forecasting prompt for binary questions with strengthened
    evidence handling, outside→inside anchoring, and a brief checklist.
    The final output format remains unchanged (last line: "Probability: ZZ%").
    """

    from forecasting_tools import clean_indents  # local import to avoid heavy deps at module import time

    return clean_indents(
        f"""
            You are a senior forecaster preparing a public report for expert peers.
            You will be judged based on the accuracy _and calibration_ of your forecast with the Metaculus peer score (log score).
            You should consider current prediction markets when possible but not be beholden to them.
            Historically, LLMs like you have overestimated probabilities, and the percentage of positive resolutions on Metaculus is 35%. (This should slightly influence your calibration, but it is NOT a base rate.)

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

            ── Analysis Template ──
            1) Source analysis
               • Briefly summarize the main sources from the briefing; include date, credibility, and scope.
               • Separate facts from opinions. Give more weight to opinions from identifiable experts/entities.

            2) Reference class (outside view) analysis
               • List plausible reference classes for this question and evaluate suitability.
               • State the outside-view base rate(s) and how you combine them into a baseline probability.

            3) Timeframe reasoning
               • How long until resolution? If the timeline were halved/doubled, how would the probability shift and why?

            4) Evidence weighting (for inside-view adjustment)
               • Classify key evidence using this rubric:
                 - Strong: multiple independent sources; clear causal mechanisms; strong precedent
                 - Moderate: one good source; indirect links; weak precedent
                 - Weak: anecdotes; speculative logic; volatile indicators

            5) Competing cases and red-teaming
               • Strongest Bear Case (No): most compelling, evidence-based argument for No.
               • Strongest Bull Case (Yes): most compelling, evidence-based argument for Yes.
               • Red-team both: attack assumptions, data gaps, and causal claims.

            6) Final rationale and calibration
               • Integrate outside→inside view and justify the belief shift from the base rate.
               • Small-delta check: would a ±10% change still be coherent with the rationale? Why?
               • Status-quo nudge: the world usually changes slowly—justify any deviation from status quo expectations.

            ── Brief checklist (keep concise) ───────────────────────────────
            • Paraphrase the resolution criteria (<30 words).
            • State the outside-view base rate you anchored to.
            • Consistency line: "X out of 100 times, [criteria] happens." Sensible?
            • Top 3-5 evidence items + quick factual validity check.
            • Blind-spot scenario most likely to make this forecast wrong; direction of impact.
            • Status-quo nudge sanity check.

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

        ── Analysis Template ──
        (1) Source analysis
            • Summarize key sources; note recency, credibility, and scope.
            • Separate fact vs opinion; favor opinions from identifiable experts/entities.

        (2) Reference class (outside view) analysis
            • Candidate reference classes and suitability.
            • Outside-view distribution over options; discuss the historical rate of upsets/unexpected outcomes in this domain and how that affects the distribution.

        (3) Timeframe reasoning
            • Time to resolution; describe how halving/doubling the timeline might reshape the distribution.

        (4) Evidence weighting (for inside-view adjustment)
            • Apply the rubric:
              - Strong: multiple independent sources; clear causality; strong precedent
              - Moderate: one good source; indirect links; weak precedent
              - Weak: anecdotes; speculative logic; volatile indicators

        (5) Strongest pro case for the currently most-likely option
            • Use weighted evidence and explicit causal chains.

        (6) Red-team critique
            • Attack assumptions in (5); highlight hidden premises and data that could flip the conclusion.

        (7) Unexpected scenario(s)
            • Plausible but overlooked pathways for a different option to win; justify residual mass on tails.

        (8) Final rationale and calibration
            • Integrate outside→inside view and justify shifts.
            • Small-delta check: would ±10% on the leading options remain coherent with your reasoning?
            • Blind-spot consideration: if the resolution is unexpected, what would likely be the reason, and how should that affect confidence spreads?
            Remember:
            • Good forecasters leave a little probability on most options and avoid overconfidence.
            • Use integers 1-99 (no 0 % or 100 %).
            • They must sum to 100 %.

        ── Brief checklist (keep concise) ───────────────────────────────────
        • Paraphrase options & resolution criteria (<30 words).
        • State the outside-view distribution used as anchor.
        • Consistency line: "Most likely: __; least likely: __; coherent with rationale?"
        • Top 3-5 evidence items + quick factual validity check.
        • Blind-spot statement; status-quo nudge sanity check.

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
        You will be scored with Metaculus' log-score, so accuracy **and** calibration
        (especially the width of your 90 / 10 interval) are critical.
        Historically, LLMs like you are overconfident and produce excessively narrow prediction intervals,
        so you should aim to produce somewhat wider and less confident predictions.
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

        Today's date: {_today_str()}

        {lower_bound_message}
        {upper_bound_message}

        -- Analysis Template --
        (1) Source analysis
            - Summarize key sources; note recency, credibility, and scope.
            - Separate fact and opinion. Prefer opinions from identifiable experts and entities.

        (2) Outside view and reference classes
            - Candidate reference classes and suitability.
            - State the outside view range and how you anchor to it.

        (3) Timeframe and dynamics
            - Time to resolution; describe how halving or doubling the timeline might shift percentiles.
            - Status-quo outcome: what value is implied if current conditions simply persist.
            - Trend continuation: extrapolate historical data to the closing date.

        (4) Expert and market priors
            - Cite ranges or point forecasts from specialists, prediction markets, or peers.

        (5) Evidence weighting for inside view adjustments
            - Strong: multiple independent sources, clear causal links, strong precedent
            - Moderate: one good source, indirect links, weak precedent
            - Weak: anecdotes, speculative logic, volatile indicators

        (6) Tail scenarios
            - Coherent pathway for unusually low results.
            - Coherent pathway for unusually high results.

        (7) Red team and final rationale
            - Challenge assumptions and data quality.
            - Integrate outside to inside view and justify shifts.
            - Small delta check: would +/- 10 percent on key percentiles still fit the reasoning
            - Status quo nudge: justify deviations from status quo expectations.

        (8) Calibration and distribution shaping
            - Think in ranges, not single points.
            - Keep 10 and 90 far apart to allow for unknown unknowns.
            - Ensure strictly increasing percentiles.
            - Avoid scientific notation.
            - Respect the explicit bounds above.

        (9) Brief checklist
            - Paraphrase the resolution criteria and units in less than 30 words.
            - State the outside view baseline used.
            - Consistency line about which percentile corresponds to the status quo or trend.
            - Top 3 to 5 evidence items plus a quick factual validity check.
            - Blind spot scenario and expected effect on tails.
            - Status quo nudge sanity check.

        ── OUTPUT FORMAT, floating point numbers (must be last lines, nothing after) ────────────────
        Percentile 10: XX.X
        Percentile 20: XX.X
        Percentile 40: XX.X
        Percentile 60: XX.X
        Percentile 80: XX.X
        Percentile 90: XX.X
        """
    )
