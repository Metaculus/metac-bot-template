from datetime import datetime

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    clean_indents,
)

__all__ = [
    "binary_prompt",
    "multiple_choice_prompt",
    "numeric_prompt",
    "stacking_binary_prompt",
    "stacking_multiple_choice_prompt",
    "stacking_numeric_prompt",
]


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def binary_prompt(question: BinaryQuestion, research: str) -> str:
    """Return the forecasting prompt for binary questions with strengthened
    evidence handling, outside→inside anchoring, and a brief checklist.
    The final output format remains unchanged (last line: "Probability: ZZ%").
    """

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
            • Use integers 1%-99% (no 0 % or 100 %).
            • They must sum to 100 %.

        ── Brief checklist (keep concise) ───────────────────────────────────
        • Paraphrase options & resolution criteria (<30 words).
        • State the outside-view distribution used as anchor.
        • Consistency line: "Most likely: __; least likely: __; coherent with rationale?"
        • Top 3-5 evidence items + quick factual validity check.
        • Blind-spot statement; status-quo nudge sanity check.

        **CRITICAL**: You MUST assign a probability (1-99%) to EVERY single option listed above.
        Even if an option seems very unlikely, assign it at least 1%. Never skip any option.

        ── Final answer (must be last lines, one line per option, all options included, in same order, nothing after) ──
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
    unit_str = question.unit_of_measure or "unknown units, assume unitless (e.g. raw count)"
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

        ── Units & Bounds (must follow) ─────────────────────────────────────
        • Base units for output values: {unit_str}
        • Allowed range (in base units): [{getattr(question, "lower_bound", "?")}, {getattr(question, "upper_bound", "?")}]
        • Note: allowed range is suggestive of units! If needed, you may use it to infer units.
        • All 8 percentiles you output must be numeric values in the base unit and fall within that range.
        • If your reasoning uses B/M/k, convert to base unit numerically (e.g., 350B → 350000000000). No suffixes, just numbers.

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

        OUTPUT FORMAT, floating point numbers 
        Must be last lines, nothing after, STRICTLY INCREASING percentiles meaning e.g. p20 > p10 and not equal.
        __Example:__

        Percentile 5: 10.1
        Percentile 10: 12.3
        Percentile 20: 23.4
        Percentile 40: 34.5
        Percentile 60: 56.7
        Percentile 80: 67.8
        Percentile 90: 78.9
        Percentile 95: 89.0
        """
    )


def stacking_binary_prompt(question: BinaryQuestion, research: str, base_predictions: list[str]) -> str:
    """Return the stacking prompt for binary questions that takes multiple model predictions as input."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        You will be judged based on the accuracy and calibration of your final forecast using the Metaculus peer score (log score).
        
        Your task is to synthesize multiple expert analyses into a single, well-calibrated probability.
        
        Your Metaculus question is:
        {question.question_text}
        
        Question background:
        {question.background_info}
        
        This question's outcome will be determined by the specific criteria below:
        {question.resolution_criteria}
        
        {question.fine_print}
        
        Your research assistant provided this context:
        {research}
        
        Today is {_today_str()}.
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Model agreement analysis
           • Where do the models agree? What shared evidence drives consensus?
           • Where do they disagree? What causes divergent reasoning?
           • Are disagreements due to different evidence weighting or different evidence sources?
        
        2) Evidence synthesis
           • Which evidence appears most frequently across analyses? Is this justified?
           • What unique evidence does each model bring? How credible is it?
           • Are there systematic biases visible across models (overconfidence, anchoring, etc.)?
        
        3) Reasoning quality assessment
           • Which models demonstrate strongest analytical rigor?
           • Which models best incorporate reference class reasoning?
           • Which models show appropriate uncertainty calibration?
        
        4) Meta-level adjustments
           • Should I weight models equally or give more weight to better-reasoned analyses?
           • Are there blind spots that all models missed?
           • How should I account for model correlation vs independence?
        
        5) Final synthesis
           • What probability best integrates all the evidence and reasoning?
           • Does this probability appropriately reflect the uncertainty in the question?
           • Sanity check: does this probability make sense given the base rate and evidence?
        
        The last thing you write MUST BE your final answer as an INTEGER percentage. "Probability: ZZ%"
        An example response is: "Probability: 50%"
        """
    )


def stacking_multiple_choice_prompt(
    question: MultipleChoiceQuestion, research: str, base_predictions: list[str]
) -> str:
    """Return the stacking prompt for multiple choice questions."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        Your accuracy and calibration will be scored with Metaculus' log-score, so avoid over-confidence 
        and make sure your probabilities sum to **100%**.
        
        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}
        
        • Options (in resolution order): {question.options}
        
        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}
        
        {question.resolution_criteria}
        {question.fine_print}
        
        ── Intelligence Briefing ────────────────────────────────
        {research}
        
        Today's date: {_today_str()}
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Model agreement analysis
           • Which options show consensus vs divergence across models?
           • What shared reasoning drives agreement on likely/unlikely options?
           • Where models disagree, what drives the different assessments?
        
        2) Evidence synthesis across models
           • What evidence appears consistently? Is this justified by source quality?
           • What unique insights does each model contribute?
           • Are there systematic biases (overconfidence on favorites, neglect of tails)?
        
        3) Probability distribution analysis
           • Which models show appropriate uncertainty (avoid 0%/100%)?
           • How do the models differ in their tail probability allocation?
           • Are there systematic patterns in how models distribute probability?
        
        4) Reasoning quality assessment
           • Which analyses demonstrate strongest logical coherence?
           • Which models best incorporate reference class reasoning?
           • Which show most appropriate calibration for this question type?
        
        5) Meta-level synthesis
           • Should models be weighted equally or by reasoning quality?
           • Are there overlooked scenarios that all models missed?
           • How should I account for correlation vs independence in model errors?
        
        6) Final distribution calibration
           • What probability distribution best synthesizes all analyses?
           • Does my distribution appropriately reflect uncertainty?
           • Are my tail probabilities justified given the evidence?
        
        **CRITICAL**: You MUST assign a probability (1-99%) to EVERY single option listed above.
        Even if an option seems very unlikely, assign it at least 1%. Never skip any option.
        
        ── Final answer (must be last lines, one line per option, all options included, in same order, nothing after) ──
        Option_A: NN%
        Option_B: NN%
        …
        Option_N: NN%
        """
    )


def stacking_numeric_prompt(
    question: NumericQuestion,
    research: str,
    base_predictions: list[str],
    lower_bound_message: str,
    upper_bound_message: str,
) -> str:
    """Return the stacking prompt for numeric questions."""
    predictions_text = "\n".join([f"Model {i + 1} Analysis:\n{pred}\n" for i, pred in enumerate(base_predictions)])

    return clean_indents(
        f"""
        You are a senior meta-forecaster specializing in combining predictions from multiple expert models.
        You will be scored with Metaculus' log-score, so accuracy **and** calibration 
        (especially the width of your 90/10 interval) are critical.
        
        ── Question ──────────────────────────────────────────────────────────
        {question.question_text}
        
        ── Context ───────────────────────────────────────────────────────────
        {question.background_info}
        
        {question.resolution_criteria}
        {question.fine_print}
        
        Units: {question.unit_of_measure or "Not stated: infer if possible"}
        
        ── Units & Bounds (must follow) ─────────────────────────────────────
        • Base unit for output values: {question.unit_of_measure or "base unit"}
        • Allowed range (base units): [{getattr(question, "lower_bound", "?")}, {getattr(question, "upper_bound", "?")}]
        • All 8 percentiles you output must be numeric values in the base unit and fall within that range.
        • If your reasoning uses B/M/k, convert to base unit numerically (e.g., 350B → 350000000000). No suffixes.
        
        ── Intelligence Briefing ────────────────────────────────
        {research}
        
        Today's date: {_today_str()}
        
        {lower_bound_message}
        {upper_bound_message}
        
        ── Multiple Expert Analyses ──
        {predictions_text}
        
        ── Meta-Analysis Framework ──
        1) Distribution comparison
           • Compare the central tendencies (medians) across models - what explains differences?
           • Compare uncertainty ranges (90% intervals) - which models show appropriate calibration?
           • Are there systematic patterns in how models approach this forecasting problem?
        
        2) Evidence synthesis
           • What evidence/approaches appear across multiple analyses?
           • What unique insights or data does each model contribute?
           • Which models demonstrate strongest analytical rigor for this question type?
        
        3) Calibration assessment
           • Which models show appropriate uncertainty given the available evidence?
           • Are any models systematically overconfident (too narrow ranges)?
           • Which uncertainty ranges seem most justified by the evidence quality?
        
        4) Reference class integration
           • How do models differ in their reference class selection?
           • Which outside view approaches seem most appropriate?
           • Should I favor models with stronger reference class reasoning?
        
        5) Meta-level synthesis
           • Should I weight models equally or by reasoning quality?
           • Are there blind spots or scenarios all models missed?
           • How should I account for correlation vs independence in model approaches?
        
        6) Final distribution calibration
           • What percentiles best synthesize all the evidence and reasoning?
           • Does my final distribution appropriately reflect epistemic uncertainty?
           • Are my tails justified given the potential for unknown unknowns?
        
        Remember: Think in ranges, not points. Keep 10th and 90th percentiles appropriately wide.
        Ensure strictly increasing percentiles and respect the bounds above.
        
        OUTPUT FORMAT, floating point numbers 
        Must be last lines, nothing after, STRICTLY INCREASING percentiles meaning e.g. p20 > p10 and not equal.
        
        Percentile 5: [value]
        Percentile 10: [value]
        Percentile 20: [value]
        Percentile 40: [value]
        Percentile 60: [value]
        Percentile 80: [value]
        Percentile 90: [value]
        Percentile 95: [value]
        """
    )
