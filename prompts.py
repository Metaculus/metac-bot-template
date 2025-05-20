from forecasting_tools import clean_indents
from datetime import datetime

# Binary prompt for PerplexityRelatedMarketsBot


def perp_related_markets_binary_prompt(question, research):
    return clean_indents(
        f"""
        You are a professional forecaster interviewing for a job.

        Your interview question is:
        {question.question_text}

        Question background:
        {question.background_info}

        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
        {question.resolution_criteria}

        {question.fine_print}

        Your research assistant found web search results and related markets info:
        {research}

        IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

        Today is {datetime.now().strftime('%Y-%m-%d')}.

        Before answering you write:
        (a) The time left until the outcome to the question is known.
        (b) The status quo outcome if nothing changed.
        (c) A brief description of a scenario that results in a No outcome.
        (d) A brief description of a scenario that results in a Yes outcome.
        (e) Write out the question again, and acknolwedge that if the No outcome is more likely your answers should be closer to 0 and if the Yes outcome is more likely your answers should be closer to 100.

        You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

        {PROBABILITY_FINAL_ANSWER_LINE}
        """
    )

# Multiple choice prompt for PerplexityRelatedMarketsBot


def perp_related_markets_mc_prompt(question, research):
    return clean_indents(
        f"""
        You are a professional forecaster interviewing for a job.

        Your interview question is:
        {question.question_text}

        The options are: {question.options}

        Background:
        {question.background_info}

        {question.resolution_criteria}

        {question.fine_print}

        Your research assistant found web search results and related markets info:
        {research}

        IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

        Today is {datetime.now().strftime('%Y-%m-%d')}.

        Before answering you write:
        (a) The time left until the outcome to the question is known.
        (b) The status quo outcome if nothing changed.
        (c) A description of an scenario that results in an unexpected outcome.

        You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

        The last thing you write is your final probabilities for the N options in this order {question.options} as:
        Option_A: Probability_A
        Option_B: Probability_B
        ...
        Option_N: Probability_N
        """
    )

# Numeric prompt for PerplexityRelatedMarketsBot


def perp_related_markets_numeric_prompt(question, research):
    lower = getattr(question, 'lower_bound', 0)
    upper = getattr(question, 'upper_bound', 100)
    lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
        question, 'lower_bound') else ""
    upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
        question, 'upper_bound') else ""
    return clean_indents(
        f"""
        You are a professional forecaster interviewing for a job.

        Your interview question is:
        {question.question_text}

        Background:
        {question.background_info}

        {question.resolution_criteria}

        {question.fine_print}

        Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

        Your research assistant found web search results and related markets info:
        {research}

        IMPORTANT: The research above was gathered by junior research assistants. It is always possible that some of it is out of date, misleading, or tangential to the question. Use only the parts that seem the most up to date and directly relevant to the question. If any information seems older, less reliable, or only tangentially related, you should ignore it when making your forecast.

        Today is {datetime.now().strftime('%Y-%m-%d')}.

        {lower_bound_message}
        {upper_bound_message}

        Formatting Instructions:
        - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
        - Never use scientific notation.
        - Always start with a smaller number (more negative if negative) and then increase from there

        Before answering you write:
        (a) The time left until the outcome to the question is known.
        (b) The outcome if nothing changed.
        (c) The outcome if the current trend continued.
        (d) The expectations of experts and markets.
        (e) A brief description of an unexpected scenario that results in a low outcome.
        (f) A brief description of an unexpected scenario that results in a high outcome.

        You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

        The last thing you write is your final answer as:
        "
        Percentile 10: XX
        Percentile 20: XX
        Percentile 40: XX
        Percentile 60: XX
        Percentile 80: XX
        Percentile 90: XX
        "
        """
    )


def nathan_v1_binary_prompt(question, research):
    return clean_indents(
        f"""
        You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.

        You are forecasting the following question:
        {question.question_text}

        Question background:
        {question.background_info}

        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
        {question.resolution_criteria}

        {question.fine_print}

        Here is relevant research:
        {research}

        Please do the following 4 things in order:
        1. Write out a quick summary of the situation as we know it, from the research. This research comes from an LLM query and may be mistaken. Please note inconsistencies.
        2. Write out 3 - 5 sentence-length scenarios based on the question, including at least one where the question would resolve no and one where it would resolve yes. The scenarios must be moderately different. If they aren't, just use yes and no.
        3. Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
        4. Consider all these scenarios and then give an overall probability as [number]%
        {PROBABILITY_FINAL_ANSWER_LINE}
        """
    )


def nathan_v1_mc_prompt(question, research):
    return clean_indents(
        f"""
        You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.

        You are forecasting the following question:
        {question.question_text}

        The options are: {question.options}

        Background:
        {question.background_info}

        {question.resolution_criteria}

        {question.fine_print}

        Here is relevant research:
        {research}

        Please do the following 4 things in order:
        1. Write out a quick summary of the situation as we know it, from the research. This research comes from an LLM query and may be mistaken. Please note inconsistencies.
        2. Write out 3 - 5 sentence-length scenarios based on the question, including at least one for each of the most likely options. The scenarios must be moderately different. If they aren't, just use the main options.
        3. Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
        4. Consider all these scenarios and then give an overall probability for each option as:
        Option_A: Probability_A
        Option_B: Probability_B
        ...
        Option_N: Probability_N
        {PROBABILITY_FINAL_ANSWER_LINE}
        """
    )


def nathan_v1_numeric_prompt(question, research):
    lower = getattr(question, 'lower_bound', 0)
    upper = getattr(question, 'upper_bound', 100)
    lower_bound_message = f"The outcome can not be lower than {lower}." if hasattr(
        question, 'lower_bound') else ""
    upper_bound_message = f"The outcome can not be higher than {upper}." if hasattr(
        question, 'upper_bound') else ""
    return clean_indents(
        f"""
        You are a forecasting bot. Please think to the best of your ability, like a Good Judgement Project Superforecaster. Please be straightforward and accurate even if it results in answers that are awkward.

        You are forecasting the following question:
        {question.question_text}

        Background:
        {question.background_info}

        {question.resolution_criteria}

        {question.fine_print}

        Units for answer: {getattr(question, 'unit_of_measure', 'Not stated (please infer this)')}

        Here is relevant research:
        {research}

        {lower_bound_message}
        {upper_bound_message}

        Please do the following 4 things in order:
        1. Write out a quick summary of the situation as we know it, from the research. This research comes from an LLM query and may be mistaken. Please note inconsistencies.
        2. Write out 3 - 5 sentence-length scenarios based on the question, including at least one for a low outcome and one for a high outcome. The scenarios must be moderately different. If they aren't, just use low and high.
        3. Write out a paragraph about each scenario, describing how it might arise from the information. Then attempt to assign a base rate to it - how long has the world been possible for this scenario to occur, how many times has it done so in that time?
        4. Consider all these scenarios and then give an overall best estimate as a single number, and optionally a 90% confidence interval as:
        Best estimate: XX
        90% confidence interval: [XX, YY]
        {PROBABILITY_FINAL_ANSWER_LINE}
        """
    )


PROBABILITY_FINAL_ANSWER_LINE = (
    "Before giving your final answer, rewrite the question as a probability statement (e.g., "
    "\"What is the probability that [event] will happen?\"), making sure it matches the outcome you are forecasting. "
    "Then, the last thing you write is your final answer as: \"Probability: ZZ%\", 0-100 (no decimals, do not include a space between the number and the % sign)."
)
