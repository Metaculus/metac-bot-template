import re
import datetime
from prompts import MULTIPLE_CHOICE_PROMPT_TEMPLATE
from llm_calls import call_openAI, create_rationale_summary


def extract_option_probabilities_from_response(forecast_text: str, options) -> float:

    # Helper function that returns a list of tuples with numbers for all lines with Percentile
    def extract_option_probabilities(text):

        # Number extraction pattern
        number_pattern = r"-?\d+(?:,\d{3})*(?:\.\d+)?"

        results = []

        # Iterate through each line in the text
        for line in text.split("\n"):
            # Extract all numbers from the line
            numbers = re.findall(number_pattern, line)
            numbers_no_commas = [num.replace(",", "") for num in numbers]
            # Convert strings to float or int
            numbers = [
                float(num) if "." in num else int(num) for num in numbers_no_commas
            ]
            # Add the tuple of numbers to results
            if len(numbers) >= 1:
                last_number = numbers[-1]
                results.append(last_number)

        return results

    option_probabilities = extract_option_probabilities(forecast_text)

    NUM_OPTIONS = len(options)

    if len(option_probabilities) > 0:
        # return the last NUM_OPTIONS items
        return option_probabilities[-NUM_OPTIONS:]
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


def generate_multiple_choice_forecast(options, option_probabilities) -> dict:
    """
    Returns: dict corresponding to the probabilities of each option.
    """

    # confirm that there is a probability for each option
    if len(options) != len(option_probabilities):
        raise ValueError(
            f"Number of options ({len(options)}) does not match number of probabilities ({len(option_probabilities)})"
        )

    # Ensure we are using decimals
    total_sum = sum(option_probabilities)
    decimal_list = [x / total_sum for x in option_probabilities]

    def normalize_list(float_list):
        # Step 1: Clamp values
        clamped_list = [max(min(x, 0.99), 0.01) for x in float_list]

        # Step 2: Calculate the sum of all elements
        total_sum = sum(clamped_list)

        # Step 3: Normalize the list so that all elements add up to 1
        normalized_list = [x / total_sum for x in clamped_list]

        # Step 4: Adjust for any small floating-point errors
        adjustment = 1.0 - sum(normalized_list)
        normalized_list[-1] += adjustment

        return normalized_list

    normalized_option_probabilities = normalize_list(decimal_list)

    probability_yes_per_category = {}
    for i in range(len(options)):
        probability_yes_per_category[options[i]] = normalized_option_probabilities[i]

    return probability_yes_per_category


async def get_multiple_choice_gpt_prediction(
    question_details: dict,
    num_runs: int,
    run_research_func,
) -> tuple[dict[str, float], str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    options = question_details["options"]

    summary_report, source_urls = await run_research_func(title)

    content = MULTIPLE_CHOICE_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
        options=options,
    )

    async def ask_llm_for_multiple_choice_probabilities(
        content: str,
    ) -> tuple[dict[str, float], str]:
        rationale = await call_openAI(content)

        option_probabilities = extract_option_probabilities_from_response(
            rationale, options
        )

        comment = (
            f"EXTRACTED_PROBABILITIES: {option_probabilities}\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )

        probability_yes_per_category = generate_multiple_choice_forecast(
            options, option_probabilities
        )
        return probability_yes_per_category, comment

    import asyncio

    probability_yes_per_category_and_comment_pairs = await asyncio.gather(
        *[ask_llm_for_multiple_choice_probabilities(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_yes_per_category_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probability_yes_per_category_dicts: list[dict[str, float]] = [
        pair[0] for pair in probability_yes_per_category_and_comment_pairs
    ]
    average_probability_yes_per_category: dict[str, float] = {}
    for option in options:
        probabilities_for_current_option: list[float] = [
            dict[option] for dict in probability_yes_per_category_dicts
        ]
        average_probability_yes_per_category[option] = sum(
            probabilities_for_current_option
        ) / len(probabilities_for_current_option)

    # Create consolidated summary if multiple runs
    consolidated_summary = ""
    if num_runs > 1:
        rationales = [pair[1].split("GPT's Answer: ", 1)[1] if "GPT's Answer: " in pair[1] else pair[1] for pair in probability_yes_per_category_and_comment_pairs]
        # Format prediction as readable percentages
        prediction_text = ", ".join([f"{option}: {prob:.1%}" for option, prob in average_probability_yes_per_category.items()])
        consolidated_summary = await create_rationale_summary(
            rationales=rationales,
            question_title=title,
            question_type="multiple_choice",
            final_prediction=prediction_text,
            source_urls=source_urls
        )

    # Build final comment with consolidated summary if available
    final_comment_parts = [f"Average Probability Yes Per Category: `{average_probability_yes_per_category}`"]
    
    if consolidated_summary:
        final_comment_parts.append(f"\n## Consolidated Analysis\n{consolidated_summary}")
    
    final_comment_parts.append("\n" + "\n\n".join(final_comment_sections))
    
    final_comment = "\n\n".join(final_comment_parts)
    return average_probability_yes_per_category, final_comment