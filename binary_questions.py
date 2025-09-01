import re
import datetime
from prompts import BINARY_PROMPT_TEMPLATE
from llm_calls import call_openAI, create_rationale_summary


def extract_probability_from_response_as_percentage_not_decimal(
    forecast_text: str,
) -> float:
    matches = re.findall(r"(\d+)%", forecast_text)
    if matches:
        # Return the last number found before a '%'
        number = int(matches[-1])
        number = min(99, max(1, number))  # clamp the number between 1 and 99
        return number
    else:
        raise ValueError(f"Could not extract prediction from response: {forecast_text}")


async def get_binary_gpt_prediction(
    question_details: dict, num_runs: int, run_research_func
) -> tuple[float, str]:

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]

    summary_report, source_urls = await run_research_func(title)

    content = BINARY_PROMPT_TEMPLATE.format(
        title=title,
        today=today,
        background=background,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        summary_report=summary_report,
    )

    async def get_rationale_and_probability(content: str) -> tuple[float, str]:
        rationale = await call_openAI(content)

        probability = extract_probability_from_response_as_percentage_not_decimal(
            rationale
        )
        comment = (
            f"Extracted Probability: {probability}%\n\nGPT's Answer: "
            f"{rationale}\n\n\n"
        )
        return probability, comment

    import asyncio
    import numpy as np

    probability_and_comment_pairs = await asyncio.gather(
        *[get_rationale_and_probability(content) for _ in range(num_runs)]
    )
    comments = [pair[1] for pair in probability_and_comment_pairs]
    final_comment_sections = [
        f"## Rationale {i+1}\n{comment}" for i, comment in enumerate(comments)
    ]
    probabilities = [pair[0] for pair in probability_and_comment_pairs]
    median_probability = float(np.median(probabilities)) / 100

    # Create consolidated summary if multiple runs
    consolidated_summary = ""
    if num_runs > 1:
        rationales = [pair[1].split("GPT's Answer: ", 1)[1] if "GPT's Answer: " in pair[1] else pair[1] for pair in probability_and_comment_pairs]
        consolidated_summary = await create_rationale_summary(
            rationales=rationales,
            question_title=title,
            question_type="binary",
            final_prediction=f"{median_probability:.2%}",
            source_urls=source_urls
        )

    # Build final comment with consolidated summary if available
    final_comment_parts = [f"Median Probability: {median_probability}"]
    
    if consolidated_summary:
        final_comment_parts.append(f"\n## Consolidated Analysis\n{consolidated_summary}")
    
    final_comment_parts.append("\n" + "\n\n".join(final_comment_sections))
    
    final_comment = "\n\n".join(final_comment_parts)
    return median_probability, final_comment