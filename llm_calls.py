import asyncio
from openai import AsyncOpenAI

CONCURRENT_REQUESTS_LIMIT = 5
llm_rate_limiter = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


async def call_openAI(prompt: str, model: str = "gpt-4.1", temperature: float = 0.3) -> str:
    """
    Makes a completion request to OpenAI's API with concurrent request limiting.
    """

    # Remove the base_url parameter to call the OpenAI API directly
    # Also checkout the package 'litellm' for one function that can call any model from any provider
    # Email ben@metaculus.com if you need credit for the Metaculus OpenAI/Anthropic proxy
    client = AsyncOpenAI(
        # base_url="https://llm-proxy.metaculus.com/proxy/openai/v1",
        # default_headers={
        #     "Content-Type": "application/json",
        #     "Authorization": f"Token {METACULUS_TOKEN}",
        # },
        # api_key="Fake API Key since openai requires this not to be NONE. This isn't used",
        max_retries=2,
    )

    async with llm_rate_limiter:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False,
        )
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("No answer returned from OpenAI")
        return answer


async def create_rationale_summary(rationales: list[str], question_title: str, question_type: str, final_prediction: str, source_urls: list[str] = None) -> str:
    """
    Create a consolidated summary of multiple rationales for a forecasting question.
    
    Args:
        rationales: List of individual rationales from multiple runs
        question_title: The forecasting question title
        question_type: Type of question (binary, numeric, multiple_choice)
        final_prediction: The final aggregated prediction
        source_urls: List of source URLs used in research (optional)
    
    Returns:
        A consolidated summary highlighting key insights, contradictions, and justification
    """
    if len(rationales) <= 1:
        return ""  # No summary needed for single rationale
    
    # Import the prompt template
    from prompts import RATIONALE_SUMMARY_PROMPT_TEMPLATE
    
    # Combine all rationales for analysis
    combined_rationales = "\n\n---RATIONALE SEPARATOR---\n\n".join([f"Rationale {i+1}:\n{rationale}" for i, rationale in enumerate(rationales)])
    
    prompt = RATIONALE_SUMMARY_PROMPT_TEMPLATE.format(
        question_title=question_title,
        question_type=question_type,
        final_prediction=final_prediction,
        num_rationales=len(rationales),
        combined_rationales=combined_rationales
    )

    try:
        summary = await call_openAI(prompt, model="gpt-4.1-mini", temperature=0.2)
        
        # Add source URLs section if available
        if source_urls:
            sources_section = f"\n\n## Sources Used\nThe following sources were used in this analysis:\n"
            for i, url in enumerate(source_urls, 1):
                sources_section += f"{i}. {url}\n"
            summary += sources_section
        
        return summary.strip()
    except Exception as e:
        print(f"Error creating rationale summary: {str(e)}")
        return "Failed to generate consolidated summary."