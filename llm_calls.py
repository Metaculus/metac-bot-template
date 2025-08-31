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