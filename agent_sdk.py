from agents import AsyncOpenAI
import os


class MetaculusAsyncOpenAI(AsyncOpenAI):
    def __init__(self, *args, **kwargs):
        # Set default base_url and api_key if not provided
        kwargs.setdefault("base_url", "https://llm-proxy.metaculus.com/proxy/openai/v1")
        kwargs.setdefault("api_key", os.getenv("METACULUS_TOKEN"))
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return super().__getattr__(name)

    @property
    def auth_headers(self):
        return {"Authorization": f"Token {os.getenv('METACULUS_TOKEN')}"}
