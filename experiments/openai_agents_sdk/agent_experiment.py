import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
import os
import sys
import pathlib
from pydantic import BaseModel
from forecasting_tools import (
    GeneralLlm,
)


# Add project root to sys.path
project_root = str(pathlib.Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent_sdk import MetaculusAsyncOpenAI


class WeatherReport(BaseModel):
    city: str
    forecast: str


@function_tool
async def web_search(query: str) -> str:
    """Searches the web for information about a given query."""
    model = GeneralLlm(
        model="metaculus/gpt-4o-search-preview",
        temperature=None,
    )
    response = await model.invoke(query)
    return response


async def run_agent_experiment():
    print("Running OpenAI Agents SDK - Tool Use Example")

    # Ensure METACULUS_TOKEN is set
    if not os.getenv("METACULUS_TOKEN"):
        print("Error: METACULUS_TOKEN environment variable is not set.")
        print("Please set it before running the experiment:")
        print("  export METACULUS_TOKEN='your_metaculus_token_here'")
        return

    try:
        # OpenAI Agents SDK "Hello world" example
        agent = Agent(
            name="Assistant",
            instructions="You are a helpful assistant. Use the available tools to find the weather and then respond with a WeatherReport.",
            model=OpenAIChatCompletionsModel(
                model="o4-mini", openai_client=MetaculusAsyncOpenAI()
            ),
            tools=[web_search],
            output_type=WeatherReport,
        )

        print("Initializing agent runner...")

        print("Invoking agent asynchronously...")
        result = await Runner.run(agent, "What is the weather in London?")

        print("Agent run completed.")
        if result and hasattr(result, "final_output"):
            print("\nAgent's Final Output:")
            print(result.final_output)
        else:
            print("No final output received or result format is different.")

    except ImportError:
        print("Error: The 'openai-agents' library is not installed.")
        print("Please install it by running: pip install openai-agents")
        return
    except Exception as e:
        print(f"An error occurred while running the agent experiment: {e}")
        return

    print("\nOpenAI Agents SDK experiment finished.")


if __name__ == "__main__":
    # Then run this script:
    # python experiments/openai_agents_sdk/agent_experiment.py

    asyncio.run(run_agent_experiment())
