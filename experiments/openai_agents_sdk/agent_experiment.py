import asyncio
import logging
import datetime
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    AgentOutputSchema,
    trace,
    ModelSettings,
)
import os
import sys
import pathlib
from pydantic import BaseModel, Field
from forecasting_tools import (
    GeneralLlm,
)
from gemini_api import gemini_web_search


# Add project root to sys.path
project_root = str(pathlib.Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Create logs directory if it doesn't exist
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_filename = f"logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("openai.agents")

from agent_sdk import MetaculusAsyncOpenAI


class PerspectiveAgent(BaseModel):
    name: str = Field(description="The name of the perspective agent.")
    instructions: str = Field(
        description="Instructions for how this agent should analyze its perspective."
    )


class DiscussionReport(BaseModel):
    perspective_web_search_evidence: list[str] = Field(
        description="A list of web search evidence for each perspective."
    )
    perspective_ratings: dict[str, int] = Field(
        description="A dictionary of perspective ratings for each option."
    )
    discussion_summary: str = Field(description="A summary of the discussion.")
    discussion_points: list[str] = Field(
        description="A list of key points from the discussion."
    )


class ContextReport(BaseModel):
    key_terms: list[str] = Field(
        description="A list of key terms and concepts about the question."
    )
    definition_of_key_terms: dict[str, str] = Field(
        description="A dictionary of definitions for each key term."
    )
    evidence_for_key_terms: list[str] = Field(
        description="The web search evidence used to support the key terms. Each term should be supported by a web search result."
    )
    summary: str = Field(description="A summary of the context.")


class BaseCaseReport(BaseModel):
    base_case_ratings: dict[str, int] = Field(
        description="A dictionary of base case ratings for each option."
    )
    base_case_summary: str = Field(description="A summary of the base case.")


@function_tool
async def web_search(query: str) -> str:
    """Searches the web for information about a given query."""
    logger.info("--- WEB SEARCH ---")
    logger.info(f"🔎 Performing web search for: '{query}'")

    gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
    gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")

    # Try Gemini with the first key
    if gemini_api_key_1:
        try:
            logger.info("Trying web search with Gemini API Key 1...")
            response = await asyncio.to_thread(
                gemini_web_search, query, gemini_api_key_1
            )
            logger.info(f"🔍 Search result: {response}")
            logger.info("--- END WEB SEARCH ---")
            return response
        except Exception as e:
            logger.warning(f"Gemini search with key 1 failed: {e}")

    # Try Gemini with the second key
    if gemini_api_key_2:
        try:
            logger.info("Trying web search with Gemini API Key 2...")
            response = await asyncio.to_thread(
                gemini_web_search, query, gemini_api_key_2
            )
            logger.info(f"🔍 Search result: {response}")
            logger.info("--- END WEB SEARCH ---")
            return response
        except Exception as e:
            logger.warning(f"Gemini search with key 2 failed: {e}")

    # Fallback to the original implementation
    logger.info("Falling back to GeneralLlm for web search.")
    model = GeneralLlm(model="metaculus/gpt-4o-search-preview", temperature=None)
    response = await model.invoke(query)
    logger.info(f"🔍 Search result: {response}")
    logger.info("--- END WEB SEARCH ---")
    return response


base_case_agent = Agent(
    name="Base Case",
    instructions=f"""You are a base case agent. Your goal is to establish a foundational viewpoint for a given question. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.
You should possess a healthy dose of curiosity, skepticism, and embrace diverse perspectives in your analysis.

For each possible option in the question, you must use the web_search tool to find any relevant historical parallels or precedents.
1. Find any relevant historical parallels or precedents.
2. Based on these parallels, determine the starting likeliness for each option.
3. Provide a base case rating for each option on a scale of 0 to 4:
   - 0: Very Unlikely
   - 1: Unlikely
   - 2: Neutral
   - 3: Likely
   - 4: Very Likely

Present your analysis and ratings clearly.""",
    model=OpenAIChatCompletionsModel(
        model="o4-mini", openai_client=MetaculusAsyncOpenAI()
    ),
    tools=[web_search],
    output_type=AgentOutputSchema(BaseCaseReport, strict_json_schema=False),
)


@function_tool
async def perspectives_factory(prompt: str) -> str:
    """Creates and runs a panel of perspective agents in parallel and returns their combined analysis as a string."""
    logger.info("--- Running Perspectives Factory ---")
    perspectives_factory_agent = Agent(
        name="Perspectives Factory",
        instructions=f"""You are a perspectives factory agent that analyzes topics and creates a diverse set of perspectives for an analysis. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.
You should possess a healthy dose of curiosity, skepticism, and embrace diverse perspectives in your analysis.

For any given topic, you will:
1. Analyze the key aspects and dimensions of the topic
2. Identify 5 distinct and relevant perspectives that could provide valuable insights
3. For each perspective, create a PerspectiveAgent with:
- A clear name describing the perspective
- Detailed instructions on how to analyze from that viewpoint
- The same base model configuration

For example, for a geopolitical topic you might create:
- A Historical Precedent Analyst who examines past patterns and analogous situations
- An Economic Pragmatist who focuses on financial incentives and economic realities
- An International Relations Strategist who analyzes geopolitical dynamics and alliances
- A Regional Security Expert who evaluates stability and conflict implications
- A Cultural Integration Specialist who considers societal and cross-border dynamics

Return the 5 PerspectiveAgents as a structured list, each with their unique analytical focus while maintaining intellectual rigor.""",
        model=OpenAIChatCompletionsModel(
            model="o4-mini", openai_client=MetaculusAsyncOpenAI()
        ),
        output_type=list[PerspectiveAgent],
    )

    logger.info("Generating perspectives...")
    result = await Runner.run(perspectives_factory_agent, prompt)
    perspectives = result.final_output
    if not perspectives:
        logger.warning("Perspectives factory agent did not return any perspectives.")
        return "No perspectives were generated."

    logger.info(
        f"Generated {len(perspectives)} perspectives. Creating and running agents in parallel."
    )

    base_instructions = f"""You are a perspective agent that analyzes a question from a given perspective. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.
You should possess a healthy dose of curiosity, skepticism, and embrace diverse perspectives in your analysis. 
At every step of the way, think to yourself: "What am I missing? Is there an angle I have not considered? What more information do I need?"
Your knowledge is definitely outdated. You MUST NOT rely on it. Your entire process must be driven by `web_search` to find the absolute latest information. 
Once you get a web_search result, you must use it to update your analysis. Then continue to ask yourself, does the web_search answer my question? 
If not, what more information do I need?
You must be skeptical of your own analysis. If you are in doubt, use `web_search` to fact check your analysis.
You must be skeptical of the analysis of the base case agent. If you are in doubt, use `web_search` to fact check their analysis.
"""

    parallel_agents = []
    for p in perspectives:
        agent = Agent(
            name=p.name,
            instructions=base_instructions + p.instructions,
            model=OpenAIChatCompletionsModel(
                model="o4-mini", openai_client=MetaculusAsyncOpenAI()
            ),
            tools=[web_search],
        )
        parallel_agents.append(agent)

    async def run_agent(agent: Agent, prompt: str):
        logger.info(f"Running agent: {agent.name}")
        result = await Runner.run(agent, prompt)
        logger.info(f"Agent {agent.name} finished.")
        return result

    tasks = [run_agent(agent, prompt) for agent in parallel_agents]
    responses = await asyncio.gather(*tasks)

    labeled_summaries = []
    for resp in responses:
        if resp and hasattr(resp, "final_output") and resp.final_output:
            labeled_summaries.append(
                f"### Perspective: {resp.last_agent.name}\n{resp.final_output}"
            )

    collected_summaries = "\n\n".join(labeled_summaries)
    logger.info("--- Perspectives Factory Finished ---")
    return collected_summaries


perspectives_panel_agent = Agent(
    name="Perspectives Panel",
    instructions=f"""You are a perspectives panel agent that analyzes a collection of viewpoints on a question to reach a consensus. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.
You should possess a healthy dose of curiosity, skepticism, and embrace diverse perspectives in your analysis.

Your role is to act as a master analyst, synthesizing a variety of perspectives.
Your task is to:
1. First, use the `perspectives_factory` tool to generate and run a diverse set of perspective agents. This tool will return a string containing the analysis from each perspective.
2. Carefully review the collected analyses. For each perspective, identify key arguments and supporting evidence.
3. Use the `web_search` tool to verify any claims, find additional supporting or conflicting evidence, and fill any gaps in the analysis. Be skeptical.
4. Synthesize the discussion by providing:
   - A summary of each perspective's key arguments and evidence (`discussion_summary` and `discussion_points`).
   - The strongest points made by each perspective.
   - Areas of agreement and disagreement between perspectives.
   - Web search evidence you found (`perspective_web_search_evidence`).
   - A final consensus rating for each option of the topic (`perspective_ratings`) on a scale of 0 to 4:
     - 0: Very Unlikely
     - 1: Unlikely  
     - 2: Neutral
     - 3: Likely
     - 4: Very Likely
5. If no consensus can be reached, explain why and provide the final ratings from each perspective if possible.
""",
    model=OpenAIChatCompletionsModel(
        model="o4-mini", openai_client=MetaculusAsyncOpenAI()
    ),
    output_type=AgentOutputSchema(DiscussionReport, strict_json_schema=False),
    tools=[
        perspectives_factory,
        web_search,
    ],
)


context_agent = Agent(
    name="Context",
    instructions=f"""You are an expert fact-checker and intelligence analyst. Your primary goal is to establish clear, evidence-based definitions for all key terms in a given forecasting question. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.

Your own knowledge is definitely outdated. You MUST NOT rely on it. Your entire process must be driven by `web_search` to find the absolute latest information. DO NOT MAKE ANY ASSUMPTIONS BASED ON YOUR KNOWLEDGE.

Your process:
1. **Identify Key Terms**: Break down the question into its key terms that need definition, including:
   - Named entities (e.g., people, organizations, places)
   - Technical concepts
   - Important phrases
   - Any potentially ambiguous terms

2. **Define Each Term**: For each identified term:
   - Use `web_search` to find authoritative definitions
   - Search for the most current meaning and usage
   - Verify the definition from multiple sources
   - For time-sensitive terms (e.g., someone's current role), explicitly search for current status
   Example queries:
   - 'What is [term] official definition'
   - '[term] meaning 2024'
   - '[person name] current position as of {datetime.date.today().strftime('%Y-%m-%d')}'

3. **Gather Evidence**: For each term:
   - Use `web_search` to find supporting evidence for the definition
   - Look for recent examples of the term's usage
   - Find authoritative sources discussing the term
   - Document the search results that best support your definition

4. **Output Format**:
   - List all identified key terms in the `key_terms` field
   - Provide clear, precise definitions in the `definition_of_key_terms` field
   - Include supporting web search evidence in the `evidence_for_key_terms` field
   - Write a brief `summary` that ties together how these terms relate to the question

Remember: Every single definition must be supported by current web search evidence. Never rely on prior knowledge.""",
    model=OpenAIChatCompletionsModel(
        model="o4-mini",
        openai_client=MetaculusAsyncOpenAI(),
    ),
    tools=[web_search],
    output_type=AgentOutputSchema(ContextReport, strict_json_schema=False),
    model_settings=ModelSettings(
        tool_choice="required",
    ),
)


forecaster_agent = Agent(
    name="Forecaster",
    instructions=f"""You are a forecaster agent that analyzes questions with multiple options to produce probability estimates. The current date is {datetime.date.today().strftime('%Y-%m-%d')}.
You should possess a healthy dose of curiosity, skepticism, and embrace diverse perspectives in your analysis. Follow these steps:

1. First, give the question verbatim to the the context tool to gather fundamental facts and ground truths about the situation and options.

2. Then, give the question verbatim AND the context report from the context tool to the base_case tool to establish base case probabilities for each option by using the base_case tool to find historical parallels and relevant precedents. Make sure to give the full context report to the base_case tool.

3. Next, give the question verbatim AND the context report AND base case report to the perspectives_panel tool to get the opinions of a variety of perspectives on the question. Make sure to give the full context report and base case report to the perspectives_panel tool.

4. Finally, synthesize all inputs into a comprehensive analysis that:
   - Summarizes the key context and facts gathered.
   - Explains the base case probabilities and historical parallels.
   - Captures the key points and the final rating from the perspectives panel discussion.
   - Provides final probability estimates for each option with detailed reasoning.

Always strive for rigorous, well-reasoned analysis grounded in facts while acknowledging uncertainties.""",
    model=OpenAIChatCompletionsModel(
        model="o4-mini", openai_client=MetaculusAsyncOpenAI()
    ),
    tools=[
        context_agent.as_tool(
            tool_name="context",
            tool_description="Provides context for the other agents.",
        ),
        base_case_agent.as_tool(
            tool_name="base_case",
            tool_description="Provides a base case based on the context by the context tool.",
        ),
        perspectives_panel_agent.as_tool(
            tool_name="perspectives_panel",
            tool_description="Facilitates a Socratic discussion among a list of perspective agents to examine the question from multiple angles and reach a consensus view. Takes a list of agents as input.",
        ),
    ],
)


async def run_agent_experiment():
    logger.info("--- Starting OpenAI Agents SDK Experiment ---")

    # Ensure METACULUS_TOKEN is set
    if not os.getenv("METACULUS_TOKEN"):
        logger.error("Error: METACULUS_TOKEN environment variable is not set.")
        logger.error("Please set it before running the experiment:")
        logger.error("  export METACULUS_TOKEN='your_metaculus_token_here'")
        return

    try:
        with trace("forecaster_agent"):
            logger.info("--- Running Forecaster Agent ---")
            question = "Will Sudan ratify AfCFTA (the pan-African free trade agreement) before July 1, 2025? Yes or No?"
            logger.info(f"Question: {question}")
            # OpenAI Agents SDK "Hello world" example
            result = await Runner.run(forecaster_agent, question)

            logger.info("--- Agent run completed ---")
            if result and hasattr(result, "final_output"):
                logger.info("--- Agent's Final Output ---")
                logger.info(result.final_output)
            else:
                logger.warning(
                    "No final output received or result format is different."
                )
                logger.info(f"Full result: {result}")

    except ImportError:
        logger.error("Error: The 'openai-agents' library is not installed.")
        logger.error("Please install it by running: pip install openai-agents")
        return
    except Exception as e:
        logger.exception(f"An error occurred while running the agent experiment: {e}")
        return

    logger.info("--- OpenAI Agents SDK experiment finished ---")


if __name__ == "__main__":
    # Then run this script:
    # python experiments/openai_agents_sdk/agent_experiment.py

    asyncio.run(run_agent_experiment())
