import sys
import pathlib
import os
import asyncio
from typing import List, Dict, Any, Optional, Callable

from pydantic import BaseModel, Field

# To import from root agent_sdk.py and forecasting_tools.py
try:
    from agent_sdk import MetaculusAsyncOpenAI
    from gemini_api import gemini_web_search
except ImportError:
    project_root = str(pathlib.Path(__file__).resolve().parents[3])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent_sdk import MetaculusAsyncOpenAI
    from gemini_api import gemini_web_search

from forecasting_tools import GeneralLlm
from agents import (
    function_tool,
    AgentOutputSchema,
)  # Assuming AgentOutputSchema is still used, if not, can be removed.


# Global constant for the web search LLM model
WEB_SEARCH_LLM_MODEL = "metaculus/gpt-4o-search-preview"


@function_tool
async def web_search(query: str) -> str:
    """Searches the web for information about a given query."""
    print(f"\n🔎 Performing web search for: '{query}'")

    gemini_api_key_1 = os.environ.get("GEMINI_API_KEY_1")
    gemini_api_key_2 = os.environ.get("GEMINI_API_KEY_2")

    # Try Gemini with the first key
    if gemini_api_key_1:
        try:
            print("Trying web search with Gemini API Key 1...")
            response = await asyncio.to_thread(
                gemini_web_search, query, gemini_api_key_1
            )
            print(f"\n🔍 Search result: {response}")
            return response
        except Exception as e:
            print(f"Gemini search with key 1 failed: {e}")

    # Try Gemini with the second key
    if gemini_api_key_2:
        try:
            print("Trying web search with Gemini API Key 2...")
            response = await asyncio.to_thread(
                gemini_web_search, query, gemini_api_key_2
            )
            print(f"\n🔍 Search result: {response}")
            return response
        except Exception as e:
            print(f"Gemini search with key 2 failed: {e}")

    # Fallback to the original implementation
    print("Falling back to GeneralLlm for web search.")
    model = GeneralLlm(model=WEB_SEARCH_LLM_MODEL, temperature=None)
    response = await model.invoke(query)
    print(f"\n🔍 Search result: {response}")
    return response


# --- Pydantic Models for BN and Data Structures (Unchanged from original) ---


class Node(BaseModel):
    name: str
    description: str
    states: Dict[str, str]
    parents: List[str] = Field(default_factory=list)
    cpt: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="The Conditional Probability Table."
    )


class BNStructure(BaseModel):
    topic: str  # topic is also in BasePipelineData, ensure consistency or decide if it's needed here
    explanation: str
    nodes: Dict[str, Node] = Field(default_factory=dict)
    target_node_name: str = Field(
        description="The name of the node that directly answers the forecasting question."
    )


class ContextReport(BaseModel):
    verified_facts: List[str] = Field(
        description="A list of key facts and definitions that have been verified against the latest information."
    )
    summary: str = Field(
        description="A brief summary of the overall context based on the verified facts."
    )


class QualitativeCpt(BaseModel):
    cpt_qualitative_estimates: Dict[str, Dict[str, int]] = Field(
        description="A dictionary where keys are string representations of parent state tuples, "
        "and values are dictionaries mapping child states to their relative likelihood scores."
    )
    justification: str = Field(
        description="A detailed explanation of the reasoning used to arrive at the estimates."
    )


# --- Agent Configuration Models (Unchanged from original) ---


class ModelConfig(BaseModel):
    model_name: str
    openai_client: Optional[Any] = None


class AgentConfig(BaseModel):
    agent_class: Callable[..., Any]
    model_settings: ModelConfig
    tools: List[Any] = Field(default_factory=list)
    output_schema: Optional[Any] = None
    instructions_template: str


# --- New Step-Specific Pipeline Data Models ---


class BasePipelineData(BaseModel):
    """
    Base model for pipeline data, ensuring topic and error message are always available.
    """

    topic: str
    error_message: Optional[str] = None


class ContextStepOutput(BasePipelineData):
    """
    Data output by the ContextStep.
    """

    current_date: str
    context_report: ContextReport
    context_facts_str: str
    context_summary: str


class ArchitectStepInput(ContextStepOutput):
    """
    Input for the ArchitectStep, taking all data from ContextStepOutput.
    """

    pass


class ArchitectStepOutput(ArchitectStepInput):
    """
    Data output by the ArchitectStep, including the BN structure.
    The bn_structure here is expected to have its structure defined but CPTs might be empty.
    """

    bn_structure: BNStructure


class CPTGenerationStepInput(ArchitectStepOutput):
    """
    Input for the CPTGenerationStep, taking all data from ArchitectStepOutput.
    """

    pass


class CPTGenerationStepOutput(CPTGenerationStepInput):
    """
    Data output by the CPTGenerationStep.
    The bn_structure field (inherited from ArchitectStepOutput) is now expected
    to be populated with CPTs.
    """

    # No new fields, but bn_structure within this model is understood to have CPTs.
    # If a distinct field name is strongly preferred in the future, it could be:
    # bn_structure_with_cpts: BNStructure
    pass


class FinalCalculationStepInput(CPTGenerationStepOutput):
    """
    Input for the FinalCalculationStep.
    """

    pass


class FinalCalculationStepOutput(FinalCalculationStepInput):
    """
    Data output by the FinalCalculationStep.
    """

    final_probabilities: Optional[Dict[str, float]] = None
    final_probabilities_str: Optional[str] = None


class ForecasterStepInput(FinalCalculationStepOutput):
    """
    Input for the ForecasterStep.
    """

    pass


class ForecasterStepOutput(ForecasterStepInput):
    """
    Data output by the ForecasterStep.
    """

    forecaster_agent_output: Optional[str] = None


# --- DEPRECATED Main Pipeline Data Structure ---
# DEPRECATED: This will be removed in favor of step-specific data models.
class PipelineData(BaseModel):
    """
    A Pydantic model to hold and pass all data between pipeline steps.
    This ensures type safety and clear data contracts.
    """

    # Initial inputs
    topic: str

    # Data generated by ContextStep
    current_date: Optional[str] = None
    context_report: Optional[ContextReport] = None
    context_facts_str: Optional[str] = None
    context_summary: Optional[str] = None

    # Data generated by ArchitectStep
    bn_structure: Optional[BNStructure] = (
        None  # Will be populated with CPTs by CPTGenerationStep
    )

    # Data generated by FinalCalculationStep
    final_probabilities: Optional[Dict[str, float]] = None
    final_probabilities_str: Optional[str] = None  # String representation or error

    # Data generated by ForecasterStep
    forecaster_agent_output: Optional[str] = None

    # General fields
    error_message: Optional[str] = (
        None  # To accumulate errors if pipeline continues on some failures
    )

    # Allows other fields to be added if necessary, for extensibility, though explicit fields are preferred.
    # class Config:
    #     extra = "allow"
