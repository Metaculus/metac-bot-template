import asyncio
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import sys
import pathlib
import itertools  # ### NEW: Imported for iterating over parent states ###

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    AgentOutputSchema,
    exceptions,
)
from pydantic import BaseModel, Field
from forecasting_tools import GeneralLlm

# ==============================================================================
# Boilerplate and Setup (Unchanged from your code)
# ==============================================================================


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


LLM_CALL_COUNT = 0
project_root = str(pathlib.Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from agent_sdk import MetaculusAsyncOpenAI

client = MetaculusAsyncOpenAI()

# Global constant for the web search LLM model
WEB_SEARCH_LLM_MODEL = "metaculus/gpt-4o-search-preview"


@function_tool
async def web_search(query: str) -> str:
    """Searches the web for information about a given query."""
    print(f"\n🔎 Performing web search for: '{query}'")
    model = GeneralLlm(model=WEB_SEARCH_LLM_MODEL, temperature=None)
    response = await model.invoke(query)
    print(f"\n🔍 Search result: {response}")
    return response


original_create = client.chat.completions.create


async def patched_create(*args, **kwargs):
    global LLM_CALL_COUNT
    LLM_CALL_COUNT += 1
    # (Logging details omitted for brevity, but they are kept from your original)
    print(f"\n>> LLM Call {LLM_CALL_COUNT} ({kwargs.get('model')})...")
    response = await original_create(*args, **kwargs)
    print(f"\n>> LLM Call {LLM_CALL_COUNT} ({kwargs.get('model')})... done")
    print(f"\n>> Response: {response}")
    return response


client.chat.completions.create = patched_create


# ==============================================================================
# Agent Configuration Models
# ==============================================================================


class ModelConfig(BaseModel):
    model_name: str
    openai_client: Any = client  # Default to the global client


class AgentConfig(BaseModel):
    agent_class: Any  # Stores the agent class, e.g., agents.Agent
    model_settings: ModelConfig
    tools: List[Any] = Field(default_factory=list)
    output_schema: Optional[Any] = None
    instructions_template: str  # Base instructions template


class PipelineAgentsConfig(BaseModel):
    context_agent: AgentConfig
    architect_agent: AgentConfig
    cpt_estimator_agent: AgentConfig
    forecaster_agent: AgentConfig


# ==============================================================================
# Pydantic Models for BN and Evidence (Unchanged from previous suggestion)
# ==============================================================================


class Node(BaseModel):
    name: str
    description: str
    states: Dict[str, str]
    parents: List[str] = Field(default_factory=list)
    cpt: Dict = Field(
        default_factory=dict, description="The Conditional Probability Table."
    )


class BNStructure(BaseModel):
    topic: str
    explanation: str
    nodes: Dict[str, Node] = Field(default_factory=dict)
    target_node_name: str = Field(
        description="The name of the node that directly answers the forecasting question."
    )


class Argument(BaseModel):
    claim: str
    evidence: str
    source_credibility: str
    strength: float = Field(..., description="Score from 0.0 to 1.0.")


class EvidenceReport(BaseModel):
    base_rate: float
    arguments_for: List[Argument]
    arguments_against: List[Argument]


# ### NEW ### Pydantic model for the context-setting agent's output
class ContextReport(BaseModel):
    """
    A report containing key facts, definitions, and the current context relevant to the forecasting topic.
    This information is verified with the latest available data.
    """

    verified_facts: List[str] = Field(
        description="A list of key facts and definitions that have been verified against the latest information. This should include resolving any time-sensitive information like who holds a political office."
    )
    summary: str = Field(
        description="A brief summary of the overall context based on the verified facts."
    )


# ### NEW ### Pydantic model for the new CPT estimation agent's output
class QualitativeCpt(BaseModel):
    """
    Holds the LLM's qualitative assessment of likelihoods for a node's CPT.
    The keys are parent state combinations, and the values are dictionaries
    mapping child states to a relative, non-normalized likelihood score (e.g., 1-100).
    """

    cpt_qualitative_estimates: Dict[str, Dict[str, int]] = Field(
        description="A dictionary where keys are string representations of parent state tuples, and values are dictionaries mapping child states to their relative likelihood scores."
    )
    justification: str = Field(
        description="A detailed explanation of the reasoning used to arrive at the estimates, citing the evidence and principles applied."
    )


# ==============================================================================
# Deterministic Calculation Function (Unchanged)
# ==============================================================================


def normalize_cpt(qualitative_cpt: QualitativeCpt) -> Dict[str, Dict[str, float]]:
    """
    Converts a QualitativeCpt with likelihood scores into a normalized CPT with probabilities.
    """
    normalized_cpt = {}
    for (
        parent_combo_str,
        child_scores,
    ) in qualitative_cpt.cpt_qualitative_estimates.items():
        total_score = sum(child_scores.values())
        if total_score == 0:
            # Avoid division by zero; distribute probability equally if all scores are 0.
            num_states = len(child_scores)
            normalized_probs = {state: 1.0 / num_states for state in child_scores}
        else:
            normalized_probs = {
                state: score / total_score for state, score in child_scores.items()
            }
        normalized_cpt[parent_combo_str] = normalized_probs
    return normalized_cpt


def calculate_final_probability(
    bn: BNStructure,
    target_node_name: str,
    evidence: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """
    Calculates the final marginal probability distribution for a target node using
    the 'inference by enumeration' algorithm on the Bayesian Network.

    Args:
        bn: The complete BNStructure object with populated CPTs.
        target_node_name: The name of the node for which to calculate the final probability.
        evidence: Optional dictionary of known states for other nodes (e.g., {'Security Threat Level': 'High'}).

    Returns:
        A dictionary mapping each state of the target node to its final calculated probability.
    """
    if evidence is None:
        evidence = {}

    # Determine a valid topological order of nodes for processing.
    nodes_with_in_degree = {n: len(d.parents) for n, d in bn.nodes.items()}
    queue = [n for n, d in nodes_with_in_degree.items() if d == 0]
    topo_order = []
    while queue:
        u = queue.pop(0)
        topo_order.append(u)
        for v_name, v_node in bn.nodes.items():
            if u in v_node.parents:
                nodes_with_in_degree[v_name] -= 1
                if nodes_with_in_degree[v_name] == 0:
                    queue.append(v_name)

    if len(topo_order) != len(bn.nodes):
        print(
            "Warning: Bayesian network may have a cycle. Probability calculation could be incorrect. Using arbitrary node order as a fallback."
        )
        topo_order = list(bn.nodes.keys())

    memo = {}

    def enumerate_all(vars_list, current_evidence):
        vars_tuple = tuple(vars_list)
        evidence_tuple = tuple(sorted(current_evidence.items()))
        memo_key = (vars_tuple, evidence_tuple)
        if memo_key in memo:
            return memo[memo_key]
        if not vars_list:
            return 1.0

        Y = vars_list[0]
        rest_vars = vars_list[1:]
        node_Y = bn.nodes[Y]

        parent_values = [current_evidence.get(p) for p in node_Y.parents]
        if any(v is None for v in parent_values):
            # This can happen if evidence contradicts the network structure.
            # Or if we query a node whose parents are not part of the evidence,
            # which is handled by the recursion.
            pass

        cpt_key = str(tuple(parent_values)) if parent_values else "()"

        if Y in current_evidence:
            prob_y = node_Y.cpt.get(cpt_key, {}).get(current_evidence[Y], 0.0)
            result = prob_y * enumerate_all(rest_vars, current_evidence)
            memo[memo_key] = result
            return result
        else:
            total = 0.0
            for y_state in node_Y.states.keys():
                prob_y = node_Y.cpt.get(cpt_key, {}).get(y_state, 0.0)
                extended_evidence = current_evidence.copy()
                extended_evidence[Y] = y_state
                total += prob_y * enumerate_all(rest_vars, extended_evidence)
            memo[memo_key] = total
            return total

    query_distribution = {}
    for state in bn.nodes[target_node_name].states.keys():
        extended_evidence = evidence.copy()
        extended_evidence[target_node_name] = state
        query_distribution[state] = enumerate_all(topo_order, extended_evidence)

    # Normalize the final distribution
    total = sum(query_distribution.values())
    if total > 0:
        return {state: prob / total for state, prob in query_distribution.items()}
    else:
        num_states = len(bn.nodes[target_node_name].states)
        if num_states > 0:
            return {
                state: 1.0 / num_states for state in bn.nodes[target_node_name].states
            }
        else:
            return {}


def calculate_probability_from_evidence(report: EvidenceReport) -> float:
    """Calculates a final probability from a structured evidence report."""
    score = report.base_rate
    for arg in report.arguments_for:
        score += (1 - score) * arg.strength * 0.5
    for arg in report.arguments_against:
        score -= score * arg.strength * 0.5
    return max(0.01, min(0.99, score))


# ==============================================================================
# Agent Instruction Templates
# ==============================================================================

CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE = (
    "You are an expert fact-checker and intelligence analyst. Your primary goal is to establish a solid, up-to-date factual baseline for a given forecasting question. "
    "Your own knowledge is definitely outdated. You MUST NOT rely on it. Your entire process must be driven by `web_search` to find the absolute latest information.\n\n"
    "Your process:\n"
    "1. **Deconstruct the Question**: Break down the user's question into its core components: entities (e.g., 'Donald Trump', 'NATO'), concepts (e.g., 'formal independence'), and timeframes (e.g., 'June 2025').\n"
    "2. **Verify and Define**: For each component, use `web_search` to find the most current status and definitions. This is especially crucial for time-sensitive facts. For example, if the question involves a politician, you MUST search for their current official title and status *today*. If it involves an organization, verify its current members and stated goals.\n"
    "   - Example Query: 'Who is the current President of the United States as of [today's date]?'\n"
    "   - Example Query: 'What is the official definition of the NATO summit?'\n"
    "3. **Synthesize Facts**: Compile your findings into a list of clear, unambiguous statements in the `verified_facts` field. Each statement should be a critical piece of context.\n"
    "4. **Summarize Context**: Provide a brief `summary` of the overall situation based on your findings.\n\n"
    "Return a single, valid JSON object conforming to the `ContextReport` schema. This report will be the foundational context for all other agents."
)

ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE = (
    "You are an expert in causal inference and systems thinking. Your task is to design a Bayesian Network (BN) for the given topic.\n"
    "Crucially, you MUST consider the current date. Your own knowledge may be outdated. Do not model events whose outcomes are already established historical facts as uncertain variables.\n\n"
    "Your process must be radically evidence-based and rigorous:\n"
    "1.  **Verify Temporal Status**: For any potential factor you consider, you MUST first use `web_search` with a direct question to verify if the event has already occurred and its outcome is known. For example, search for 'who won the 2024 US presidential election'. If an outcome is a known fact, explicitly state it in your reasoning and DO NOT include it as a node in the BN. Build the network only around factors that are genuinely uncertain as of today.\n"
    "2.  **Evidence-First for Structure**: For the remaining uncertain variables, use `web_search` extensively to discover causal links. Formulate your search queries as specific, probing questions aimed at uncovering causal links, not just keywords. For example, instead of 'Taiwan independence factors', ask 'What are the military, economic, and political factors that influence Taiwan's decision to declare formal independence?'.\n"
    "3.  **Identify States**: For each node you do include, define a set of mutually exclusive and collectively exhaustive states. Use `web_search` with targeted questions to find common or expert-defined states for a variable. For instance, ask 'What are the standard ways to categorize public support for a policy?'.\n\n"
    "Return the entire structure as a single, valid JSON object conforming to the `BNStructure` schema. You MUST also identify and specify the `target_node_name` in the output, which should be the name of the node that directly answers the forecasting question. Do not populate the CPTs; that will be done by another agent."
)

CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE = (
    "You are an expert in causal analysis and quantitative estimation. Your task is to analyze the relationship between a 'child' node and its 'parent' nodes in a Bayesian Network and produce a qualitative estimate of its Conditional Probability Table (CPT).\n\n"
    "**Your Process:**\n"
    "1.  **Holistic Research:** Use `web_search` with targeted, fully-formed questions to understand the causal system as a whole. Do not just search for one variable or use keywords. Instead, ask probing questions about how the parent nodes *jointly* influence the child. For example: 'How does the level of international tension and domestic political stability jointly affect the likelihood of a country investing in renewable energy?'.\n"
    "2.  **Relative Estimation:** For each possible combination of parent states, your goal is to determine the *relative likelihood* of each of the child's states. Express this using integer scores (e.g., from 1 to 100). A score of 0 is acceptable for impossible outcomes.\n"
    "3.  **Focus on Ratios, Not Absolutes:** The absolute numbers don't matter as much as their ratios. If you think Child State A is three times more likely than Child State B, you could score them as `{'A': 75, 'B': 25}` or `{'A': 60, 'B': 20}`. The normalization will be handled later.\n"
    "4.  **Provide Justification:** In the `justification` field, explain your reasoning. Why are certain parent states more influential? What evidence underpins your likelihood scores? This is crucial for transparency.\n\n"
    "Return a single, valid JSON object conforming to the `QualitativeCpt` schema."
)

FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE = "You are an expert forecaster and data analyst. You have been provided with a complete Bayesian Network (BN), including Conditional Probability Tables (CPTs), that has been meticulously constructed and populated based on evidence.\nYour task is to interpret this model and provide a final forecast.\nYour analysis must be based *exclusively* on the provided BN data. Do NOT use your own knowledge or any external information. Your role is to be the analytical interpreter of the model, not an independent researcher."

# Note: EvidenceCollectorAgent is not part of the PipelineAgentsConfig for this refactoring,
# so its instructions are not included here. If it were, its template would be defined similarly.


# ==============================================================================
# ### MODIFIED ### Generic Orchestration Pipeline
# ==============================================================================


async def run_cpt_estimator_for_node(
    cpt_estimator_agent_config: AgentConfig,  # MODIFIED: Pass agent config
    base_prompt_template_for_retries: str,  # Used for retries, contains agent's core instructions + initial dynamic parts
    node_name: str,
    max_retries: int,
    # context_summary: str, # No longer needed directly, should be part of base_prompt_template
    # topic: str, # No longer needed directly
    # node_obj_description: str, # No longer needed directly
    # node_obj_states: List[str], # No longer needed directly
    # parents_info_str: str, # No longer needed directly
    combo_list_str_for_retry: str,  # Still needed for retry prompt construction
    # initial_estimator_prompt_template: str, # Renamed to base_prompt_template_for_retries
) -> tuple[str, Optional[QualitativeCpt]]:
    """
    Helper async function to run CptEstimatorAgent for a single node with retry logic.
    Returns the node name and the qualitative CPT or None if it fails.
    """
    qualitative_cpt = None

    # Instantiate CPT Estimator Agent model
    cpt_model = OpenAIChatCompletionsModel(
        model=cpt_estimator_agent_config.model_settings.model_name,
        openai_client=cpt_estimator_agent_config.model_settings.openai_client,
    )
    # Instantiate CPT Estimator Agent
    cpt_estimator_agent_instance = cpt_estimator_agent_config.agent_class(
        name="CptEstimatorAgentDynamic",  # Name can be dynamic or fixed
        model=cpt_model,
        instructions=cpt_estimator_agent_config.instructions_template,  # Base instructions
        tools=cpt_estimator_agent_config.tools,
        output_type=cpt_estimator_agent_config.output_schema,
    )

    # The 'prompt' passed to this function is now the fully formed initial prompt
    # including dynamic parts like context, node info etc. which was built based on instructions_template
    current_prompt = base_prompt_template_for_retries

    for i in range(max_retries):
        try:
            print(f"  - Attempt {i + 1}/{max_retries} for CPT of node '{node_name}'...")
            # Use the dynamically instantiated agent
            cpt_estimator_result = await Runner.run(
                cpt_estimator_agent_instance, current_prompt
            )

            if cpt_estimator_result.final_output and isinstance(
                cpt_estimator_result.final_output, QualitativeCpt
            ):
                qualitative_cpt = cpt_estimator_result.final_output
                print(
                    f"  - Successfully received qualitative CPT for '{node_name}' on attempt {i + 1}."
                )
                break  # Success
            else:
                error_message = f"Agent returned invalid output type on attempt {i+1}. Expected QualitativeCpt but got {type(cpt_estimator_result.final_output)}."
                print(f"  - {error_message} for node '{node_name}'")
                current_prompt = (
                    base_prompt_template_for_retries  # Start with the base prompt (which includes original instructions and dynamic info)
                    + f"\n\nYour previous attempt for node '{node_name}' failed. {error_message}. "
                    "Please ensure you return a single, valid JSON object that strictly follows the `QualitativeCpt` schema. "
                    "Do not add any commentary before or after the JSON. "
                    f"Remember to include all parent state combinations:\n{combo_list_str_for_retry}"
                )

        except exceptions.ModelBehaviorError as e:
            print(
                f"  - Attempt {i + 1}/{max_retries} for node '{node_name}' failed with validation error: {e}"
            )
            current_prompt = (
                base_prompt_template_for_retries
                + f"\n\nYour previous attempt for node '{node_name}' failed with a validation error. The JSON you provided was invalid. Please fix it. The error was: {str(e)}. "
                "You MUST return a single, valid JSON object conforming to the `QualitativeCpt` schema. Do not output any text before or after the JSON object. "
                f"Remember to include all parent state combinations:\n{combo_list_str_for_retry}"
            )
        except Exception as e:
            print(
                f"  - Attempt {i + 1}/{max_retries} for node '{node_name}' failed with an unexpected error: {type(e).__name__} - {e}"
            )
            current_prompt = (
                base_prompt_template_for_retries
                + f"\n\nYour previous attempt for node '{node_name}' failed with an unexpected error: {type(e).__name__} - {e}. Please try again, ensuring you follow all instructions. "
                f"Remember to include all parent state combinations:\n{combo_list_str_for_retry}"
            )
    return node_name, qualitative_cpt


async def run_forecasting_pipeline(
    topic: str, agent_configs: PipelineAgentsConfig
):  # MODIFIED: Added agent_configs
    """
    ### MODIFIED ###
    Main pipeline that now takes any `topic` as input.
    """
    global LLM_CALL_COUNT
    LLM_CALL_COUNT = 0
    print(f"--- STARTING GENERIC FORECASTING PIPELINE FOR: {topic} ---")

    # --- Phase 0: ContextAgent establishes the factual baseline ---
    print("\n--- Phase 0: ContextAgent is establishing the factual baseline... ---")
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Instantiate Context Agent model
    context_model = OpenAIChatCompletionsModel(
        model=agent_configs.context_agent.model_settings.model_name,
        openai_client=agent_configs.context_agent.model_settings.openai_client,
    )
    # Instantiate Context Agent
    context_agent_instance = agent_configs.context_agent.agent_class(
        name="ContextAgentDynamic",
        model=context_model,
        instructions=agent_configs.context_agent.instructions_template,
        tools=agent_configs.context_agent.tools,
        output_type=agent_configs.context_agent.output_schema,
    )
    # Dynamic part of the prompt for ContextAgent
    context_prompt = (
        f"The current date is {current_date}. Please establish the factual context for the topic: '{topic}'.\n"
        "Follow your instructions carefully. Use web search to find the latest information on all key entities and concepts. Your output will serve as the ground truth for all subsequent analysis."
    )
    context_report: Optional[ContextReport] = None
    try:
        # Run the dynamically instantiated ContextAgent
        context_result = await Runner.run(context_agent_instance, context_prompt)
        if context_result.final_output and isinstance(
            context_result.final_output, ContextReport
        ):
            context_report = context_result.final_output
        else:
            print(
                f"ContextAgent returned invalid output type. Expected ContextReport but got {type(context_result.final_output)}."
            )
    except Exception as e:
        print(
            f"An unexpected error of type {type(e).__name__} occurred during ContextAgent execution: {e}"
        )

    if not context_report:
        print("ContextAgent failed to return a valid ContextReport. Exiting.")
        return
    print("\n--- Context Phase Complete. Verified Facts: ---")
    print(context_report.model_dump_json(indent=2))
    context_facts_str = "\n".join(f"- {fact}" for fact in context_report.verified_facts)
    context_summary = context_report.summary

    # --- Phase 1: ArchitectAgent builds the graph for the given topic ---
    print("\n--- Phase 1: ArchitectAgent is defining the BN structure... ---")

    # Instantiate Architect Agent model
    architect_model = OpenAIChatCompletionsModel(
        model=agent_configs.architect_agent.model_settings.model_name,
        openai_client=agent_configs.architect_agent.model_settings.openai_client,
    )
    # Instantiate Architect Agent
    architect_agent_instance = agent_configs.architect_agent.agent_class(
        name="ArchitectAgentDynamic",
        model=architect_model,
        instructions=agent_configs.architect_agent.instructions_template,  # Base instructions from config
        tools=agent_configs.architect_agent.tools,
        output_type=agent_configs.architect_agent.output_schema,
    )
    # Dynamic part of the prompt for ArchitectAgent (prepended to instructions from template)
    architect_prompt = (
        f"**Established Context:**\n{context_summary}\n\n**Verified Facts:**\n{context_facts_str}\n\n"
        f"Based on the context above, and with the current date being {current_date}, please design a Bayesian Network for the topic: '{topic}'.\n"
        "Follow your agent's core instructions carefully (already provided). Use the provided context as your source of truth. Use web search to verify temporal statuses of events. Do not model known past events as uncertain variables. "
        "Your final network should only contain nodes representing events that are still uncertain."
    )
    bn_structure: Optional[BNStructure] = None
    try:
        # Run the dynamically instantiated ArchitectAgent
        architect_result = await Runner.run(
            architect_agent_instance, architect_prompt, max_turns=20
        )
        if architect_result.final_output and isinstance(
            architect_result.final_output, BNStructure
        ):
            bn_structure = architect_result.final_output
        else:
            print(
                f"ArchitectAgent returned invalid output type. Expected BNStructure but got {type(architect_result.final_output)}."
            )
    except Exception as e:
        print(
            f"An unexpected error of type {type(e).__name__} occurred during ArchitectAgent execution: {e}"
        )

    if not bn_structure:
        print("ArchitectAgent failed to return a valid BNStructure. Exiting.")
        return
    print("\n--- Architect Phase Complete. Generated Structure: ---")
    print(bn_structure.model_dump_json(indent=2, exclude={"cpt"}))

    # --- Phase 2: Systematically populate CPT for every node ---
    print("\n--- Phase 2: Evidence Collection and CPT Calculation (Concurrent)... ---")
    tasks = []
    max_retries = 3  # Define max_retries for CPT estimation

    # Store initial prompt templates for each node to use in retries
    initial_prompts_map = {}

    for node_name, node_obj in bn_structure.nodes.items():
        parent_nodes = [bn_structure.nodes[p_name] for p_name in node_obj.parents]
        parents_info_str = "\n".join(
            [
                f"- **{p.name}**: {p.description} (States: {list(p.states.keys())})"
                for p in parent_nodes
            ]
        )
        if not parents_info_str:
            parents_info_str = "This is a root node with no parents."

        parent_states_list = [list(p.states.keys()) for p in parent_nodes]
        parent_state_combinations = list(itertools.product(*parent_states_list))
        if not parent_state_combinations:
            parent_state_combinations.append(())

        combo_list_str = "\n".join(
            [f"- {str(combo)}" for combo in parent_state_combinations]
        )

        # The CPT Estimator Agent's core instructions are in agent_configs.cpt_estimator_agent.instructions_template
        # The dynamic parts (context, topic, node info) will be formatted into this template.

        # This is the fully formed prompt for the first attempt, based on the agent's instruction template
        # and filled with dynamic information. This will also serve as the base for retries.
        initial_prompt_for_node_cpt_estimation = (
            agent_configs.cpt_estimator_agent.instructions_template  # Base instructions
            + f"\n\n**Established Context:**\n{context_summary}\n\n**Verified Facts:**\n{context_facts_str}\n\n"
            f"You are creating the CPT for the node '{node_name}' in a Bayesian Network about '{topic}'. Your work must be consistent with the established context above and your core instructions.\n\n"
            f"**Child Node Information:**\n"
            f"- **Name:** {node_name}\n"
            f"- **Description:** {node_obj.description}\n"
            f"- **States:** {list(node_obj.states.keys())}\n\n"
            f"**Parent Node Information:**\n"
            f"{parents_info_str}\n\n"
            "Your task is to generate a `QualitativeCpt` JSON object as per your core instructions. "
            "You must first perform `web_search` to understand how the parent factors influence the child node. "
            "Then, for each of the parent state combinations listed below, provide a dictionary of relative likelihood scores (from 1 to 10) for each child state.\n\n"
            f"**Parent State Combinations to Estimate:**\n{combo_list_str}\n\n"
            "The `cpt_qualitative_estimates` field in your JSON output must have an entry for every combination listed above. The key for each entry must be the string representation of the tuple (e.g., `\"('High', 'Low')\"`)."
        )
        # initial_prompts_map[node_name] = initial_estimator_prompt_template_for_node # No longer needed with new helper sig

        print(f"  - Creating task for CPT estimation of node '{node_name}'...")
        task = run_cpt_estimator_for_node(
            cpt_estimator_agent_config=agent_configs.cpt_estimator_agent,  # Pass the config
            base_prompt_template_for_retries=initial_prompt_for_node_cpt_estimation,  # Full initial prompt
            node_name=node_name,
            max_retries=max_retries,
            combo_list_str_for_retry=combo_list_str,
        )
        tasks.append(task)

    print(f"\n  - Waiting for {len(tasks)} CPT estimation tasks to complete...")
    # Execute all CPT estimation tasks concurrently
    processed_cpt_results = await asyncio.gather(*tasks)
    print("  - All CPT estimation tasks have completed.")

    # Process the results
    for node_name, qualitative_cpt in processed_cpt_results:
        if qualitative_cpt:
            print(
                f"  - Successfully processed CPT for node '{node_name}'. Justification: {qualitative_cpt.justification}"
            )
            # Normalize the qualitative scores into a valid CPT
            normalized_cpt = normalize_cpt(qualitative_cpt)
            # Update the BN structure
            if node_name in bn_structure.nodes:
                bn_structure.nodes[node_name].cpt = normalized_cpt
            else:
                print(
                    f"  - Error: Node '{node_name}' not found in bn_structure after async processing. This should not happen."
                )
        else:
            print(
                f"  - Warning: Failed to get qualitative CPT for node '{node_name}' after {max_retries} attempts. CPT will be empty."
            )

    print("\n--- CPT Population Complete. Final Structure with CPTs: ---")
    final_bn_json = bn_structure.model_dump_json(indent=2)
    print(final_bn_json)

    # --- Phase 3: Calculate Final Probability and Deliver Forecast ---
    print("\n--- Phase 3: Calculating Final Probability & Delivering Verdict... ---")

    final_probabilities_str = "Not calculated."
    if (
        bn_structure.target_node_name
        and bn_structure.target_node_name in bn_structure.nodes
    ):
        print(
            f"\n--- Calculating Final Probability for Target Node: '{bn_structure.target_node_name}' ---"
        )
        try:
            final_probabilities = calculate_final_probability(
                bn=bn_structure,
                target_node_name=bn_structure.target_node_name,
                evidence=None,  # No external evidence provided in this pipeline
            )
            print("Final Probabilities:")
            print(json.dumps(final_probabilities, indent=2))
            final_probabilities_str = json.dumps(final_probabilities, indent=2)
        except Exception as e:
            print(f"An error occurred during final probability calculation: {e}")
            final_probabilities_str = f"Error during calculation: {e}"
    elif not bn_structure.target_node_name:
        print(
            "Warning: No target node specified by ArchitectAgent. Skipping final probability calculation."
        )
    else:
        print(
            f"Warning: Target node '{bn_structure.target_node_name}' not found in the network. Skipping calculation."
        )

    # --- Phase 4: ForecasterAgent gives the final summary ---
    print("\n--- Phase 4: Forecaster Agent delivering the final verdict... ---")

    # Instantiate Forecaster Agent model
    forecaster_model = OpenAIChatCompletionsModel(
        model=agent_configs.forecaster_agent.model_settings.model_name,
        openai_client=agent_configs.forecaster_agent.model_settings.openai_client,
    )
    # Instantiate Forecaster Agent
    forecaster_agent_instance = agent_configs.forecaster_agent.agent_class(
        name="ForecasterAgentDynamic",
        model=forecaster_model,
        instructions=agent_configs.forecaster_agent.instructions_template,  # Base instructions
        tools=agent_configs.forecaster_agent.tools,
        output_type=agent_configs.forecaster_agent.output_schema,  # Should be None or handle plain text
    )

    # Dynamic part of the prompt for ForecasterAgent
    forecaster_prompt = (
        f"{agent_configs.forecaster_agent.instructions_template}\n\n"  # Start with core instructions
        f"**Established Context:**\n{context_summary}\n\n**Verified Facts:**\n{context_facts_str}\n\n"
        f"Analyze the provided Bayesian Network and its CPTs for the topic: '{topic}'.\n\n"
        "Your analysis must be based exclusively on the data below. Do not use outside knowledge. The context and facts provided above are the ground truth from a previous step and are incorporated into the network.\n"
        f"Based on the model, the calculated probability distribution for the target node '{bn_structure.target_node_name}' is:\n{final_probabilities_str}\n\n"
        "1. State the final probability distribution from the calculation above clearly at the beginning of your response.\n"
        "2. Provide a qualitative forecast, summarizing the most likely outcomes according to the model and the calculated probabilities.\n"
        "3. Identify the key drivers and relationships. Which nodes have the most influence on the final outcome? Refer to the CPTs in your explanation.\n"
        "4. Discuss any competing factors or notable sensitivities you observe in the model's structure and probabilities.\n"
        "Be firm and clear in your final summary, as it is the conclusion of a rigorous, evidence-based process.\n\n"
        f"Here is the complete Bayesian Network: {final_bn_json}"
    )
    # Run the dynamically instantiated ForecasterAgent
    final_result = await Runner.run(forecaster_agent_instance, forecaster_prompt)

    print("\n--- PIPELINE COMPLETE. FINAL FORECAST: ---")
    if final_result and final_result.final_output:
        print(final_result.final_output)
    else:
        print("ForecasterAgent did not produce a final output.")
    print(f"\n--- 🚀 TOTAL LLM CALLS: {LLM_CALL_COUNT} ---")


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Primary Input: Forecasting Topic
    # --------------------------------------------------------------------------
    # Modify the `forecasting_topic` variable to change the subject of the forecast.
    # Examples:
    # forecasting_topic = "Will Taiwan declare formal independence by the end of 2026?" # Geopolitical
    # forecasting_topic = "Will the US Federal Reserve cut interest rates in Q3 2025?" # Economic
    forecasting_topic = "Will Sudan ratify AfCFTA (the pan-African free trade agreement) before July 1, 2025?"

    # Political/Geopolitical

    # --------------------------------------------------------------------------
    # Agent Configuration Setup
    # --------------------------------------------------------------------------
    # `default_pipeline_configs` holds all agent configurations for the pipeline.
    # You can customize agent behavior by modifying this object before passing it
    # to `run_forecasting_pipeline`.

    # Example: Change the model for a specific agent:
    #   default_pipeline_configs.architect_agent.model_settings.model_name = "gpt-4o"
    #   (Make sure the client supports this model and you have access)

    # Example: Change instructions for an agent:
    #   1. Modify one of the INSTRUCTIONS_TEMPLATE constants above.
    #   2. Or, for a one-off change:
    #      default_pipeline_configs.context_agent.instructions_template = "New custom instructions for context agent..."

    # Define the default model configuration shared by agents
    default_model_config = ModelConfig(model_name="o4-mini", openai_client=client)

    # Populate the pipeline agent configurations using the defaults
    default_pipeline_configs = PipelineAgentsConfig(
        context_agent=AgentConfig(
            agent_class=Agent,  # Using the base agents.Agent class
            model_settings=default_model_config,
            tools=[web_search],
            output_schema=AgentOutputSchema(ContextReport, strict_json_schema=False),
            instructions_template=CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE,
        ),
        architect_agent=AgentConfig(
            agent_class=Agent,  # Using the base agents.Agent class
            model_settings=default_model_config,
            tools=[web_search],
            output_schema=AgentOutputSchema(BNStructure, strict_json_schema=False),
            instructions_template=ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE,
        ),
        cpt_estimator_agent=AgentConfig(
            agent_class=Agent,  # Using the base agents.Agent class
            model_settings=default_model_config,
            tools=[web_search],
            output_schema=AgentOutputSchema(QualitativeCpt, strict_json_schema=False),
            instructions_template=CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE,
        ),
        forecaster_agent=AgentConfig(
            agent_class=Agent,  # Using the base agents.Agent class
            model_settings=default_model_config,
            tools=[],  # No tools for forecaster
            output_schema=None,  # Outputs plain text, handled by Runner
            instructions_template=FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE,
        ),
    )

    # Example of how to modify a configuration before running the pipeline:
    # print("\n!!! EXAMPLE: Overriding ArchitectAgent model to 'gpt-4o' (example) !!!\n")
    # default_pipeline_configs.architect_agent.model_settings.model_name = "gpt-4o"

    log_dir = pathlib.Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"bayesian_experiment_{timestamp}.log"
    original_stdout = sys.stdout
    with open(log_filename, "w") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            asyncio.run(
                run_forecasting_pipeline(
                    topic=forecasting_topic, agent_configs=default_pipeline_configs
                )
            )  # Pass configs
        finally:
            sys.stdout = original_stdout
    print(f"\nFull logs have been exported to {log_filename}")
