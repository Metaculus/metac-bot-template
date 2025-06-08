import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Tuple

# Agent framework imports
from agents import (
    Agent as ActualAgentClass,
    Runner,
    OpenAIChatCompletionsModel,
    AgentOutputSchema,
    exceptions as agent_exceptions,
)

# Project-specific imports
from .pipeline_core import PipelineStep
from .pipeline_models import (
    PipelineData,
    AgentConfig,
    ContextReport,
    BNStructure,
    QualitativeCpt,
    Node,
)

from datetime import datetime
import itertools
import json  # Added for FinalCalculationStep


class BaseAgentStep(PipelineStep, ABC):
    def __init__(self, agent_config: AgentConfig, agent_name: Optional[str] = None):
        self.agent_config = agent_config
        self.agent_name = agent_name if agent_name else self.__class__.__name__
        if not issubclass(self.agent_config.agent_class, ActualAgentClass):
            raise ValueError(
                f"agent_config.agent_class ({self.agent_config.agent_class}) "
                f"must be a subclass of agents.Agent ({ActualAgentClass})"
            )
        self._model = OpenAIChatCompletionsModel(
            model=self.agent_config.model_settings.model_name,
            openai_client=self.agent_config.model_settings.openai_client,
        )
        self._agent = self.agent_config.agent_class(
            name=self.agent_name,
            model=self._model,
            instructions=self.agent_config.instructions_template,
            tools=self.agent_config.tools,
            output_type=self.agent_config.output_schema,
        )
        print(
            f"Initialized {self.agent_name} with model {self.agent_config.model_settings.model_name}"
        )

    @abstractmethod
    def _prepare_prompt(self, data: PipelineData) -> str:
        pass

    @abstractmethod
    def _process_output(
        self, agent_raw_output: Any, data: PipelineData
    ) -> PipelineData:
        pass

    async def _run_agent_with_retry(
        self,
        initial_prompt: str,
        max_retries: int = 3,
        prompt_retry_instruction_addon: str = "Please review your previous output and try again, ensuring you adhere to the required format and instructions.",
    ) -> Any:
        current_prompt = initial_prompt
        agent_final_output = None
        for attempt in range(max_retries):
            print(f"  - {self.agent_name} attempt {attempt + 1}/{max_retries}...")
            try:
                agent_result = await Runner.run(self._agent, current_prompt)
                if agent_result.final_output is not None:
                    agent_final_output = agent_result.final_output
                    print(f"  - {self.agent_name} successful on attempt {attempt + 1}.")
                    break
                else:
                    error_message = (
                        f"{self.agent_name} returned no output on attempt {attempt+1}."
                    )
                    print(f"  - {error_message}")
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            f"{self.agent_name} failed after {max_retries} attempts: {error_message}"
                        )
                    current_prompt = (
                        initial_prompt
                        + f"\n\nPREVIOUS ATTEMPT FAILED: {error_message}. {prompt_retry_instruction_addon}"
                    )
            except agent_exceptions.ModelBehaviorError as e:
                print(
                    f"  - {self.agent_name} attempt {attempt + 1}/{max_retries} failed with validation error: {e}"
                )
                if attempt == max_retries - 1:
                    raise agent_exceptions.ModelBehaviorError(
                        f"{self.agent_name} validation error after {max_retries} attempts: {e}"
                    ) from e
                current_prompt = (
                    initial_prompt
                    + f"\n\nPREVIOUS ATTEMPT FAILED with validation error: {e}. {prompt_retry_instruction_addon}"
                )
            except Exception as e:
                print(
                    f"  - {self.agent_name} attempt {attempt + 1}/{max_retries} failed with an unexpected error: {type(e).__name__} - {e}"
                )
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"{self.agent_name} failed with unexpected error after {max_retries} attempts: {e}"
                    ) from e
                current_prompt = (
                    initial_prompt
                    + f"\n\nPREVIOUS ATTEMPT FAILED with an unexpected error: {type(e).__name__} - {e}. {prompt_retry_instruction_addon}"
                )
        if agent_final_output is None and max_retries > 0:
            raise RuntimeError(
                f"{self.agent_name} failed to produce output after {max_retries} attempts and no specific exception was caught in the last try."
            )
        return agent_final_output

    async def execute(self, data: PipelineData) -> PipelineData:
        print(f"Executing {self.__class__.__name__} (agent: {self.agent_name})...")
        try:
            prompt = self._prepare_prompt(data)
            if not prompt:
                print(
                    f"  - Prompt preparation for {self.agent_name} returned no prompt. Skipping agent run."
                )
                return data
            max_run_retries = 3
            if (
                hasattr(self.agent_config, "max_retries")
                and self.agent_config.max_retries is not None
            ):  # Check if attribute exists and is not None
                max_run_retries = self.agent_config.max_retries
            # Ensure agent_config is a Pydantic model for .dict() or handle dict case
            elif hasattr(self.agent_config, "dict") and callable(
                getattr(self.agent_config, "dict")
            ):  # Check if it has dict() method
                max_run_retries = self.agent_config.dict().get("max_retries", 3)
            elif (
                isinstance(self.agent_config, dict)
                and "max_retries" in self.agent_config
            ):
                max_run_retries = self.agent_config.get("max_retries", 3)

            agent_raw_output = await self._run_agent_with_retry(
                initial_prompt=prompt, max_retries=max_run_retries
            )
            updated_data = self._process_output(agent_raw_output, data)
            print(f"{self.__class__.__name__} completed.")
            return updated_data
        except Exception as e:
            print(f"  - CRITICAL ERROR in {self.__class__.__name__}: {e}")
            error_field = "error_message"
            if hasattr(data, error_field) and getattr(data, error_field):
                setattr(
                    data,
                    error_field,
                    getattr(data, error_field)
                    + f"; Error in {self.__class__.__name__}: {str(e)}",
                )
            elif hasattr(data, error_field):
                setattr(
                    data, error_field, f"Error in {self.__class__.__name__}: {str(e)}"
                )
            raise


class ContextStep(BaseAgentStep):
    def __init__(self, agent_config: AgentConfig, agent_name: str = "ContextAgent"):
        super().__init__(agent_config=agent_config, agent_name=agent_name)
        if not (
            self.agent_config.output_schema
            and hasattr(self.agent_config.output_schema, "pydantic_model")
            and self.agent_config.output_schema.pydantic_model is ContextReport
        ):
            print(
                f"Warning: {self.agent_name} initialized with AgentConfig that might not specifically output ContextReport."
            )

    def _prepare_prompt(self, data: PipelineData) -> str:
        topic = data.topic
        if not topic:
            raise ValueError("Topic not found in PipelineData for ContextStep.")
        self.current_date_for_step = datetime.now().strftime("%Y-%m-%d")
        return f"The current date is {self.current_date_for_step}. Please establish the factual context for the topic: '{topic}'."

    def _process_output(
        self, agent_raw_output: Any, data: PipelineData
    ) -> PipelineData:
        if not isinstance(agent_raw_output, ContextReport):
            error_msg = f"{self.agent_name} did not return a valid ContextReport. Output: {type(agent_raw_output)}"
            print(f"ERROR in ContextStep._process_output: {error_msg}")
            data.error_message = f"{data.error_message if data.error_message else ''}; {error_msg}".strip(
                "; "
            )
            raise TypeError(error_msg)
        data.current_date = getattr(
            self, "current_date_for_step", datetime.now().strftime("%Y-%m-%d")
        )
        data.context_report = agent_raw_output
        data.context_facts_str = "\n".join(
            f"- {fact}" for fact in agent_raw_output.verified_facts
        )
        data.context_summary = agent_raw_output.summary
        print(f"  ContextStep processed. Summary: {data.context_summary[:100]}...")
        return data


class ArchitectStep(BaseAgentStep):
    def __init__(self, agent_config: AgentConfig, agent_name: str = "ArchitectAgent"):
        super().__init__(agent_config=agent_config, agent_name=agent_name)
        if not (
            self.agent_config.output_schema
            and hasattr(self.agent_config.output_schema, "pydantic_model")
            and self.agent_config.output_schema.pydantic_model is BNStructure
        ):
            print(
                f"Warning: {self.agent_name} initialized with AgentConfig that might not specifically output BNStructure."
            )

    def _prepare_prompt(self, data: PipelineData) -> str:
        if not all(
            [
                data.topic,
                data.current_date,
                data.context_summary,
                data.context_facts_str,
            ]
        ):
            raise ValueError("Missing required fields for ArchitectStep.")
        return (
            f"**Established Context:**\n{data.context_summary}\n\n**Verified Facts:**\n{data.context_facts_str}\n\n"
            f"Based on the context above, and with the current date being {data.current_date}, "
            f"please design a Bayesian Network for the topic: '{data.topic}'.\n"
            "Your agent's core instructions (already provided via AgentConfig) contain details. Focus on uncertain events.\n\n"
            "IMPORTANT: For each node, the `states` field must be a JSON object (a dictionary), where each key is a state name (string) and its value is a brief description of that state (string)."
        )

    def _process_output(
        self, agent_raw_output: Any, data: PipelineData
    ) -> PipelineData:
        if not isinstance(agent_raw_output, BNStructure):
            error_msg = f"{self.agent_name} did not return a valid BNStructure. Output: {type(agent_raw_output)}"
            print(f"ERROR in ArchitectStep._process_output: {error_msg}")
            data.error_message = f"{data.error_message if data.error_message else ''}; {error_msg}".strip(
                "; "
            )
            raise TypeError(error_msg)
        data.bn_structure = agent_raw_output
        print(
            f"  ArchitectStep processed. BN for '{agent_raw_output.topic}'. Target: {agent_raw_output.target_node_name}"
        )
        return data


def normalize_cpt(qualitative_cpt: QualitativeCpt) -> Dict[str, Dict[str, float]]:
    normalized_cpt_output = {}
    for (
        parent_combo_str,
        child_scores,
    ) in qualitative_cpt.cpt_qualitative_estimates.items():
        total_score = sum(child_scores.values())
        if total_score == 0:
            num_states = len(child_scores)
            normalized_probs = (
                {state: 1.0 / num_states for state in child_scores}
                if num_states > 0
                else {}
            )
        else:
            normalized_probs = {
                state: score / total_score for state, score in child_scores.items()
            }
        normalized_cpt_output[parent_combo_str] = normalized_probs
    return normalized_cpt_output


class CPTGenerationStep(PipelineStep):
    def __init__(
        self, agent_config: AgentConfig, agent_name_prefix: str = "CptEstimator"
    ):
        self.agent_config = agent_config
        self.agent_name_prefix = agent_name_prefix
        if not issubclass(self.agent_config.agent_class, ActualAgentClass):
            raise ValueError(
                f"agent_config.agent_class must be subclass of agents.Agent"
            )
        if not (
            self.agent_config.output_schema
            and isinstance(self.agent_config.output_schema, AgentOutputSchema)
            and hasattr(self.agent_config.output_schema, "pydantic_model")
            and self.agent_config.output_schema.pydantic_model is QualitativeCpt
        ):
            print(
                f"Warning: {self.agent_name_prefix} initialized with AgentConfig not for QualitativeCpt."
            )

    async def _run_single_cpt_estimation(
        self,
        node_name: str,
        node_obj: Node,
        bn_structure_topic: str,
        context_summary: str,
        context_facts_str: str,
        parent_nodes_info: List[Tuple[str, str, List[str]]],
        parent_state_combinations: List[Tuple[str, ...]],
    ) -> Optional[QualitativeCpt]:
        agent_dynamic_name = (
            f"{self.agent_name_prefix}_{node_name.replace(' ', '_').replace('-', '_')}"
        )
        model = OpenAIChatCompletionsModel(
            model=self.agent_config.model_settings.model_name,
            openai_client=self.agent_config.model_settings.openai_client,
        )
        agent_instance = self.agent_config.agent_class(
            name=agent_dynamic_name,
            model=model,
            instructions=self.agent_config.instructions_template,
            tools=self.agent_config.tools,
            output_type=self.agent_config.output_schema,
        )
        parents_info_str = (
            "\n".join(
                [
                    f"- **{p_name}**: {p_desc} (States: {p_states})"
                    for p_name, p_desc, p_states in parent_nodes_info
                ]
            )
            or "This is a root node."
        )
        combo_list_str = "\n".join(
            [f"- {str(combo)}" for combo in parent_state_combinations]
        )
        initial_prompt = (
            f"**Context:**\n{context_summary}\n\n**Facts:**\n{context_facts_str}\n\n"
            f"CPT for node '{node_name}' in BN about '{bn_structure_topic}'. Core instructions apply.\n"
            f"**Child Node:** {node_name} ({node_obj.description}) States: {list(node_obj.states.keys())}\n"
            f"**Parent Nodes:**\n{parents_info_str}\n\n"
            f"Generate `QualitativeCpt` JSON. Use `web_search` for influence. Estimate likelihood scores for child states given parent combinations:\n{combo_list_str}\n"
            "Ensure `cpt_qualitative_estimates` keys match these combinations."
        )
        current_prompt = initial_prompt
        # max_retries should be part of agent_config, e.g. agent_config.max_retries
        max_run_retries = 3
        if (
            hasattr(self.agent_config, "max_retries")
            and self.agent_config.max_retries is not None
        ):
            max_run_retries = self.agent_config.max_retries
        elif hasattr(self.agent_config, "dict") and callable(
            getattr(self.agent_config, "dict")
        ):
            max_run_retries = self.agent_config.dict().get("max_retries", 3)
        elif isinstance(self.agent_config, dict) and "max_retries" in self.agent_config:
            max_run_retries = self.agent_config.get("max_retries", 3)

        for attempt in range(max_run_retries):
            print(
                f"  - CPT for '{node_name}': {agent_dynamic_name} attempt {attempt + 1}/{max_run_retries}..."
            )
            try:
                agent_result = await Runner.run(agent_instance, current_prompt)
                if isinstance(agent_result.final_output, QualitativeCpt):
                    print(f"  - Success CPT for '{node_name}' attempt {attempt + 1}.")
                    return agent_result.final_output
                error_message = (
                    f"Invalid output type: {type(agent_result.final_output)}."
                )
                if attempt == max_run_retries - 1:
                    break
                current_prompt = (
                    initial_prompt + f"\n\nFAILED: {error_message} Review & retry."
                )
            except Exception as e:
                print(
                    f"  - CPT for '{node_name}': Error attempt {attempt + 1}: {type(e).__name__} - {e}"
                )
                if attempt == max_run_retries - 1:
                    break
                current_prompt = (
                    initial_prompt
                    + f"\n\nFAILED (Error: {type(e).__name__} - {e}). Retry."
                )
        print(f"  - Failed CPT for '{node_name}' after {max_run_retries} attempts.")
        return None

    async def execute(self, data: PipelineData) -> PipelineData:
        print("Executing CPTGenerationStep...")
        if not data.bn_structure:
            raise ValueError("BNStructure missing for CPTGenerationStep.")
        if not all([data.context_summary, data.context_facts_str, data.topic]):
            raise ValueError("Context/topic missing.")
        bn_obj, tasks = data.bn_structure, []
        for name, node in bn_obj.nodes.items():
            parents_info = [
                (p.name, p.description, list(p.states.keys()))
                for p_name in node.parents
                if (p := bn_obj.nodes.get(p_name))
            ]
            parent_states = [info[2] for info in parents_info]
            state_combos = list(itertools.product(*parent_states)) or [tuple()]
            print(f"  - Task for CPT of node '{name}'...")
            tasks.append(
                asyncio.create_task(
                    self._run_single_cpt_estimation(
                        name,
                        node,
                        bn_obj.topic or data.topic,
                        data.context_summary,
                        data.context_facts_str,
                        parents_info,
                        state_combos,
                    ),
                    name=f"CPT_{name}",
                )
            )
        print(f"  - Waiting for {len(tasks)} CPT tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print("  - All CPT tasks completed.")
        count = 0
        for i, res in enumerate(results):
            node_name = list(bn_obj.nodes.keys())[i]
            if isinstance(res, QualitativeCpt):
                print(
                    f"  - Processed CPT for '{node_name}'. Justification: {res.justification}"
                )
                bn_obj.nodes[node_name].cpt = normalize_cpt(res)
                count += 1
            else:
                data.error_message = f"{data.error_message or ''}; CPT fail for {node_name}: {res}".strip(
                    "; "
                )
        if count < len(bn_obj.nodes):
            data.error_message = f"{data.error_message or ''}; Only {count}/{len(bn_obj.nodes)} CPTs generated.".strip(
                "; "
            )
        data.bn_structure = bn_obj
        print(
            f"CPTGenerationStep complete. {count}/{len(bn_obj.nodes)} CPTs generated."
        )
        return data


def calculate_final_probability(
    bn: BNStructure, target_node_name: str, evidence: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
    evidence = evidence or {}
    if not bn.nodes:
        return {}

    order, q, in_degree = (
        [],
        [n for n, d in bn.nodes.items() if not d.parents],
        {n: len(d.parents) for n, d in bn.nodes.items()},
    )
    processed = set()
    while q:
        u = q.pop(0)
        order.append(u)
        processed.add(u)
        for v_name, v_node in bn.nodes.items():
            if u in v_node.parents:
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0 and v_name not in processed:
                    q.append(v_name)
    if len(order) != len(bn.nodes):
        order.extend([n for n in bn.nodes if n not in order])

    memo = {}

    def enumerate_all_recursive(
        vars_list: List[str], current_evidence: Dict[str, str]
    ) -> float:
        vt, et = tuple(sorted(vars_list)), tuple(sorted(current_evidence.items()))
        if (vt, et) in memo:
            return memo[(vt, et)]
        if not vars_list:
            return 1.0
        Y, rest = vars_list[0], vars_list[1:]
        if Y not in bn.nodes:
            return enumerate_all_recursive(rest, current_evidence)

        node_Y, cpt_key_parts = bn.nodes[Y], []
        for p_name in node_Y.parents:
            cpt_key_parts.append(current_evidence.get(p_name))
        cpt_key = str(tuple(cpt_key_parts)) if node_Y.parents else "()"

        if Y in current_evidence:
            prob_y = node_Y.cpt.get(cpt_key, {}).get(current_evidence[Y], 0.0)
            res = prob_y * enumerate_all_recursive(rest, current_evidence)
        else:
            res = sum(
                node_Y.cpt.get(cpt_key, {}).get(y_s, 0.0)
                * enumerate_all_recursive(rest, {**current_evidence, Y: y_s})
                for y_s in (node_Y.states or {}).keys()
            )
        memo[(vt, et)] = res
        return res

    if target_node_name not in bn.nodes or not bn.nodes[target_node_name].states:
        return {}

    dist = {
        s: enumerate_all_recursive(order, {**evidence, target_node_name: s})
        for s in bn.nodes[target_node_name].states
    }
    total = sum(dist.values())
    return (
        {s: p / total for s, p in dist.items()}
        if total > 0
        else {s: 1.0 / len(dist) for s in dist} if dist else {}
    )


class FinalCalculationStep(PipelineStep):
    def __init__(self, evidence: Optional[Dict[str, str]] = None):
        self.evidence = evidence or {}
        print(
            f"Initialized FinalCalculationStep with evidence: {self.evidence if self.evidence else 'None'}"
        )

    async def execute(self, data: PipelineData) -> PipelineData:
        print("Executing FinalCalculationStep...")
        if not data.bn_structure:
            error_msg = "BNStructure not found for FinalCalculationStep."
            data.error_message = f"{data.error_message or ''}; {error_msg}".strip("; ")
            data.final_probabilities, data.final_probabilities_str = None, error_msg
            raise ValueError(error_msg)

        bn_obj, target_name = data.bn_structure, data.bn_structure.target_node_name
        probs, probs_str, err_msg = None, "Not calculated.", None

        if not target_name:
            err_msg = "No target node in BN. Skipping calculation."
        elif target_name not in bn_obj.nodes:
            err_msg = f"Target '{target_name}' not in BN. Skipping."
        else:
            print(f"  - Calculating Final Probability for Target: '{target_name}'")
            try:
                for name, node in bn_obj.nodes.items():
                    if (
                        not node.parents
                        and not node.cpt
                        and name not in self.evidence
                        and name != target_name
                    ):
                        print(
                            f"  - Warning: Root Node '{name}' has no CPT and is not evidence."
                        )
                    elif node.parents and not node.cpt and name != target_name:
                        print(f"  - Warning: Node '{name}' has parents but no CPT.")

                probs = calculate_final_probability(
                    bn=bn_obj, target_node_name=target_name, evidence=self.evidence
                )

                if probs:
                    probs_str = json.dumps(probs, indent=2)
                    print(f"  - Final Probabilities: {probs_str}")
                else:
                    probs_str = "Calculation resulted in empty or None probabilities."
                    print(f"  - {probs_str}")

            except Exception as e:
                err_msg = f"Error during final probability calculation: {type(e).__name__} - {e}"
                import traceback

                print(f"  - {err_msg}\n{traceback.format_exc()}")
                probs_str = f"Error: {e}"

        if err_msg:
            print(f"  - {err_msg}")
            probs_str = err_msg if probs_str == "Not calculated." else probs_str
            if data.error_message:
                data.error_message += f"; {err_msg}"
            else:
                data.error_message = err_msg

        data.final_probabilities = probs
        data.final_probabilities_str = probs_str

        print("FinalCalculationStep complete.")
        return data


class ForecasterStep(BaseAgentStep):
    """
    Pipeline step where a ForecasterAgent provides a final summary and interpretation
    based on the generated Bayesian Network and calculated probabilities.
    Inherits from BaseAgentStep.
    """

    def __init__(self, agent_config: AgentConfig, agent_name: str = "ForecasterAgent"):
        """
        Initializes the ForecasterStep.
        AgentConfig should be pre-configured for a Forecaster agent, including:
        - output_schema=None (for plain text output)
        - instructions_template=FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE
        """
        super().__init__(agent_config=agent_config, agent_name=agent_name)
        if self.agent_config.output_schema is not None:
            # print(f"Warning: {self.agent_name} initialized with AgentConfig that has an output_schema. Expected None for plain text.")
            pass

    def _prepare_prompt(self, data: PipelineData) -> str:
        """
        Prepares the dynamic part of the prompt for the ForecasterAgent.
        """
        topic = data.topic
        context_summary = data.context_summary
        context_facts_str = data.context_facts_str
        bn_structure_obj = data.bn_structure
        final_probs_str = data.final_probabilities_str

        if not all(
            [
                topic,
                context_summary,
                context_facts_str,
                bn_structure_obj,
                final_probs_str,
            ]
        ):
            missing_parts = [
                part
                for part, val in {
                    "topic": topic,
                    "context_summary": context_summary,
                    "context_facts_str": context_facts_str,
                    "bn_structure": bn_structure_obj,
                    "final_probabilities_str": final_probs_str,
                }.items()
                if not val
            ]
            if not bn_structure_obj:
                raise ValueError(
                    f"BNStructure is missing in PipelineData for ForecasterStep. Missing parts: {missing_parts}"
                )
            print(
                f"Warning: Missing some data for ForecasterStep prompt: {missing_parts}. Proceeding with available info."
            )

        final_bn_json_str = (
            bn_structure_obj.model_dump_json(indent=2) if bn_structure_obj else "{}"
        )
        target_node_name_str = (
            bn_structure_obj.target_node_name
            if bn_structure_obj and bn_structure_obj.target_node_name
            else "N/A"
        )

        prompt = (
            f"**Established Context:**\n{context_summary or 'Not available'}\n\n"
            f"**Verified Facts:**\n{context_facts_str or 'Not available'}\n\n"
            f"Analyze the provided Bayesian Network and its CPTs for the topic: '{topic}'.\n\n"
            "Your analysis must be based exclusively on the data below. Do not use outside knowledge. "
            "The context and facts provided above are the ground truth from previous steps and are incorporated into the network.\n"
            f"Based on the model, the calculated probability distribution for the target node '{target_node_name_str}' is:\n{final_probs_str or 'Not available/Error in calculation'}\n\n"
            "1. State the final probability distribution from the calculation above clearly at the beginning of your response.\n"
            "2. Provide a qualitative forecast, summarizing the most likely outcomes according to the model and the calculated probabilities.\n"
            "3. Identify the key drivers and relationships. Which nodes have the most influence on the final outcome? Refer to the CPTs in your explanation.\n"
            "4. Discuss any competing factors or notable sensitivities you observe in the model's structure and probabilities.\n"
            "Be firm and clear in your final summary, as it is the conclusion of a rigorous, evidence-based process.\n\n"
            f"Here is the complete Bayesian Network (JSON):\n{final_bn_json_str}"
        )
        return prompt

    def _process_output(
        self, agent_raw_output: Any, data: PipelineData
    ) -> PipelineData:
        """
        Processes the plain text output from the ForecasterAgent and updates PipelineData.
        """
        if agent_raw_output is None:
            forecaster_text_output = (
                "ForecasterAgent did not produce any output after retries."
            )
            print(f"ERROR in ForecasterStep._process_output: {forecaster_text_output}")
            data.error_message = f"{data.error_message if data.error_message else ''}; {forecaster_text_output}".strip(
                "; "
            )
        else:
            forecaster_text_output = str(agent_raw_output)

        data.forecaster_agent_output = forecaster_text_output
        print(
            f"  ForecasterStep processed. Output length: {len(data.forecaster_agent_output or '')}"
        )
        return data
