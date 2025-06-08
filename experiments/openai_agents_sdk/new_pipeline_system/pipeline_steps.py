import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type, Tuple, Generic # Added Generic

# Agent framework imports
from agents import (
    Agent as ActualAgentClass,
    Runner,
    OpenAIChatCompletionsModel,
    AgentOutputSchema,
    exceptions as agent_exceptions,
)

# Project-specific imports
from .pipeline_core import PipelineStep, InputStepData, OutputStepData # Updated
from .pipeline_models import (
    # Removed: PipelineData
    BasePipelineData, # Added
    ContextStepOutput, # Added
    ArchitectStepInput, # Added
    ArchitectStepOutput, # Added
    CPTGenerationStepInput, # Added
    CPTGenerationStepOutput, # Added
    FinalCalculationStepInput, # Added
    FinalCalculationStepOutput, # Added
    ForecasterStepInput, # Added
    ForecasterStepOutput, # Added
    AgentConfig,
    ContextReport,
    BNStructure,
    QualitativeCpt,
    Node,
)

from datetime import datetime
import itertools
import json


class BaseAgentStep(PipelineStep[InputStepData, OutputStepData], ABC, Generic[InputStepData, OutputStepData]):
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
    def _prepare_prompt(self, data: InputStepData) -> str:
        pass

    @abstractmethod
    def _process_output(
        self, agent_raw_output: Any, data: InputStepData
    ) -> OutputStepData:
        pass

    async def _run_agent_with_retry(
        self,
        initial_prompt: str,
        max_retries: int = 3,
        prompt_retry_instruction_addon: str = "Please review your previous output and try again, ensuring you adhere to the required format and instructions.",
    ) -> Any:
        # This method's core logic remains the same as it deals with prompts and raw outputs,
        # not the structured PipelineData objects directly.
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
        if agent_final_output is None and max_retries > 0: # Ensure agent_final_output is checked only if retries were attempted
             raise RuntimeError(
                f"{self.agent_name} failed to produce output after {max_retries} attempts and no specific exception was caught in the last try."
            )
        return agent_final_output

    async def execute(self, data: InputStepData) -> OutputStepData:
        print(f"Executing {self.__class__.__name__} (agent: {self.agent_name})...")
        try:
            prompt = self._prepare_prompt(data)
            if not prompt: # If prompt is empty, indicates an issue or intentional skip
                print(
                    f"  - Prompt preparation for {self.agent_name} returned no prompt. Skipping agent run."
                )
                # If we need to return an OutputStepData instance, it must be constructible.
                # This assumes that OutputStepData can be instantiated from InputStepData or has a default.
                # This part might need adjustment based on how OutputStepData is defined for specific steps.
                # For now, if a step wants to skip, it should prepare data with an error or appropriate fields.
                # A simple pass-through might not be valid if OutputStepData has new required fields.
                # This needs careful handling in _prepare_prompt or _process_output of subclasses.
                # A common pattern is to create an output object with an error message.
                # We assume _process_output will be called with None and handle it.
                return self._process_output(None, data)


            max_run_retries = 3 # Default
            # Attempt to get max_retries from AgentConfig, which could be Pydantic model or dict
            if hasattr(self.agent_config, "max_retries") and self.agent_config.max_retries is not None:
                max_run_retries = self.agent_config.max_retries
            elif isinstance(self.agent_config, dict) and "max_retries" in self.agent_config: # Fallback for dict
                 max_run_retries = self.agent_config.get("max_retries", 3)


            agent_raw_output = await self._run_agent_with_retry(
                initial_prompt=prompt, max_retries=max_run_retries
            )
            updated_data = self._process_output(agent_raw_output, data)
            print(f"{self.__class__.__name__} completed.")
            return updated_data
        except Exception as e:
            print(f"  - CRITICAL ERROR in {self.__class__.__name__}: {e}")
            # data.error_message is guaranteed by BasePipelineData (via InputStepData bound)
            if data.error_message:
                data.error_message += f"; Error in {self.__class__.__name__}: {str(e)}"
            else:
                data.error_message = f"Error in {self.__class__.__name__}: {str(e)}"
            # Instead of re-raising, we should return an OutputStepData with the error.
            # This requires _process_output to be robust enough to be called even after an error
            # or for us to construct a valid OutputStepData here.
            # For now, re-raising to maintain original behavior, but this means the pipeline halts.
            # To allow continuation, _process_output should handle agent_raw_output=None and populate error,
            # and the execute should return that. Let's assume _process_output is designed for this.
            # If an error occurs before _process_output (e.g. in _prepare_prompt or _run_agent_with_retry itself throws),
            # then we need a way to construct OutputStepData.
            # A pragmatic approach: if _process_output can't be called, we might not be able to form OutputStepData.
            # The current structure of _process_output implies it's called with *some* output.
            # Let's refine: if an error occurs in _run_agent_with_retry, agent_raw_output might be an exception or None.
            # _process_output should handle this. If error is in _prepare_prompt, then it's trickier.
            # For now, we will rely on _process_output to create the OutputStepData with error message.
            # If the exception happens before _process_output can be meaningfully called (e.g. prompt prep fails),
            # this step cannot produce its defined OutputStepData easily.
            # The prompt modification "return self._process_output(None, data)" for empty prompt is a start.
            # If e is the exception object, then:
            try:
                return self._process_output(e, data) # Pass exception as raw output
            except Exception as proc_e:
                 # If _process_output itself fails, then we have a problem creating OutputStepData.
                 # This path should ideally not be taken.
                 # A fallback would be to try to create a generic OutputStepData if possible,
                 # but OutputStepData is abstract.
                 # This highlights a need for robust error object creation in _process_output.
                 print(f"    Additionally, error during _process_output handling the original error: {proc_e}")
                 # Re-raise the original error if _process_output fails to handle it
                 raise e


class ContextStep(BaseAgentStep[BasePipelineData, ContextStepOutput]):
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
        self.current_date_for_step: str = "" # Initialize

    def _prepare_prompt(self, data: BasePipelineData) -> str:
        topic = data.topic
        if not topic:
            # This should ideally set data.error_message and return "" to be handled by BaseAgentStep.execute
            data.error_message = (data.error_message or "") + "; Topic not found for ContextStep."
            return "" # Empty prompt will be handled by execute
        self.current_date_for_step = datetime.now().strftime("%Y-%m-%d")
        return f"The current date is {self.current_date_for_step}. Please establish the factual context for the topic: '{topic}'."

    def _process_output(
        self, agent_raw_output: Any, data: BasePipelineData
    ) -> ContextStepOutput:
        current_date = getattr(self, "current_date_for_step", datetime.now().strftime("%Y-%m-%d"))
        error_msg = data.error_message or ""

        if isinstance(agent_raw_output, Exception):
            error_msg += f"; Error during agent execution: {str(agent_raw_output)}"
            return ContextStepOutput(
                topic=data.topic,
                current_date=current_date,
                context_report=ContextReport(verified_facts=[], summary="Error state"),
                context_facts_str="Error state",
                context_summary="Error state",
                error_message=error_msg.strip("; "),
            )

        if not isinstance(agent_raw_output, ContextReport):
            error_msg += f"; {self.agent_name} did not return a valid ContextReport. Output: {type(agent_raw_output)}"
            print(f"ERROR in ContextStep._process_output: {error_msg}")
            return ContextStepOutput(
                topic=data.topic,
                current_date=current_date,
                context_report=ContextReport(verified_facts=[], summary="Invalid output from agent"),
                context_facts_str="",
                context_summary="Invalid output from agent",
                error_message=error_msg.strip("; "),
            )

        context_facts_str = "\n".join(
            f"- {fact}" for fact in agent_raw_output.verified_facts
        )
        context_summary = agent_raw_output.summary
        print(f"  ContextStep processed. Summary: {context_summary[:100]}...")
        return ContextStepOutput(
            topic=data.topic,
            current_date=current_date,
            context_report=agent_raw_output,
            context_facts_str=context_facts_str,
            context_summary=context_summary,
            error_message=error_msg.strip("; ") or None, # Ensure None if empty
        )


class ArchitectStep(BaseAgentStep[ArchitectStepInput, ArchitectStepOutput]):
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

    def _prepare_prompt(self, data: ArchitectStepInput) -> str:
        if not all([data.topic, data.current_date, data.context_summary, data.context_facts_str]):
            data.error_message = (data.error_message or "") + "; Missing required fields for ArchitectStep prompt."
            return "" # Empty prompt
        return (
            f"**Established Context:**\n{data.context_summary}\n\n**Verified Facts:**\n{data.context_facts_str}\n\n"
            f"Based on the context above, and with the current date being {data.current_date}, "
            f"please design a Bayesian Network for the topic: '{data.topic}'.\n"
            "Your agent's core instructions (already provided via AgentConfig) contain details. Focus on uncertain events.\n\n"
            "IMPORTANT: For each node, the `states` field must be a JSON object (a dictionary), where each key is a state name (string) and its value is a brief description of that state (string)."
        )

    def _process_output(
        self, agent_raw_output: Any, data: ArchitectStepInput
    ) -> ArchitectStepOutput:
        error_msg = data.error_message or ""
        default_bn_structure = BNStructure(topic=data.topic, explanation="Error or invalid output", nodes={}, target_node_name="")

        if isinstance(agent_raw_output, Exception):
            error_msg += f"; Error during agent execution: {str(agent_raw_output)}"
            return ArchitectStepOutput(
                **data.model_dump(),
                bn_structure=default_bn_structure, # Provide a default BNStructure
                error_message=error_msg.strip("; "),
            )

        if not isinstance(agent_raw_output, BNStructure):
            error_msg += f"; {self.agent_name} did not return a valid BNStructure. Output: {type(agent_raw_output)}"
            print(f"ERROR in ArchitectStep._process_output: {error_msg}")
            return ArchitectStepOutput(
                **data.model_dump(),
                bn_structure=default_bn_structure,
                error_message=error_msg.strip("; "),
            )

        print(f"  ArchitectStep processed. BN for '{agent_raw_output.topic}'. Target: {agent_raw_output.target_node_name}")
        return ArchitectStepOutput(
            **data.model_dump(), # Pass through all fields from input
            bn_structure=agent_raw_output,
            error_message=error_msg.strip("; ") or None,
        )


def normalize_cpt(qualitative_cpt: QualitativeCpt) -> Dict[str, Dict[str, float]]:
    # This helper function remains unchanged internally.
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


class CPTGenerationStep(PipelineStep[CPTGenerationStepInput, CPTGenerationStepOutput]):
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

    async def _run_single_cpt_estimation( # Internal method, signature doesn't need InputStepData/OutputStepData
        self,
        node_name: str,
        node_obj: Node,
        bn_structure_topic: str,
        context_summary: str,
        context_facts_str: str,
        parent_nodes_info: List[Tuple[str, str, List[str]]],
        parent_state_combinations: List[Tuple[str, ...]],
    ) -> Optional[QualitativeCpt]:
        # ... (inner logic of _run_single_cpt_estimation remains largely the same)
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
        max_run_retries = 3
        if hasattr(self.agent_config, "max_retries") and self.agent_config.max_retries is not None: # type: ignore
            max_run_retries = self.agent_config.max_retries # type: ignore
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
                    # Return None to indicate failure after retries
                    print(f"  - Failed CPT for '{node_name}' due to invalid output type after {max_run_retries} attempts.")
                    return None
                current_prompt = (
                    initial_prompt + f"\n\nFAILED: {error_message} Review & retry."
                )
            except Exception as e:
                print(
                    f"  - CPT for '{node_name}': Error attempt {attempt + 1}: {type(e).__name__} - {e}"
                )
                if attempt == max_run_retries - 1:
                    # Return None or raise, here we return None to allow processing of other CPTs
                    print(f"  - Failed CPT for '{node_name}' due to exception after {max_run_retries} attempts: {e}")
                    return None # Propagate as a failure for this CPT
                current_prompt = (
                    initial_prompt
                    + f"\n\nFAILED (Error: {type(e).__name__} - {e}). Retry."
                )
        # Fallthrough if loop finishes without returning (e.g. max_retries is 0)
        print(f"  - Failed CPT for '{node_name}' after {max_run_retries} attempts (fallthrough).")
        return None

    async def execute(self, data: CPTGenerationStepInput) -> CPTGenerationStepOutput:
        print("Executing CPTGenerationStep...")
        current_error_message = data.error_message or ""

        if not data.bn_structure:
            error_msg = "BNStructure missing for CPTGenerationStep."
            current_error_message += f"; {error_msg}"
            # Return a CPTGenerationStepOutput with the error
            return CPTGenerationStepOutput(**data.model_dump(), error_message=current_error_message.strip("; "))

        if not all([data.context_summary, data.context_facts_str, data.topic]):
            error_msg = "Context summary, facts, or topic missing for CPTGenerationStep."
            current_error_message += f"; {error_msg}"
            # Return a CPTGenerationStepOutput with the error
            return CPTGenerationStepOutput(**data.model_dump(), error_message=current_error_message.strip("; "))

        bn_obj = data.bn_structure # Operate directly on the bn_structure from the input data
        tasks = []
        for name, node in bn_obj.nodes.items():
            parents_info = [
                (p.name, p.description, list(p.states.keys()))
                for p_name_str in node.parents # p_name is string
                if (p := bn_obj.nodes.get(p_name_str)) # get node object
            ]
            parent_states_lists = [info[2] for info in parents_info] # list of lists of states
            # Ensure state_combos gets [tuple()] for nodes without parents
            state_combos = list(itertools.product(*parent_states_lists)) if parent_states_lists else [tuple()]

            print(f"  - Task for CPT of node '{name}'... Combinations: {len(state_combos)}")
            tasks.append(
                asyncio.create_task(
                    self._run_single_cpt_estimation(
                        name,
                        node,
                        bn_obj.topic or data.topic, # Use bn_obj.topic first
                        data.context_summary,
                        data.context_facts_str,
                        parents_info,
                        state_combos,
                    ),
                    name=f"CPT_{name}",
                )
            )

        print(f"  - Waiting for {len(tasks)} CPT tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True) # Catch exceptions from tasks
        print("  - All CPT tasks completed.")

        successful_cpt_count = 0
        for i, res_or_exc in enumerate(results):
            # Original node list used for task creation determines the node name
            node_name = list(bn_obj.nodes.keys())[i]
            if isinstance(res_or_exc, QualitativeCpt):
                print(f"  - Processed CPT for '{node_name}'. Justification: {res_or_exc.justification[:50]}...")
                # Modify bn_obj (which is data.bn_structure) in place
                bn_obj.nodes[node_name].cpt = normalize_cpt(res_or_exc)
                successful_cpt_count += 1
            elif isinstance(res_or_exc, Exception): # Exception from gather
                error_detail = f"CPT generation task for {node_name} failed with exception: {type(res_or_exc).__name__} - {str(res_or_exc)}"
                print(f"  - ERROR: {error_detail}")
                current_error_message += f"; {error_detail}"
            else: # None or other unexpected result from _run_single_cpt_estimation
                error_detail = f"CPT generation for {node_name} failed to produce valid output or was None."
                print(f"  - ERROR: {error_detail}")
                current_error_message += f"; {error_detail}"

        if successful_cpt_count < len(bn_obj.nodes):
            summary_error = f"Only {successful_cpt_count}/{len(bn_obj.nodes)} CPTs successfully generated."
            current_error_message += f"; {summary_error}"
            print(f"  - WARNING: {summary_error}")

        # data.bn_structure was modified in-place.
        # Create CPTGenerationStepOutput by passing all fields from input data, error message updated.
        output_data = CPTGenerationStepOutput(
            **data.model_dump(),
            error_message=current_error_message.strip("; ") or None
        )
        print(f"CPTGenerationStep complete. {successful_cpt_count}/{len(bn_obj.nodes)} CPTs generated.")
        return output_data


# calculate_final_probability helper function remains the same.
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
    # Kahn's algorithm for topological sort
    queue_for_sort = list(q) # Use a copy for manipulation
    while queue_for_sort:
        u = queue_for_sort.pop(0)
        order.append(u)
        processed.add(u)
        # Iterate over all nodes to find children of u
        for v_name, v_node in bn.nodes.items():
            if u in v_node.parents: # u is a parent of v_name
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0 and v_name not in processed:
                    queue_for_sort.append(v_name)

    if len(order) != len(bn.nodes):
        # Handle cycles or disconnected components if necessary, or raise error
        # For now, extend with remaining nodes, though this might not be ideal for faulty BNs
        order.extend([n for n in bn.nodes if n not in order])


    memo = {}

    def enumerate_all_recursive(
        vars_list: List[str], current_evidence: Dict[str, str]
    ) -> float:
        vt, et = tuple(sorted(vars_list)), tuple(sorted(current_evidence.items()))
        if not vars_list: # Base case: no variables left to sum out
            return 1.0
        if (vt, et) in memo: # Memoization check
            return memo[(vt, et)]

        Y, rest_vars = vars_list[0], vars_list[1:]
        node_Y = bn.nodes.get(Y)

        if not node_Y: # Should not happen in a valid BN
            return enumerate_all_recursive(rest_vars, current_evidence)

        # Construct CPT key from parent states in current_evidence
        parent_states_values = []
        if node_Y.parents:
            for p_name in node_Y.parents:
                parent_states_values.append(current_evidence.get(p_name))
        # Key for CPT: string representation of tuple of parent states
        cpt_key = str(tuple(parent_states_values)) if node_Y.parents else "()"


        if Y in current_evidence: # If Y is part of evidence
            prob_y_given_parents = node_Y.cpt.get(cpt_key, {}).get(current_evidence[Y], 0.0)
            if prob_y_given_parents == 0.0: # Check for zero probability with evidence
                 # This might happen if evidence contradicts CPT.
                 # Depending on desired behavior, could return 0.0 or handle as error.
                 pass # Allow zero probability to propagate
            result_sum = prob_y_given_parents * enumerate_all_recursive(rest_vars, current_evidence)
        else: # Y needs to be summed out
            result_sum = 0.0
            if not node_Y.states: # Node has no states defined
                 # This is problematic; skip or assign uniform? For now, skip contribution.
                 result_sum = enumerate_all_recursive(rest_vars, current_evidence) # Effectively P(Y|parents)=1 for this path
            else:
                for y_state in node_Y.states.keys():
                    prob_y_state_given_parents = node_Y.cpt.get(cpt_key, {}).get(y_state, 0.0)
                    result_sum += prob_y_state_given_parents * enumerate_all_recursive(
                        rest_vars, {**current_evidence, Y: y_state}
                    )

        memo[(vt, et)] = result_sum
        return result_sum

    if target_node_name not in bn.nodes or not bn.nodes[target_node_name].states:
        return {} # Target node or its states not defined

    # Calculate probability for each state of the target node
    distribution = {}
    for target_state in bn.nodes[target_node_name].states.keys():
        # For each state of Y, P(Y=y_s, e) = P(Y=y_s | parents(Y), e_parents) * P(e_other_vars | Y=y_s, e_parents)
        # This should be P(Y=target_state | evidence) by calculating P(Y=target_state, evidence) / P(evidence)
        # enumerate_all_recursive calculates sum over all vars for P(vars_list | current_evidence)
        # So, P(target_state, evidence) = enumerate_all_recursive(order, {**evidence, target_node_name: target_state})
        distribution[target_state] = enumerate_all_recursive(
            order, {**evidence, target_node_name: target_state}
        )

    total_probability = sum(distribution.values())

    if total_probability == 0:
        # This can happen if evidence is contradictory or CPTs lead to zero paths.
        # Return uniform distribution over states or error.
        num_target_states = len(bn.nodes[target_node_name].states)
        return {s: 1.0 / num_target_states for s in bn.nodes[target_node_name].states} if num_target_states > 0 else {}

    # Normalize to get P(target_node | evidence)
    normalized_distribution = {
        state: prob / total_probability for state, prob in distribution.items()
    }
    return normalized_distribution


class FinalCalculationStep(PipelineStep[FinalCalculationStepInput, FinalCalculationStepOutput]):
    def __init__(self, evidence: Optional[Dict[str, str]] = None):
        self.evidence = evidence or {}
        print(
            f"Initialized FinalCalculationStep with evidence: {self.evidence if self.evidence else 'None'}"
        )

    async def execute(self, data: FinalCalculationStepInput) -> FinalCalculationStepOutput:
        print("Executing FinalCalculationStep...")
        current_error_message = data.error_message or ""
        final_probabilities: Optional[Dict[str, float]] = None
        final_probabilities_str: Optional[str] = "Not calculated."

        if not data.bn_structure:
            error_msg = "BNStructure not found for FinalCalculationStep."
            current_error_message += f"; {error_msg}"
            final_probabilities_str = error_msg
            return FinalCalculationStepOutput(
                **data.model_dump(),
                final_probabilities=None,
                final_probabilities_str=final_probabilities_str,
                error_message=current_error_message.strip("; ")
            )

        bn_obj = data.bn_structure
        target_name = bn_obj.target_node_name
        calculation_error_details = ""

        if not target_name:
            calculation_error_details = "No target node specified in BNStructure. Skipping calculation."
        elif target_name not in bn_obj.nodes:
            calculation_error_details = f"Target node '{target_name}' not found in BN nodes. Skipping calculation."
        else:
            print(f"  - Calculating Final Probability for Target Node: '{target_name}' with evidence: {self.evidence}")
            try:
                # Sanity check CPTs (optional, but good for debugging)
                for node_name_check, node_check in bn_obj.nodes.items():
                    if not node_check.cpt and (node_check.parents or not self.evidence.get(node_name_check)): # Root nodes not in evidence need CPTs
                        print(f"  - Warning: Node '{node_name_check}' might be missing CPTs or is an unevidenced root.")

                final_probabilities = calculate_final_probability(
                    bn=bn_obj, target_node_name=target_name, evidence=self.evidence
                )

                if final_probabilities:
                    final_probabilities_str = json.dumps(final_probabilities, indent=2)
                    print(f"  - Final Probabilities calculated: {final_probabilities_str}")
                else:
                    final_probabilities_str = "Calculation resulted in empty or None probabilities. This might indicate issues with CPTs or evidence consistency."
                    print(f"  - {final_probabilities_str}")

            except Exception as e:
                import traceback
                calculation_error_details = f"Error during final probability calculation: {type(e).__name__} - {e}. Trace: {traceback.format_exc()}"
                print(f"  - {calculation_error_details}")
                final_probabilities_str = f"Error: {str(e)}" # Keep it concise for the data field

        if calculation_error_details:
            current_error_message += f"; {calculation_error_details}"
            # final_probabilities_str might already be set to a specific error from calculation
            if final_probabilities_str == "Not calculated.":
                 final_probabilities_str = calculation_error_details # More specific error

        output_data = FinalCalculationStepOutput(
            **data.model_dump(),
            final_probabilities=final_probabilities,
            final_probabilities_str=final_probabilities_str,
            error_message=current_error_message.strip("; ") or None
        )
        print("FinalCalculationStep complete.")
        return output_data


class ForecasterStep(BaseAgentStep[ForecasterStepInput, ForecasterStepOutput]):
    def __init__(self, agent_config: AgentConfig, agent_name: str = "ForecasterAgent"):
        super().__init__(agent_config=agent_config, agent_name=agent_name)
        if self.agent_config.output_schema is not None:
            # This is a preference, not a strict error if the agent handles it.
            # print(f"Warning: {self.agent_name} initialized with AgentConfig that has an output_schema. Expected None for plain text.")
            pass

    def _prepare_prompt(self, data: ForecasterStepInput) -> str:
        missing_parts = []
        if not data.topic: missing_parts.append("topic")
        if not data.context_summary: missing_parts.append("context_summary")
        # bn_structure and final_probabilities_str can be None/error states, prompt should reflect that.

        if missing_parts:
            # Allow proceeding but record error. Prompt will use 'Not available'.
            data.error_message = (data.error_message or "") + f"; Missing critical data for ForecasterStep prompt: {', '.join(missing_parts)}"
            # Do not return "" here, let the prompt be formed with "Not available"

        final_bn_json_str = data.bn_structure.model_dump_json(indent=2) if data.bn_structure else "{}"
        target_node_name_str = data.bn_structure.target_node_name if data.bn_structure and data.bn_structure.target_node_name else "N/A"

        prompt = (
            f"**Established Context:**\n{data.context_summary or 'Not available'}\n\n"
            f"**Verified Facts:**\n{data.context_facts_str or 'Not available'}\n\n"
            f"Analyze the provided Bayesian Network and its CPTs for the topic: '{data.topic or 'Not available'}'.\n\n"
            "Your analysis must be based exclusively on the data below. Do not use outside knowledge. "
            "The context and facts provided above are the ground truth from previous steps and are incorporated into the network.\n"
            f"Based on the model, the calculated probability distribution for the target node '{target_node_name_str}' is:\n{data.final_probabilities_str or 'Not available/Error in calculation'}\n\n"
            "1. State the final probability distribution from the calculation above clearly at the beginning of your response.\n"
            "2. Provide a qualitative forecast, summarizing the most likely outcomes according to the model and the calculated probabilities.\n"
            "3. Identify the key drivers and relationships. Which nodes have the most influence on the final outcome? Refer to the CPTs in your explanation.\n"
            "4. Discuss any competing factors or notable sensitivities you observe in the model's structure and probabilities.\n"
            "Be firm and clear in your final summary, as it is the conclusion of a rigorous, evidence-based process.\n\n"
            f"Here is the complete Bayesian Network (JSON):\n{final_bn_json_str}"
        )
        return prompt

    def _process_output(
        self, agent_raw_output: Any, data: ForecasterStepInput
    ) -> ForecasterStepOutput:
        error_msg = data.error_message or ""
        forecaster_text_output: Optional[str] = None

        if isinstance(agent_raw_output, Exception):
            error_msg += f"; Error during agent execution: {str(agent_raw_output)}"
            forecaster_text_output = f"ForecasterAgent failed: {str(agent_raw_output)}"
        elif agent_raw_output is None:
            no_output_err = "ForecasterAgent did not produce any output after retries."
            error_msg += f"; {no_output_err}"
            forecaster_text_output = no_output_err
            print(f"ERROR in ForecasterStep._process_output: {forecaster_text_output}")
        else:
            forecaster_text_output = str(agent_raw_output)

        print(f"  ForecasterStep processed. Output length: {len(forecaster_text_output or '')}")
        return ForecasterStepOutput(
            **data.model_dump(),
            forecaster_agent_output=forecaster_text_output,
            error_message=error_msg.strip("; ") or None,
        )
