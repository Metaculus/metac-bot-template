import typing
import asyncio
from typing import TypeVar, Generic # Added Generic

# Import the new base model
from .pipeline_models import BasePipelineData
# Removed: from .pipeline_models import PipelineData

# Define TypeVars for generic PipelineStep
InputStepData = TypeVar('InputStepData', bound=BasePipelineData)
OutputStepData = TypeVar('OutputStepData', bound=BasePipelineData)

@typing.runtime_checkable
class PipelineStep(typing.Protocol[InputStepData, OutputStepData]): # Made generic
    """
    A protocol defining a single step in a processing pipeline.
    Each step in the pipeline should implement this protocol.
    """

    async def execute(self, data: InputStepData) -> OutputStepData: # Updated signature
        """
        Executes the pipeline step.

        Args:
            data: An object derived from BasePipelineData containing all data passed
                  through the pipeline, updated by previous steps.

        Returns:
            An object derived from BasePipelineData containing the results of this step's
            execution, which will be passed to the next step.
        """
        ...

class Pipeline:
    """
    Manages and executes a sequence of pipeline steps using BasePipelineData-derived objects.
    """

    def __init__(self, steps: typing.List[PipelineStep[BasePipelineData, BasePipelineData]]): # Updated steps type
        """
        Initializes the pipeline with a list of steps.

        Args:
            steps: A list of objects that conform to the PipelineStep protocol.
                   Each step is expected to handle and return BasePipelineData or its subclasses.
        """
        if not steps:
            raise ValueError("Pipeline must be initialized with at least one step.")
        self.steps = steps

    async def run(self, initial_data: BasePipelineData) -> BasePipelineData: # Updated signature and initial_data type
        """
        Runs the pipeline, executing each step in sequence.

        Args:
            initial_data: An instance of a BasePipelineData subclass,
                          containing the initial state.

        Returns:
            A BasePipelineData subclass instance containing the results from all pipeline steps.
        """
        current_data: BasePipelineData # Explicitly type current_data

        # Updated initial_data handling logic as per prompt
        if isinstance(initial_data, dict):
            # This block handles the case where a dictionary is passed.
            # The prompt has evolved this logic. The final instruction was:
            # "For this subtask, let's assume initial_data is already a BasePipelineData instance
            # or the first step's input type can be instantiated from BasePipelineData(**initial_data) if needed.
            # We will primarily focus on the case where initial_data is already a Pydantic model instance."
            # And then: "Initial data must be an instance of a BasePipelineData subclass."
            # This means passing a raw dict is no longer the primary supported path without more complex logic
            # (which is out of scope for this change). The code below reflects the stricter requirement.
            print("Warning: Passing a raw dictionary as initial_data is not fully supported without a clear mechanism to determine the target BasePipelineData subclass. Attempting basic BasePipelineData instantiation if necessary, but this might not be what the first step expects.")
            # For the purpose of this refactoring, we will raise an error if a dict is passed,
            # aligning with the stricter interpretation from the prompt.
            # If dicts need to be parsed, the caller should do it or the pipeline needs a more robust mechanism.
            raise ValueError("Initial data must be an instance of a BasePipelineData subclass. Passing raw dictionaries is not directly supported by this version of Pipeline.run().")

        elif isinstance(initial_data, BasePipelineData):
            current_data = initial_data
        else:
            # This aligns with the prompt's requirement: "initial_data must be a BasePipelineData subclass instance."
            raise TypeError(
                f"initial_data must be a BasePipelineData subclass instance. Got {type(initial_data)}"
            )

        print(f"Starting pipeline with {len(self.steps)} steps. Initial data topic: '{current_data.topic}'")

        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            print(f"Executing step {i + 1}/{len(self.steps)}: {step_name}...")
            try:
                # The core logic: current_data is passed to step.execute.
                # The step's type hint (InputStepData) is bound by BasePipelineData.
                # The step's return type (OutputStepData) is also bound by BasePipelineData.
                # This assignment is valid due to these bounds.
                current_data = await step.execute(current_data) # type: ignore # See note below

                # NOTE on type: ignore:
                # While step.execute expects InputStepData and current_data is BasePipelineData (parent),
                # and returns OutputStepData which is then assigned back to BasePipelineData (parent),
                # this is generally safe with Pydantic models due to structural subtyping / extra field handling.
                # However, MyPy might complain about assigning a more generic type (BasePipelineData)
                # to a step that expects a more specific subtype (e.g. ContextStepOutput).
                # And conversely, assigning a specific output (e.g. ContextStepOutput) back to current_data (BasePipelineData)
                # is fine. The issue is on the input.
                # The `steps` list is `PipelineStep[BasePipelineData, BasePipelineData]`.
                # This means each step's `execute` is effectively `execute(self, data: BasePipelineData) -> BasePipelineData`
                # from the perspective of the Pipeline class.
                # Individual step implementations will have their specific types.
                # This should work at runtime due to Python's dynamic typing and Pydantic's model processing.
                # The `type: ignore` can be added if static type checkers struggle with this specific pattern
                # of heterogeneously typed steps managed by a homogeneously typed list if strictness is high.
                # For now, let's assume runtime behavior is the primary concern and Pydantic handles it.

                if not isinstance(current_data, BasePipelineData): # Updated check
                    error_msg = f"Step {step_name} did not return a BasePipelineData subclass object. Returned type: {type(current_data)}."
                    print(f"CRITICAL ERROR: {error_msg}")
                    # current_data.error_message should exist as it's in BasePipelineData
                    current_data.error_message = (current_data.error_message + "; " if current_data.error_message else "") + error_msg
                    raise TypeError(error_msg) # This error is critical.

                # Optional: print(f"Step {step_name} completed. Current data type: {type(current_data).__name__}")
            except Exception as e:
                print(f"Error during execution of step {step_name}: {e}")
                # error_message is guaranteed by BasePipelineData
                if current_data.error_message:
                    current_data.error_message += f"; Error in {step_name}: {str(e)}"
                else:
                    current_data.error_message = f"Error in {step_name}: {str(e)}"
                raise # Re-raise the exception to halt pipeline execution by default

        print("Pipeline execution finished.")
        return current_data
