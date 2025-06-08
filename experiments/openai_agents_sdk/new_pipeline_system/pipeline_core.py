import typing
import asyncio

# Assuming PipelineData is defined in .pipeline_models
# Adjust if the import path needs to be different based on execution context.
from .pipeline_models import PipelineData


@typing.runtime_checkable
class PipelineStep(typing.Protocol):
    """
    A protocol defining a single step in a processing pipeline.
    Each step in the pipeline should implement this protocol.
    """

    async def execute(self, data: PipelineData) -> PipelineData:
        """
        Executes the pipeline step.

        Args:
            data: A PipelineData object containing all data passed through
                  the pipeline, updated by previous steps.

        Returns:
            An updated PipelineData object containing the results of this step's
            execution, which will be passed to the next step.
        """
        ...

class Pipeline:
    """
    Manages and executes a sequence of pipeline steps using a typed PipelineData object.
    """

    def __init__(self, steps: typing.List[PipelineStep]):
        """
        Initializes the pipeline with a list of steps.

        Args:
            steps: A list of objects that conform to the PipelineStep protocol.
        """
        if not steps:
            raise ValueError("Pipeline must be initialized with at least one step.")
        self.steps = steps

    async def run(self, initial_data: typing.Union[PipelineData, dict]) -> PipelineData:
        """
        Runs the pipeline, executing each step in sequence with typed data.

        Args:
            initial_data: A PipelineData object or a dictionary that can be
                          parsed into PipelineData, containing the initial state.

        Returns:
            A PipelineData object containing the results from all pipeline steps.
        """
        if isinstance(initial_data, dict):
            try:
                current_data = PipelineData(**initial_data)
            except Exception as e: # Catch Pydantic validation errors or other issues
                print(f"Error converting initial_data dict to PipelineData: {e}")
                raise ValueError(f"Initial data dictionary is not compatible with PipelineData model: {e}") from e
        elif isinstance(initial_data, PipelineData):
            current_data = initial_data
        else:
            raise TypeError("initial_data must be a PipelineData instance or a compatible dict.")

        print(f"Starting pipeline with {len(self.steps)} steps. Initial PipelineData fields: {list(current_data.model_fields_set)}")

        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__
            print(f"Executing step {i + 1}/{len(self.steps)}: {step_name}...")
            try:
                current_data = await step.execute(current_data)
                if not isinstance(current_data, PipelineData):
                    # This is a more critical error now as steps are expected to return PipelineData
                    error_msg = f"Step {step_name} did not return a PipelineData object. Returned type: {type(current_data)}."
                    print(f"CRITICAL ERROR: {error_msg}")
                    # You might want to update current_data.error_message if possible or handle this severely.
                    raise TypeError(error_msg)
                # print(f"Step {step_name} completed. PipelineData fields: {list(current_data.model_fields_set)}") # Can be verbose
            except Exception as e:
                print(f"Error during execution of step {step_name}: {e}")
                # Update error message in PipelineData if the field exists
                if hasattr(current_data, 'error_message'):
                    if current_data.error_message:
                        current_data.error_message += f"; Error in {step_name}: {str(e)}"
                    else:
                        current_data.error_message = f"Error in {step_name}: {str(e)}"
                raise # Re-raise the exception to halt pipeline execution by default

        print("Pipeline execution finished.")
        return current_data
