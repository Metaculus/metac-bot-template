import asyncio
import json
import pathlib
import sys
from datetime import datetime # Not strictly needed by runner, but good for log timestamping if done here
from typing import List, Dict, Any # Standard typing imports

# --- Imports from the new pipeline system ---
from .pipeline_models import (
    PipelineData,
    ModelConfig,
    AgentConfig,
    # Specific Pydantic models for agent outputs (used for AgentOutputSchema)
    ContextReport,
    BNStructure,
    QualitativeCpt,
    # Tools like web_search, if AgentConfig expects the actual function
    web_search,
)
from .pipeline_constants import (
    CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE,
    ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE,
    CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE,
    FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE,
)
from .pipeline_core import Pipeline
from .pipeline_steps import (
    ContextStep,
    ArchitectStep,
    CPTGenerationStep,
    FinalCalculationStep,
    ForecasterStep,
)

# --- Imports from agent framework & project root ---
# Assuming agent_sdk.py is in the project root, 3 levels up from new_pipeline_system
try:
    from agent_sdk import MetaculusAsyncOpenAI
except ImportError:
    # Adjust path if the script is not run from a context where root is in PYTHONPATH
    project_root_for_sdk = str(pathlib.Path(__file__).resolve().parents[3])
    if project_root_for_sdk not in sys.path:
        sys.path.insert(0, project_root_for_sdk)
    try:
        from agent_sdk import MetaculusAsyncOpenAI
    except ImportError as e:
        print(f"CRITICAL: Could not import MetaculusAsyncOpenAI from agent_sdk. Ensure agent_sdk.py is in project root. Error: {e}")
        sys.exit(1)

from agents import Agent as ActualAgentClass # The actual agent class
from agents import AgentOutputSchema


# Simple Tee class for logging (can be moved to a shared util if used more widely)
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f_handle in self.files:
            f_handle.write(obj)
            f_handle.flush() # Ensure immediate flush
    def flush(self):
        for f_handle in self.files:
            f_handle.flush()

async def main_pipeline_run():
    forecasting_topic = "Will commercial quantum computing achieve 'quantum supremacy' for a practical problem (excluding research demonstrations) by the end of 2028?"

    # Setup logging (similar to original experiment)
    log_dir = pathlib.Path(__file__).parent / "runner_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"new_runner_{timestamp}.log"
    original_stdout = sys.stdout

    print(f"--- NEW BAYESIAN PIPELINE RUNNER ---")
    print(f"Forecasting Topic: {forecasting_topic}")
    print(f"Full logs will be saved to: {log_filename}")

    # Initialize the OpenAI client (or any other client MetaculusAsyncOpenAI wraps)
    # This client instance will be passed to ModelConfigs.
    try:
        shared_openai_client = MetaculusAsyncOpenAI()
        # Optional: Add LLM call counting patch if desired (as in original experiment)
        # This would require access to the global LLM_CALL_COUNT or similar mechanism.
        # For simplicity, omitting the patch here, but it can be added if metrics are needed.
    except Exception as e:
        print(f"CRITICAL: Failed to initialize MetaculusAsyncOpenAI client: {e}")
        sys.exit(1)

    with open(log_filename, "w") as log_file_handle:
        # Tee stdout to console and log file
        # Ensure Tee is robust or replace with standard logging if issues arise
        # For now, Tee should work if file handle is valid.
        # Check if Tee should be initialized outside and just write, or if it handles file opening.
        # Original Tee takes file objects, so this is correct.
        sys.stdout = Tee(original_stdout, log_file_handle)

        try:
            # 1. Configure Agent Settings
            default_model_cfg = ModelConfig(
                model_name="gpt-4o-mini", # Changed to gpt-4o-mini as o4-mini is not a model.
                openai_client=shared_openai_client
            )
            # Example: Using a more powerful model for the architect
            # powerful_model_cfg = ModelConfig(model_name="gpt-4o", openai_client=shared_openai_client)

            context_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[web_search], # web_search function from pipeline_models
                output_schema=AgentOutputSchema(ContextReport, strict_json_schema=False),
                instructions_template=CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            architect_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg, # Consider powerful_model_cfg for complex tasks
                tools=[web_search],
                output_schema=AgentOutputSchema(BNStructure, strict_json_schema=False),
                instructions_template=ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            cpt_estimator_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[web_search],
                output_schema=AgentOutputSchema(QualitativeCpt, strict_json_schema=False),
                instructions_template=CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            forecaster_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[], # No tools
                output_schema=None, # Plain text output
                instructions_template=FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            # 2. Instantiate Pipeline Steps
            context_step = ContextStep(agent_config=context_agent_config)
            architect_step = ArchitectStep(agent_config=architect_agent_config)
            cpt_step = CPTGenerationStep(agent_config=cpt_estimator_agent_config)
            calculation_step = FinalCalculationStep(evidence=None) # Optional: provide evidence
            forecaster_step = ForecasterStep(agent_config=forecaster_agent_config)

            # 3. Create the Pipeline
            pipeline = Pipeline(steps=[
                context_step,
                architect_step,
                cpt_step,
                calculation_step,
                forecaster_step,
            ])

            # 4. Define Initial PipelineData
            # Only 'topic' is mandatory for PipelineData to be valid initially.
            # Other fields are Optional and will be populated by the steps.
            initial_pipeline_data = PipelineData(topic=forecasting_topic)

            # 5. Run the Pipeline
            print(f"\n--- RUNNING PIPELINE ---")
            final_data = await pipeline.run(initial_pipeline_data) # Pass the Pydantic object
            print(f"--- PIPELINE EXECUTION FINISHED ---")

            # 6. Process and Display Results from final_data
            print("\n--- FINAL PIPELINE DATA ---")
            if final_data.error_message:
                print(f"Pipeline completed with errors: {final_data.error_message}")

            print(f"Topic: {final_data.topic}")

            if final_data.context_report:
                print(f"\nContext Summary: {final_data.context_report.summary}")

            if final_data.bn_structure:
                print(f"\nBN Target Node: {final_data.bn_structure.target_node_name}")
                # For full BN structure:
                # print(f"\nBN Structure (JSON):\n{final_data.bn_structure.model_dump_json(indent=2, exclude_none=True)}")

            print(f"\nFinal Probabilities: {final_data.final_probabilities_str or 'Not calculated.'}")

            if final_data.forecaster_agent_output:
                print(f"\nForecaster Output:\n{final_data.forecaster_agent_output}")

            # You can also dump the entire PipelineData object for debugging (can be verbose)
            # print(f"\nFull PipelineData (JSON):\n{final_data.model_dump_json(indent=2, exclude_none=True)}")

        except Exception as e:
            # This catches errors from pipeline.run() itself or unhandled ones from main_pipeline_run setup
            print(f"CRITICAL - Unhandled error in main_pipeline_run: {type(e).__name__} - {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        finally:
            sys.stdout = original_stdout # Restore stdout
            print(f"\nFull logs saved to {log_filename}")
            print(f"--- RUNNER FINISHED ---")


if __name__ == "__main__":
    # Ensure project root is in path if running this file directly for agent_sdk import
    # The try-except block for MetaculusAsyncOpenAI import already handles one level of this.

    # Python 3.8+ specific for Windows asyncio
    if sys.platform == "win32" and sys.version_info >= (3,8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_pipeline_run())
