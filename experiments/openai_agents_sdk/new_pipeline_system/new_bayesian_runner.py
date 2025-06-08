import asyncio
import json
import pathlib
import sys
from datetime import (
    datetime,
)
from typing import List, Dict, Any

# Add project root to sys.path to resolve imports when running script directly
project_root = str(pathlib.Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# --- Imports from the new pipeline system ---
from experiments.openai_agents_sdk.new_pipeline_system.pipeline_models import (
    BasePipelineData,  # Changed from PipelineData
    ForecasterStepOutput,  # Added for type hinting final_data
    ModelConfig,
    AgentConfig,
    ContextReport,
    BNStructure,
    QualitativeCpt,
    web_search,
)
from experiments.openai_agents_sdk.new_pipeline_system.pipeline_constants import (
    CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE,
    ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE,
    CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE,
    FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE,
)
from experiments.openai_agents_sdk.new_pipeline_system.pipeline_core import Pipeline
from experiments.openai_agents_sdk.new_pipeline_system.pipeline_steps import (
    ContextStep,
    ArchitectStep,
    CPTGenerationStep,
    FinalCalculationStep,
    ForecasterStep,
)

# --- Imports from agent framework & project root ---
try:
    from agent_sdk import MetaculusAsyncOpenAI
except ImportError as e:
    print(
        f"CRITICAL: Could not import MetaculusAsyncOpenAI from agent_sdk. "
        f"Ensure agent_sdk.py is in project root ('{project_root}'). Error: {e}"
    )
    sys.exit(1)

from agents import Agent as ActualAgentClass
from agents import AgentOutputSchema


# Simple Tee class for logging
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f_handle in self.files:
            f_handle.write(obj)
            f_handle.flush()

    def flush(self):
        for f_handle in self.files:
            f_handle.flush()


async def main_pipeline_run():
    forecasting_topic = "Will Donald Trump attend the NATO Summit in June 2025?"

    log_dir = pathlib.Path(__file__).parent / "runner_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"new_runner_{timestamp}.log"
    original_stdout = sys.stdout

    print(f"--- NEW BAYESIAN PIPELINE RUNNER ---")
    print(f"Forecasting Topic: {forecasting_topic}")
    print(f"Full logs will be saved to: {log_filename}")

    try:
        shared_openai_client = MetaculusAsyncOpenAI()
    except Exception as e:
        print(f"CRITICAL: Failed to initialize MetaculusAsyncOpenAI client: {e}")
        sys.exit(1)

    with open(log_filename, "w") as log_file_handle:
        sys.stdout = Tee(original_stdout, log_file_handle)

        try:
            default_model_cfg = ModelConfig(
                model_name="o4-mini",
                openai_client=shared_openai_client,
            )

            context_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[web_search],
                output_schema=AgentOutputSchema(
                    ContextReport, strict_json_schema=False
                ),
                instructions_template=CONTEXT_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            architect_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[web_search],
                output_schema=AgentOutputSchema(BNStructure, strict_json_schema=False),
                instructions_template=ARCHITECT_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            cpt_estimator_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[web_search],
                output_schema=AgentOutputSchema(
                    QualitativeCpt, strict_json_schema=False
                ),
                instructions_template=CPT_ESTIMATOR_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            forecaster_agent_config = AgentConfig(
                agent_class=ActualAgentClass,
                model_settings=default_model_cfg,
                tools=[],
                output_schema=None,
                instructions_template=FORECASTER_AGENT_INSTRUCTIONS_TEMPLATE,
            )

            context_step = ContextStep(agent_config=context_agent_config)
            architect_step = ArchitectStep(agent_config=architect_agent_config)
            cpt_step = CPTGenerationStep(agent_config=cpt_estimator_agent_config)
            calculation_step = FinalCalculationStep(evidence=None)
            forecaster_step = ForecasterStep(agent_config=forecaster_agent_config)

            pipeline = Pipeline(
                steps=[
                    context_step,
                    architect_step,
                    cpt_step,
                    calculation_step,
                    forecaster_step,
                ]
            )

            # Updated: Use BasePipelineData for initial data
            initial_pipeline_data = BasePipelineData(topic=forecasting_topic)

            print(f"\n--- RUNNING PIPELINE ---")
            # Added type hint for final_data for clarity
            final_data: ForecasterStepOutput = await pipeline.run(initial_pipeline_data)
            print(f"--- PIPELINE EXECUTION FINISHED ---")

            print("\n--- FINAL PIPELINE DATA ---")
            if final_data.error_message:
                print(f"Pipeline completed with errors: {final_data.error_message}")

            print(f"Topic: {final_data.topic}")

            if final_data.context_report:
                print(f"\nContext Summary: {final_data.context_report.summary}")
            else:
                print("\nContext Summary: Not available (ContextReport is None)")

            if final_data.bn_structure:
                print(f"\nBN Target Node: {final_data.bn_structure.target_node_name}")
            else:
                print("\nBN Target Node: Not available (BNStructure is None)")

            print(
                f"\nFinal Probabilities: {final_data.final_probabilities_str or 'Not calculated.'}"
            )

            if final_data.forecaster_agent_output:
                print(f"\nForecaster Output:\n{final_data.forecaster_agent_output}")
            else:
                print("\nForecaster Output: Not available.")

        except Exception as e:
            print(
                f"CRITICAL - Unhandled error in main_pipeline_run: {type(e).__name__} - {e}",
                file=sys.stderr,
            )
            import traceback

            traceback.print_exc(file=sys.stderr)
        finally:
            sys.stdout = original_stdout
            print(f"\nFull logs saved to {log_filename}")
            print(f"--- RUNNER FINISHED ---")


if __name__ == "__main__":
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_pipeline_run())
