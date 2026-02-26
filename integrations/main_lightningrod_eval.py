"""
LightningRod SDK Evaluation Example

Demonstrates the full evaluation pipeline:
  news collection -> question generation -> labeling -> multi-model evaluation -> metrics

Usage:
    poetry install --with integrations
    poetry run python integrations/main_lightningrod_eval.py --max-questions 10

Requires LIGHTNINGROD_API_KEY in your .env file.
Sign up at https://dashboard.lightningrod.ai to get your API key and $50 of free credits.
"""

import argparse
from datetime import datetime, timedelta

from dotenv import load_dotenv
from lightningrod import (
    LightningRod,
    NewsSeedGenerator,
    ForwardLookingQuestionGenerator,
    WebSearchLabeler,
    QuestionPipeline,
    NewsContextGenerator,
    QuestionRenderer,
    RolloutGenerator,
    RolloutScorer,
    BinaryAnswerType,
    open_router_model,
)
from lightningrod.utils import compute_consensus, compute_metrics_summary

import os

def run_news_eval(lr, max_questions):
    """Configure and run the full news evaluation pipeline."""

    # Date range: ~3 months ago to ~1 month ago so questions can be resolved
    end_date = datetime.now() - timedelta(days=60)
    start_date = end_date - timedelta(days=120)

    seed_generator = NewsSeedGenerator(
        start_date=start_date,
        end_date=end_date,
        search_query="breaking news",
    )

    answer_type = BinaryAnswerType()

    question_generator = ForwardLookingQuestionGenerator(
        instructions=(
            "Generate questions about the provided article. "
            "Questions should be clearly resolvable within 1-2 months."
        ),
        answer_type=answer_type,
    )

    labeler = WebSearchLabeler(answer_type=answer_type)

    context_generator = NewsContextGenerator()

    renderer = QuestionRenderer(answer_type=answer_type)

    models = [
        open_router_model("openai/gpt-4.1-mini"),
        open_router_model("anthropic/claude-sonnet-4"),
        open_router_model("google/gemini-2.5-flash"),
    ]

    rollout_generator = RolloutGenerator(models=models)

    scorer = RolloutScorer(answer_type=answer_type)

    pipeline = QuestionPipeline(
        seed_generator=seed_generator,
        question_generator=question_generator,
        context_generators=[context_generator],
        labeler=labeler,
        renderer=renderer,
        rollout_generator=rollout_generator,
        scorer=scorer,
    )

    dataset = lr.transforms.run(
        pipeline, max_questions=max_questions, name="News Forecasting Benchmark"
    )

    return dataset.download()


def print_results(samples: list[Sample]):
    """Print summary, per-model metrics, and consensus analysis."""

    # --- Summary ---
    valid = sum(1 for s in samples if s.rollouts)
    print(f"\n{'='*60}")
    print(f"  Results: {len(samples)} samples, {valid} valid "
          f"({valid / len(samples) * 100:.0f}%)")
    print(f"{'='*60}\n")

    # --- Per-model metrics ---
    summary = compute_metrics_summary(samples)
    print("Per-model metrics:")
    print(f"  {'Model':<30} {'Mean Reward':>12} {'Parse Rate':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for model, metrics in summary.items():
        short = model.split("/")[-1] if "/" in model else model
        print(f"  {short:<30} {metrics['mean_reward']:>12.4f} {metrics['parse_rate']:>12.2%}")
    print()

    # --- Consensus analysis ---
    consensus = compute_consensus(samples)
    if not consensus:
        print("No consensus data available.\n")
        return

    n_agree = sum(1 for c in consensus if c["all_agree"])
    n_total = len(consensus)
    mean_spread = sum(c["spread"] for c in consensus) / n_total

    print("Consensus analysis:")
    print(f"  Agreement: {n_agree}/{n_total} questions ({n_agree / n_total * 100:.0f}%)")
    print(f"  Mean spread: {mean_spread:.3f}\n")

    print(f"  {'Question':<50} {'Label':>5} {'Spread':>7} {'Agree':>6}")
    print(f"  {'-'*50} {'-'*5} {'-'*7} {'-'*6}")
    for c in consensus:
        q = c["question_text"]
        if len(q) > 48:
            q = q[:48] + ".."
        label = str(c["label"])
        spread = f"{c['spread']:.3f}"
        agree = "Yes" if c["all_agree"] else "No"
        print(f"  {q:<50} {label:>5} {spread:>7} {agree:>6}")

        for model, prob in c["predictions"].items():
            short = model.split("/")[-1] if "/" in model else model
            print(f"    {short}: {prob:.3f}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightningRod evaluation example")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=20,
        help="Maximum number of questions to generate (default: 20)",
    )
    args = parser.parse_args()

    load_dotenv()
    lr = LightningRod(api_key=os.getenv("LIGHTNINGROD_API_KEY"))

    print(f"Running news evaluation with up to {args.max_questions} questions...\n")
    samples = run_news_eval(lr, args.max_questions)
    print_results(samples)
