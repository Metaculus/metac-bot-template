#!/usr/bin/env python3
"""
Standalone script to analyze correlations from existing benchmark results.

Usage:
    python analyze_correlations.py benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    python analyze_correlations.py benchmarks/ --max-cost 0.3 --max-size 3

Or via Makefile:
    make analyze_correlations FILE=benchmarks/benchmarks_2025-08-10_15-04-51.jsonl
    make analyze_correlations_latest
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot

from metaculus_bot.correlation_analysis import CorrelationAnalyzer
from metaculus_bot.scoring_patches import apply_scoring_patches

logger = logging.getLogger(__name__)


def extract_timestamp_from_filename(filepath: str) -> Optional[str]:
    """Extract timestamp from benchmark filename like 'benchmarks_2025-08-10_15-04-51.jsonl'"""
    filename = Path(filepath).name
    # Match pattern: benchmarks_YYYY-MM-DD_HH-MM-SS
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None


def load_benchmarks_from_path(benchmark_path: str) -> List[BenchmarkForBot]:
    """Load benchmark data from a file or directory."""
    path = Path(benchmark_path)
    benchmarks = []

    if path.is_file():
        # Single file - handle both .json and .jsonl
        try:
            with open(path, "r") as f:
                if path.suffix == ".jsonl":
                    # JSON Lines format - one benchmark per line
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            benchmark = BenchmarkForBot.model_validate(data)
                            benchmarks.append(benchmark)
                else:
                    # Regular JSON
                    data = json.load(f)
                    if isinstance(data, list):
                        for bench_data in data:
                            benchmark = BenchmarkForBot.model_validate(bench_data)
                            benchmarks.append(benchmark)
                    else:
                        benchmark = BenchmarkForBot.model_validate(data)
                        benchmarks.append(benchmark)
        except Exception as e:
            logger.error(f"Could not load {path}: {e}")
            return []

    elif path.is_dir():
        # Directory - load all .json and .jsonl files
        for pattern in ["*.json", "*.jsonl"]:
            for json_file in path.glob(pattern):
                if json_file.name.startswith("correlation_"):
                    continue  # Skip correlation analysis files
                benchmarks.extend(load_benchmarks_from_path(str(json_file)))

    else:
        logger.error(f"Path does not exist: {benchmark_path}")
        return []

    logger.info(f"Loaded {len(benchmarks)} benchmarks from {benchmark_path}")
    return benchmarks


def main():
    parser = argparse.ArgumentParser(description="Analyze model correlations from benchmark results")
    parser.add_argument("benchmark_path", help="Path to benchmark file (.json/.jsonl) or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for correlation report (default: correlation_analysis.md)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost per question for ensemble recommendations",
    )
    parser.add_argument("--max-size", type=int, default=7, help="Maximum ensemble size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--question-types",
        nargs="*",
        choices=["binary", "numeric", "multiple_choice"],
        help="Filter analysis to specific question types",
    )
    parser.add_argument(
        "--score-stats",
        dest="score_stats",
        action="store_true",
        default=True,
        help="Print score scaling stats by question type (default: on)",
    )
    parser.add_argument(
        "--no-score-stats",
        dest="score_stats",
        action="store_false",
        help="Disable printing score scaling stats",
    )
    parser.add_argument(
        "--score-stats-per-question",
        action="store_true",
        default=False,
        help="Also compute per-question stats (average across models per question)",
    )
    parser.add_argument(
        "--score-stats-json",
        type=str,
        default=None,
        help="Optional path to write score stats JSON (includes per-report and per-question if requested)",
    )
    parser.add_argument(
        "--model-stats-json",
        type=str,
        default=None,
        help="Optional path to write per-model, per-type score stats JSON",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help=(
            "Exclude models by substring match (case-insensitive). " "Example: --exclude-models grok-4 gemini-2.5-pro"
        ),
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help=(
            "Only include models matching these substrings (case-insensitive). "
            "Mutually exclusive with --exclude-models."
        ),
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load benchmarks
    try:
        benchmarks = load_benchmarks_from_path(args.benchmark_path)
    except Exception as e:
        logger.error(f"Failed to load benchmarks: {e}")
        sys.exit(1)

    if len(benchmarks) < 2:
        logger.error("Need at least 2 benchmark results for correlation analysis")
        sys.exit(1)

    # Apply scoring patches for mixed question types
    apply_scoring_patches()

    # Analysis-only: suppress noisy numeric fallback warnings from scoring_patches while counting them.
    class _NumericFallbackFilter(logging.Filter):
        def __init__(self) -> None:
            super().__init__()
            self.suppressed_lines = 0
            self.qids: set[str] = set()
            self._pat_qid1 = re.compile(r"Numeric Question (\d+)")
            self._pat_qid2 = re.compile(r"Numeric q=(\d+)")

        def filter(self, record: logging.LogRecord) -> bool:  # True keeps, False drops
            if record.name == "metaculus_bot.scoring_patches":
                msg = str(record.getMessage())
                low = msg.lower()
                if ("cannot compute model cdf" in low) or ("using percentile fallback" in low):
                    self.suppressed_lines += 1
                    m = self._pat_qid1.search(msg) or self._pat_qid2.search(msg)
                    if m:
                        self.qids.add(m.group(1))
                    return False
            return True

    _fallback_filter = _NumericFallbackFilter()
    logging.getLogger("metaculus_bot.scoring_patches").addFilter(_fallback_filter)

    # Perform analysis
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(benchmarks)

    # Apply include/exclude filtering before analysis
    if args.include_models and args.exclude_models:
        logger.error("--include-models and --exclude-models are mutually exclusive")
        sys.exit(2)

    filter_summary = analyzer.filter_models_inplace(include=args.include_models, exclude=args.exclude_models)
    if args.include_models or args.exclude_models:
        print("Applied model filters:")
        if args.include_models:
            print(f"  include tokens: {args.include_models}")
        if args.exclude_models:
            print(f"  exclude tokens: {args.exclude_models}")
        unmatched_inc = filter_summary.get("unmatched_includes", [])
        unmatched_exc = filter_summary.get("unmatched_excludes", [])
        if unmatched_inc:
            print(f"  unmatched include tokens: {unmatched_inc}")
        if unmatched_exc:
            print(f"  unmatched exclude tokens: {unmatched_exc}")

    # Ensure at least two models remain
    try:
        remaining_models = analyzer.get_model_names()  # type: ignore[attr-defined]
    except Exception:
        remaining_models = None

    if isinstance(remaining_models, (list, tuple, set)) and len(remaining_models) < 2:
        logger.error(
            f"Analysis requires ≥2 models after filtering. Remaining: {remaining_models if remaining_models else 'none'}"
        )
        sys.exit(1)

    # Score scaling stats after filters
    if args.score_stats:

        def _detect_type_from_report(rep) -> str:
            # Delegate to analyzer's helper which avoids touching .cdf
            try:
                return analyzer._get_question_type(rep)  # type: ignore[attr-defined]
            except Exception:
                return "binary"

        def _collect_scores_per_report(benches: List[BenchmarkForBot]) -> Dict[str, List[float]]:
            buckets: Dict[str, List[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
            for b in benches:
                for r in b.forecast_reports:
                    s = getattr(r, "expected_baseline_score", None)
                    if s is None:
                        continue
                    qtype = _detect_type_from_report(r)
                    buckets.setdefault(qtype, []).append(float(s))
            return buckets

        def _collect_scores_per_question(benches: List[BenchmarkForBot]) -> Dict[str, List[float]]:
            # Aggregate scores per (question_id, type) across models, then average
            per_q: Dict[Tuple[int, str], List[float]] = {}
            for b in benches:
                for r in b.forecast_reports:
                    s = getattr(r, "expected_baseline_score", None)
                    if s is None:
                        continue
                    qid = getattr(getattr(r, "question", None), "id_of_question", None)
                    if qid is None:
                        continue
                    qtype = _detect_type_from_report(r)
                    per_q.setdefault((int(qid), qtype), []).append(float(s))
            # Average within question, then bucket by type
            buckets: Dict[str, List[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
            for (qid, qtype), vals in per_q.items():
                if vals:
                    buckets.setdefault(qtype, []).append(float(np.mean(vals)))
            return buckets

        def _summarize(buckets: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for qtype, vals in buckets.items():
                if not vals:
                    out[qtype] = {
                        "n": 0,
                        "mean": float("nan"),
                        "mean_abs": float("nan"),
                        "min": float("nan"),
                        "max": float("nan"),
                    }
                else:
                    arr = np.array(vals, dtype=float)
                    out[qtype] = {
                        "n": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "mean_abs": float(np.mean(np.abs(arr))),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
            return out

        benches_filtered = getattr(analyzer, "benchmarks", benchmarks)
        per_report_buckets = _collect_scores_per_report(benches_filtered)
        per_report_summary = _summarize(per_report_buckets)

        print("\n" + "=" * 60)
        print("SCORE SCALING (After Filters) — Per-Report")
        print("=" * 60)
        for qtype in ["binary", "numeric", "multiple_choice"]:
            s = per_report_summary.get(qtype, {})
            print(
                f"{qtype:16} n={s.get('n',0):4d} | mean={s.get('mean', float('nan')):7.2f} | "
                f"mean|score|={s.get('mean_abs', float('nan')):7.2f} | min={s.get('min', float('nan')):7.2f} | max={s.get('max', float('nan')):7.2f}"
            )

        per_question_summary = None
        if args.score_stats_per_question:
            per_q_buckets = _collect_scores_per_question(benches_filtered)
            per_question_summary = _summarize(per_q_buckets)
            print("\nSCORE SCALING — Per-Question (average across models per question)")
            print("-" * 60)
            for qtype in ["binary", "numeric", "multiple_choice"]:
                s = per_question_summary.get(qtype, {})
                print(
                    f"{qtype:16} n={s.get('n',0):4d} | mean={s.get('mean', float('nan')):7.2f} | "
                    f"mean|score|={s.get('mean_abs', float('nan')):7.2f} | min={s.get('min', float('nan')):7.2f} | max={s.get('max', float('nan')):7.2f}"
                )

        # Optional JSON export
        if args.score_stats_json:
            try:
                import json as _json

                blob = {"per_report": per_report_summary}
                if per_question_summary is not None:
                    blob["per_question"] = per_question_summary
                with open(args.score_stats_json, "w") as f:
                    _json.dump(blob, f, indent=2)
                print(f"\nScore stats written to: {args.score_stats_json}")
            except Exception as e:
                logger.warning(f"Failed to write score stats JSON: {e}")

    # Per-model, per-type stats (basic per-report view)
    try:
        benches_filtered = getattr(analyzer, "benchmarks", benchmarks)

        def _is_stacking_bench(b) -> bool:
            try:
                return bool(analyzer._is_stacking_benchmark(b))  # type: ignore[attr-defined]
            except Exception:
                try:
                    cfg = getattr(b, "forecast_bot_config", {}) or {}
                    strat = cfg.get("aggregation_strategy")
                    if hasattr(strat, "value"):
                        strat = strat.value
                    return isinstance(strat, str) and strat.lower() == "stacking"
                except Exception:
                    return False

        def _model_name_for(b) -> str:
            try:
                return str(analyzer._extract_model_name(b))  # type: ignore[attr-defined]
            except Exception:
                return str(getattr(b, "name", "unknown"))

        buckets_mt: Dict[Tuple[str, str], List[float]] = {}
        for b in benches_filtered:
            if _is_stacking_bench(b):
                continue
            mname = _model_name_for(b)
            for r in b.forecast_reports:
                s = getattr(r, "expected_baseline_score", None)
                if s is None:
                    continue
                qtype = _detect_type_from_report(r)
                buckets_mt.setdefault((mname, qtype), []).append(float(s))

        def _summ_stats(vals: List[float]) -> Dict[str, float]:
            arr = np.array(vals, dtype=float)
            return {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "mean_abs": float(np.mean(np.abs(arr))),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        # Build model->type->stats mapping
        per_model: Dict[str, Dict[str, Dict[str, float]]] = {}
        for (mname, qtype), vals in buckets_mt.items():
            per_model.setdefault(mname, {})[qtype] = _summ_stats(vals)

        # Print compact breakdown
        if per_model:
            print("\n" + "=" * 60)
            print("MODEL STATS BY TYPE (After Filters)")
            print("=" * 60)
            for mname in sorted(per_model.keys()):
                print(f"\n{mname}")
                for qtype in ["binary", "numeric", "multiple_choice"]:
                    st = per_model[mname].get(qtype)
                    if not st or st.get("n", 0) == 0:
                        continue
                    print(
                        f"  {qtype:16} n={int(st['n']):4d} | mean={st['mean']:7.2f} | "
                        f"mean|score|={st['mean_abs']:7.2f} | min={st['min']:7.2f} | max={st['max']:7.2f}"
                    )

        # Optional JSON export
        if args.model_stats_json:
            try:
                import json as _json

                with open(args.model_stats_json, "w") as f:
                    _json.dump(per_model, f, indent=2)
                print(f"\nModel stats written to: {args.model_stats_json}")
            except Exception as e:
                logger.warning(f"Failed to write model stats JSON: {e}")
    except Exception as e:
        logger.warning(f"Failed to compute per-model stats: {e}")

    # Check if we have mixed question types
    has_mixed_types = analyzer._has_mixed_question_types()
    if has_mixed_types:
        logger.info("Detected mixed question types - using component-wise correlation analysis")
        type_breakdown = analyzer._get_question_type_breakdown()
        logger.info(f"Question type distribution: {type_breakdown}")
    else:
        logger.info("Using traditional correlation analysis for binary questions")

    # Generate report with timestamped filename
    if args.output:
        output_file = args.output
    else:
        # Default output location with timestamp from input file
        benchmark_path = Path(args.benchmark_path)
        timestamp = extract_timestamp_from_filename(args.benchmark_path)

        if timestamp:
            filename = f"correlation_analysis_{timestamp}.md"
        else:
            # Fallback to current timestamp if can't extract from input
            from datetime import datetime

            current_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"correlation_analysis_{current_timestamp}.md"

        if benchmark_path.is_file():
            output_file = benchmark_path.parent / filename
        else:
            output_file = benchmark_path / filename
    report = analyzer.generate_correlation_report(output_file)

    print("=" * 60)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 60)
    print(report)

    # Show top ensemble recommendations
    print("\n" + "=" * 60)
    print("ENSEMBLE RECOMMENDATIONS")
    print("=" * 60)

    optimal_ensembles = analyzer.find_optimal_ensembles(
        max_ensemble_size=args.max_size, max_cost_per_question=args.max_cost
    )

    if optimal_ensembles:
        print(f"\nTop 10 Ensembles (Both Aggregation Strategies, Cost ≤ ${args.max_cost}/question):")
        for i, ensemble in enumerate(optimal_ensembles[:10], 1):
            models = " + ".join(ensemble.model_names)
            print(f"{i:2}. {models} ({ensemble.aggregation_strategy.upper()})")
            print(
                f"    Score: {ensemble.avg_performance:.2f} | "
                f"Cost: ${ensemble.avg_cost:.3f} | "
                f"Diversity: {ensemble.diversity_score:.3f} | "
                f"Overall: {ensemble.ensemble_score:.3f}"
            )
        # Ablations and per-type diagnostics for top K
        # TODO: monolith needs refactor
        try:
            K = min(5, len(optimal_ensembles))
            print("\nENSEMBLE ABLATIONS (Top {} by Overall)".format(K))
            for idx in range(K):
                ens = optimal_ensembles[idx]
                base_models = list(ens.model_names)
                agg = ens.aggregation_strategy
                base_score = analyzer._simulate_ensemble_performance(base_models, agg)  # type: ignore[attr-defined]
                # Questions used baseline
                # Build per-model qid sets
                benches_filtered = getattr(analyzer, "benchmarks", [])

                def _model_name_for(b) -> str:
                    try:
                        return str(analyzer._extract_model_name(b))  # type: ignore[attr-defined]
                    except Exception:
                        return str(getattr(b, "name", "unknown"))

                qsets: Dict[str, set] = {}
                for b in benches_filtered:
                    m = _model_name_for(b)
                    if m not in base_models:
                        continue
                    s = qsets.setdefault(m, set())
                    for r in b.forecast_reports:
                        qid = getattr(getattr(r, "question", None), "id_of_question", None)
                        if isinstance(qid, int):
                            s.add(qid)
                inter_base = set.intersection(*(qsets[m] for m in base_models)) if base_models else set()
                print(
                    f"\n{idx+1}. {' + '.join(base_models)} ({agg.upper()})  | baseline={base_score:.2f} | Q={len(inter_base)}"
                )

                # Per-type split for baseline
                # Reuse helper from above (defined in ALL-MODEL block). If not present in scope, inline minimal.
                # We'll re-call the nested helper by reconstructing a minimal version here to avoid cross-scope issues.
                def _ensemble_per_type_local(models_list: List[str], agg_local: str) -> Dict[str, Dict[str, float]]:
                    from types import SimpleNamespace as _SNS

                    from metaculus_bot.scoring_patches import (
                        calculate_multiple_choice_baseline_score as _score_mc,
                    )
                    from metaculus_bot.scoring_patches import (
                        calculate_numeric_baseline_score as _score_num,
                    )

                    qmap: Dict[int, Dict[str, any]] = {}
                    for b in benches_filtered:
                        m = _model_name_for(b)
                        if m not in models_list:
                            continue
                        for r in b.forecast_reports:
                            qid = getattr(getattr(r, "question", None), "id_of_question", None)
                            if not isinstance(qid, int):
                                continue
                            qmap.setdefault(qid, {})[m] = r
                    stats: Dict[str, List[float]] = {"binary": [], "numeric": [], "multiple_choice": []}
                    for qid, m2r in qmap.items():
                        if any(m not in m2r for m in models_list):
                            continue
                        rep0 = next(iter(m2r.values()))
                        qtype = analyzer._get_question_type(rep0)  # type: ignore[attr-defined]
                        if qtype == "binary":
                            vals = [float(m2r[m].prediction) for m in models_list]
                            agg_p = float(np.mean(vals)) if agg_local == "mean" else float(np.median(vals))
                            c = getattr(rep0.question, "community_prediction_at_access_time", None)
                            if c is None:
                                continue
                            p = max(0.001, min(0.999, agg_p))
                            score = 100.0 * (c * (np.log2(p) + 1.0) + (1.0 - c) * (np.log2(1.0 - p) + 1.0))
                            stats[qtype].append(float(score))
                        elif qtype == "multiple_choice":
                            first = m2r[models_list[0]].prediction
                            if not hasattr(first, "predicted_options") or not first.predicted_options:
                                continue
                            option_names = [getattr(o, "option_name", str(o)) for o in first.predicted_options]
                            agg_probs: List[float] = []
                            for name in option_names:
                                vals = []
                                for m in models_list:
                                    pred = m2r[m].prediction
                                    for opt in pred.predicted_options:
                                        if getattr(opt, "option_name", str(opt)) == name:
                                            vals.append(float(getattr(opt, "probability", 0.0)))
                                            break
                                agg_probs.append(
                                    float(np.mean(vals)) if agg_local == "mean" else float(np.median(vals))
                                )
                            s = sum(agg_probs)
                            agg_probs = (
                                [p / s for p in agg_probs] if s > 0 else [1.0 / len(option_names)] * len(option_names)
                            )
                            pred_obj = _SNS(
                                predicted_options=[
                                    _SNS(option_name=n, probability=p) for n, p in zip(option_names, agg_probs)
                                ]
                            )
                            fake = _SNS(question=rep0.question, prediction=pred_obj)
                            sc = _score_mc(fake)
                            if sc is not None:
                                stats[qtype].append(float(sc))
                        elif qtype == "numeric":
                            cdfs: List[List[any]] = []
                            for m in models_list:
                                cdf = analyzer._get_safe_numeric_cdf(m, rep0.question, m2r[m].prediction)  # type: ignore[attr-defined]
                                if cdf is None:
                                    cdfs = []
                                    break
                                cdfs.append(cdf)
                            if not cdfs:
                                continue
                            L = min(len(c) for c in cdfs)
                            cdfs = [c[:L] for c in cdfs]
                            percs = np.array([[float(getattr(pt, "percentile", 0.0)) for pt in c] for c in cdfs])
                            agg_percs = percs.mean(axis=0) if agg_local == "mean" else np.median(percs, axis=0)
                            x = [float(getattr(pt, "value", i)) for i, pt in enumerate(cdfs[0][:L])]
                            pred_obj = _SNS(cdf=[_SNS(value=xi, percentile=float(pi)) for xi, pi in zip(x, agg_percs)])
                            fake = _SNS(question=rep0.question, prediction=pred_obj)
                            sc = _score_num(fake)
                            if sc is not None:
                                stats[qtype].append(float(sc))
                    out: Dict[str, Dict[str, float]] = {}
                    for qt, vals in stats.items():
                        if not vals:
                            continue
                        arr = np.array(vals, dtype=float)
                        out[qt] = {
                            "n": int(arr.size),
                            "mean": float(np.mean(arr)),
                            "mean_abs": float(np.mean(np.abs(arr))),
                            "min": float(np.min(arr)),
                            "max": float(np.max(arr)),
                        }
                    return out

                per_type = _ensemble_per_type_local(base_models, agg)
                if per_type:
                    print("  per-type:")
                    for qt in ["binary", "numeric", "multiple_choice"]:
                        st = per_type.get(qt)
                        if not st:
                            continue
                        print(
                            f"    {qt:16} n={int(st['n']):4d} | mean={st['mean']:7.2f} | mean|score|={st['mean_abs']:7.2f} | min={st['min']:7.2f} | max={st['max']:7.2f}"
                        )

                # Leave-one-out ablation
                contribs = []
                for m in base_models:
                    subset = [x for x in base_models if x != m]
                    score_wo = analyzer._simulate_ensemble_performance(subset, agg)  # type: ignore[attr-defined]
                    dq = len(set.intersection(*(qsets[x] for x in subset))) if subset else 0
                    delta = base_score - score_wo
                    contribs.append((m, score_wo, delta, dq))
                contribs.sort(key=lambda x: x[2], reverse=True)
                print("  leave-one-out impacts (Δscore):")
                for m, scwo, d, dq in contribs:
                    print(f"    - {m:20} Δ={d:+6.2f} | score_wo={scwo:6.2f} | Q={dq}")
        except Exception as e:
            logger.warning(f"Failed to compute ensemble ablations: {e}")
    else:
        print("No ensembles found meeting the cost constraint.")

    # Show correlation matrix highlights
    # Use appropriate correlation method based on question types
    if has_mixed_types:
        corr_matrix = analyzer.calculate_correlation_matrix_by_components()
    else:
        corr_matrix = analyzer.calculate_correlation_matrix()
    print(f"\n{'-' * 40}")
    print("CORRELATION HIGHLIGHTS")
    print(f"{'-' * 40}")

    least_correlated = corr_matrix.get_least_correlated_pairs(threshold=0.8)
    print("\nMost Independent Model Pairs:")
    for model1, model2, corr in least_correlated[:8]:
        print(f"  {model1:20} ↔ {model2:20} | r = {corr:6.3f}")

    # Also show most correlated pairs (by absolute r), excluding self and near-1.0
    try:
        pm = corr_matrix.pearson_matrix
        pairs = []
        names = list(pm.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                val = pm.iloc[i, j]
                if np.isnan(val):
                    continue
                if abs(val) >= 0.999:  # skip trivial self/near-identity
                    continue
                pairs.append((names[i], names[j], float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print("\nMost Correlated Model Pairs:")
        for model1, model2, corr in pairs[:8]:
            print(f"  {model1:20} ↔ {model2:20} | r = {corr:6.3f}")
    except Exception:
        pass

    print(f"\nDetailed report saved to: {output_file}")

    # Analysis-only summary for suppressed numeric fallback logs
    try:
        suppressed = getattr(_fallback_filter, "suppressed_lines", 0)
        qids = getattr(_fallback_filter, "qids", set())
        if suppressed:
            print(f"\n[analysis] Suppressed numeric fallback warnings: {suppressed} lines across {len(qids)} questions")
    except Exception:
        pass

    # ALL-MODEL ENSEMBLE (After Filters): compare mean/median across all remaining base models
    try:
        benches_filtered = getattr(analyzer, "benchmarks", benchmarks)

        # Build included base model list (exclude stacking)
        def _is_stacking_bench(b) -> bool:
            try:
                return bool(analyzer._is_stacking_benchmark(b))  # type: ignore[attr-defined]
            except Exception:
                try:
                    cfg = getattr(b, "forecast_bot_config", {}) or {}
                    strat = cfg.get("aggregation_strategy")
                    if hasattr(strat, "value"):
                        strat = strat.value
                    return isinstance(strat, str) and strat.lower() == "stacking"
                except Exception:
                    return False

        def _model_name_for(b) -> str:
            try:
                return str(analyzer._extract_model_name(b))  # type: ignore[attr-defined]
            except Exception:
                return str(getattr(b, "name", "unknown"))

        model_to_qids: Dict[str, set[int]] = {}
        models: List[str] = []
        for b in benches_filtered:
            if _is_stacking_bench(b):
                continue
            m = _model_name_for(b)
            if m not in model_to_qids:
                model_to_qids[m] = set()
                models.append(m)
            for r in b.forecast_reports:
                qid = getattr(getattr(r, "question", None), "id_of_question", None)
                if isinstance(qid, int):
                    model_to_qids[m].add(qid)

        # Filter to models that actually have any questions
        models = [m for m in models if model_to_qids.get(m)]

        if len(models) >= 2:
            # Questions used = intersection across all included models
            all_sets = [model_to_qids[m] for m in models]
            inter = set.intersection(*all_sets) if all_sets else set()
            uni = set.union(*all_sets) if all_sets else set()

            # Average cost per question from analyzer stats
            try:
                stats = analyzer._calculate_model_statistics()  # type: ignore[attr-defined]
            except Exception:
                stats = {}
            avg_costs = [stats[m]["avg_cost"] for m in models if m in stats]
            avg_cost = float(np.mean(avg_costs)) if avg_costs else float("nan")

            # Simulate performance for mean and median
            mean_score = analyzer._simulate_ensemble_performance(models, "mean")  # type: ignore[attr-defined]
            median_score = analyzer._simulate_ensemble_performance(models, "median")  # type: ignore[attr-defined]

            # Print summary
            def _short_list(names: List[str], max_len: int = 6) -> str:
                if len(names) <= max_len:
                    return ", ".join(names)
                return ", ".join(names[:max_len]) + f" … (+{len(names) - max_len} more)"

            print("\n" + "=" * 60)
            print("ALL-MODEL ENSEMBLE (After Filters)")
            print("=" * 60)
            print(f"Models included ({len(models)}): {_short_list(models)}")
            coverage = (len(inter) / max(len(uni), 1)) * 100.0 if uni else 0.0
            print(f"Questions used: {len(inter)} of {len(uni)} ({coverage:.1f}% coverage)")
            print(f"Avg cost per question: ${avg_cost:.3f}")
            print(f"MEAN   ensemble score: {mean_score:.2f}")
            print(f"MEDIAN ensemble score: {median_score:.2f}")

            # Per-type diagnostics for ALL-MODEL ensemble
            # TODO: monolith needs refactor
            def _ensemble_per_type(models_list: List[str], agg: str) -> Dict[str, Dict[str, float]]:
                from types import SimpleNamespace as _SNS

                from metaculus_bot.scoring_patches import (
                    calculate_multiple_choice_baseline_score as _score_mc,
                )
                from metaculus_bot.scoring_patches import (
                    calculate_numeric_baseline_score as _score_num,
                )

                # Build index: qid -> {model: report}
                qmap: Dict[int, Dict[str, any]] = {}
                for b in benches_filtered:
                    m = _model_name_for(b)
                    if m not in models_list:
                        continue
                    for r in b.forecast_reports:
                        qid = getattr(getattr(r, "question", None), "id_of_question", None)
                        if not isinstance(qid, int):
                            continue
                        qmap.setdefault(qid, {})[m] = r

                stats: Dict[str, List[float]] = {"binary": [], "numeric": [], "multiple_choice": []}

                for qid, m2r in qmap.items():
                    if any(m not in m2r for m in models_list):
                        continue  # need all models
                    rep0 = next(iter(m2r.values()))
                    qtype = _detect_type_from_report(rep0)
                    if qtype == "binary":
                        vals = [float(m2r[m].prediction) for m in models_list]
                        agg_p = float(np.mean(vals)) if agg == "mean" else float(np.median(vals))
                        # Score using binary baseline formula via analyzer's helper (or inline)
                        try:
                            # Reuse analyzer's internal baseline calc for binary
                            # Fallback: inline formula (identical to scoring patch)
                            c = getattr(rep0.question, "community_prediction_at_access_time", None)
                            if c is None:
                                continue
                            p = max(0.001, min(0.999, agg_p))
                            score = 100.0 * (c * (np.log2(p) + 1.0) + (1.0 - c) * (np.log2(1.0 - p) + 1.0))
                            stats[qtype].append(float(score))
                        except Exception:
                            continue

                    elif qtype == "multiple_choice":
                        first = m2r[models_list[0]].prediction
                        if not hasattr(first, "predicted_options") or not first.predicted_options:
                            continue
                        option_names = [getattr(o, "option_name", str(o)) for o in first.predicted_options]
                        agg_probs: List[float] = []
                        for name in option_names:
                            vals = []
                            for m in models_list:
                                pred = m2r[m].prediction
                                for opt in pred.predicted_options:
                                    if getattr(opt, "option_name", str(opt)) == name:
                                        vals.append(float(getattr(opt, "probability", 0.0)))
                                        break
                            if not vals:
                                agg_probs.append(0.0)
                            else:
                                agg_probs.append(float(np.mean(vals)) if agg == "mean" else float(np.median(vals)))
                        s = sum(agg_probs)
                        agg_probs = (
                            [p / s for p in agg_probs] if s > 0 else [1.0 / len(option_names)] * len(option_names)
                        )
                        pred_obj = _SNS(
                            predicted_options=[
                                _SNS(option_name=n, probability=p) for n, p in zip(option_names, agg_probs)
                            ]
                        )
                        fake = _SNS(question=rep0.question, prediction=pred_obj)
                        sc = _score_mc(fake)
                        if sc is not None:
                            stats[qtype].append(float(sc))

                    elif qtype == "numeric":
                        # Build aggregated CDF using analyzer's safe accessor
                        cdfs: List[List[any]] = []
                        for m in models_list:
                            cdf = analyzer._get_safe_numeric_cdf(m, rep0.question, m2r[m].prediction)  # type: ignore[attr-defined]
                            if cdf is None:
                                cdfs = []
                                break
                            cdfs.append(cdf)
                        if not cdfs:
                            continue
                        # Assume aligned lengths
                        L = min(len(c) for c in cdfs)
                        cdfs = [c[:L] for c in cdfs]
                        percs = np.array([[float(getattr(pt, "percentile", 0.0)) for pt in c] for c in cdfs])
                        agg_percs = percs.mean(axis=0) if agg == "mean" else np.median(percs, axis=0)
                        x = [float(getattr(pt, "value", i)) for i, pt in enumerate(cdfs[0][:L])]
                        pred_obj = _SNS(cdf=[_SNS(value=xi, percentile=float(pi)) for xi, pi in zip(x, agg_percs)])
                        fake = _SNS(question=rep0.question, prediction=pred_obj)
                        sc = _score_num(fake)
                        if sc is not None:
                            stats[qtype].append(float(sc))

                # Summarize
                out: Dict[str, Dict[str, float]] = {}
                for qt, vals in stats.items():
                    if not vals:
                        continue
                    arr = np.array(vals, dtype=float)
                    out[qt] = {
                        "n": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "mean_abs": float(np.mean(np.abs(arr))),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                    }
                return out

            for agg_name, score_val in [("MEAN", mean_score), ("MEDIAN", median_score)]:
                per_type = _ensemble_per_type(models, agg_name.lower())
                if per_type:
                    print(f"{agg_name} per-type:")
                    for qt in ["binary", "numeric", "multiple_choice"]:
                        st = per_type.get(qt)
                        if not st:
                            continue
                        print(
                            f"  {qt:16} n={int(st['n']):4d} | mean={st['mean']:7.2f} | mean|score|={st['mean_abs']:7.2f} | min={st['min']:7.2f} | max={st['max']:7.2f}"
                        )
        else:
            print("\nALL-MODEL ENSEMBLE: skipped (need ≥2 base models after filters)")
    except Exception as e:
        logger.warning(f"Failed to compute ALL-MODEL ensemble summary: {e}")


if __name__ == "__main__":
    main()
