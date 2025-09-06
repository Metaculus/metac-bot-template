from __future__ import annotations

from typing import List, Sequence

from forecasting_tools import PredictedOptionList

from .constants import MC_PROB_MAX, MC_PROB_MIN
from .simple_types import OptionProbability


def _normalize_name(name: str) -> str:
    # Trim common prefixes like "Option X:" while preserving canonical names when matching
    stripped = name.strip()
    # Remove leading "Option" labels if present
    lowered = stripped.lower()
    if lowered.startswith("option ") or lowered.startswith("option:"):
        # drop leading token up to colon/space
        parts = stripped.split(":", 1)
        if len(parts) == 2:
            return parts[1].strip().lower()
        # fallback: remove first word
        return " ".join(stripped.split(" ")[1:]).strip().lower()
    return stripped.lower()


def build_mc_prediction(
    raw_options: Sequence[OptionProbability],
    allowed_options: Sequence[str],
) -> PredictedOptionList:
    """Convert loosely parsed MC options to a strict PredictedOptionList.

    - Filters to allowed option names (case-insensitive match against provided canonicals).
    - Aggregates duplicates by summing probabilities.
    - Renormalizes probabilities to sum to 1.0.
    - Applies clamping to [MC_PROB_MIN, MC_PROB_MAX] followed by a second renormalization.
    - Preserves the order of `allowed_options` in the final list.
    """
    # Map normalized allowed names to canonical
    allowed_norm_to_canonical = {opt.lower(): opt for opt in allowed_options}

    # Aggregate by canonical option name
    accum: dict[str, float] = {}
    for item in raw_options:
        norm = _normalize_name(item.option_name)
        if norm in allowed_norm_to_canonical:
            canonical = allowed_norm_to_canonical[norm]
            accum[canonical] = accum.get(canonical, 0.0) + float(item.probability)

    # Create list in allowed order, skipping truly missing options
    pairs: List[tuple[str, float]] = [(name, accum[name]) for name in allowed_options if name in accum]

    # If everything was filtered out, fall back to an even distribution over allowed options
    if not pairs and allowed_options:
        even = 1.0 / len(allowed_options)
        pairs = [(name, even) for name in allowed_options]

    # First renormalization (before constructing PredictedOptionList)
    total = sum(p for _, p in pairs)
    if total > 0:
        pairs = [(n, p / total) for n, p in pairs]

    # Construct PredictedOptionList (will validate sum close to 1.0)
    pol = PredictedOptionList(predicted_options=[{"option_name": n, "probability": p} for n, p in pairs])

    # Clamp and re-normalize to our configured bounds
    # Clamp
    for option in pol.predicted_options:
        option.probability = max(MC_PROB_MIN, min(MC_PROB_MAX, option.probability))
    # Normalize again
    total2 = sum(o.probability for o in pol.predicted_options)
    if total2 > 0:
        for option in pol.predicted_options:
            option.probability /= total2

    return pol


__all__ = ["build_mc_prediction"]
