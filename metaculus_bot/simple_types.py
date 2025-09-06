from __future__ import annotations

from pydantic import BaseModel


class OptionProbability(BaseModel):
    """Lightweight schema for MC parsing before strict validation.

    Parsed first to allow clamping/renormalization before constructing
    forecasting_tools PredictedOptionList (which enforces strict sum checks).
    """

    option_name: str
    probability: float


__all__ = ["OptionProbability"]
