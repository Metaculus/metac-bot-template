"""
Configuration constants for numeric forecasting pipeline.

Extracted from main.py to centralize magic numbers and make them more maintainable.
These constants control various aspects of the numeric prediction processing pipeline.
"""

from typing import List

# --- Percentile Processing Constants ---

# Number of standard percentiles expected in a complete distribution
EXPECTED_PERCENTILE_COUNT: int = 11

# Standard percentile values that should be present in a complete forecast
# Expressed as decimals in [0,1]
STANDARD_PERCENTILES: List[float] = [
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.50,
    0.60,
    0.80,
    0.90,
    0.95,
    0.975,
]

# Minimum required number of percentiles to proceed with processing
MIN_PERCENTILES_REQUIRED: int = 3

# --- PCHIP CDF Configuration ---

# Number of points to generate in the PCHIP CDF
PCHIP_CDF_POINTS: int = 201

# Minimum acceptable step size in probability space for CDF ramp smoothing
MIN_CDF_PROB_STEP: float = 5e-5

# Maximum allowable step size in probability space
MAX_CDF_PROB_STEP: float = 0.59

# Smoothing factor for CDF ramp correction (higher = more aggressive smoothing)
CDF_RAMP_K_FACTOR: float = 3.0

# --- Cluster Detection and Spreading Constants ---

# Minimum relative tolerance for detecting clusters (values considered identical)
CLUSTER_DETECTION_RTOL: float = 1e-9

# Absolute tolerance for detecting clusters
CLUSTER_DETECTION_ATOL: float = 1e-12

# Base delta multiplier for spreading clustered values
CLUSTER_SPREAD_BASE_DELTA: float = 1e-6

# Threshold for detecting "count-like" distributions (integer-adjacent values)
COUNT_LIKE_THRESHOLD: float = 0.1

# Enhanced delta multiplier for count-like distributions
COUNT_LIKE_DELTA_MULTIPLIER: float = 1.0

# --- Jitter and Validation Constants ---

# Small epsilon for ensuring strict ordering of percentile values
STRICT_ORDERING_EPSILON: float = 1e-12

# Maximum number of iterations for iterative jitter application
MAX_JITTER_ITERATIONS: int = 10

# Convergence tolerance for jitter application
JITTER_CONVERGENCE_TOL: float = 1e-10

# --- Boundary Handling Constants ---

# Safety margin when clamping values near bounds (as fraction of range)
BOUNDARY_SAFETY_MARGIN: float = 0.01

# Minimum distance from boundary to avoid numerical issues
MIN_BOUNDARY_DISTANCE: float = 1e-9

# --- Diagnostic and Logging Thresholds ---

# Threshold for logging warnings about large corrections
LARGE_CORRECTION_THRESHOLD: float = 0.1

# Maximum number of percentiles to show in diagnostic messages
MAX_DIAGNOSTIC_PERCENTILES: int = 5

# Threshold for detecting and warning about extreme probability steps
EXTREME_STEP_THRESHOLD: float = 0.5

# --- PCHIP Fallback Configuration ---

# Maximum number of PCHIP generation attempts before falling back
MAX_PCHIP_ATTEMPTS: int = 3

# Backoff multiplier for PCHIP retry delays
PCHIP_RETRY_BACKOFF: float = 1.5

# Base retry delay in seconds
PCHIP_BASE_RETRY_DELAY: float = 0.1

# --- Validation Tolerances ---

# Tolerance for validating percentile ordering
PERCENTILE_ORDER_TOLERANCE: float = 1e-10

# Tolerance for bound validation
BOUND_VALIDATION_TOLERANCE: float = 1e-8

# Maximum relative error allowed in percentile values
MAX_PERCENTILE_RELATIVE_ERROR: float = 1e-6

# --- Tail Widening (disabled by default) ---

# Enable/disable transform-space tail widening of declared percentiles before CDF generation
TAIL_WIDENING_ENABLE: bool = True

# Tail widening stretch factor applied in transformed space around the median in tails
# e.g., 1.25 means 25% stretch at deepest tails, ramping to 0% near the center
TAIL_WIDEN_K_TAIL: float = 1.25

# Tail start region (fraction of percentile distance from median where widening begins)
# Example: 0.2 means no widening for p in [0.3, 0.7], linearly ramp to full widening by p<=0.1 or p>=0.9
TAIL_WIDEN_TAIL_START: float = 0.2

# Span floor gamma to ensure tail spans are at least gamma times adjacent inner spans
# Applies to (p05 - p02.5) vs (p10 - p05) and (p97.5 - p95) vs (p95 - p90)
TAIL_WIDEN_SPAN_FLOOR_GAMMA: float = 1.0
