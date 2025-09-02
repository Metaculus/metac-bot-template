from __future__ import annotations

"""
Configuration constants for numeric forecasting pipeline.

Extracted from main.py to centralize magic numbers and make them more maintainable.
These constants control various aspects of the numeric prediction processing pipeline.
"""

from typing import List

# --- Percentile Processing Constants ---

# Number of standard percentiles expected in a complete distribution
EXPECTED_PERCENTILE_COUNT: int = 8

# Standard percentile values that should be present in a complete forecast
STANDARD_PERCENTILES: List[float] = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95]

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
