# Real Ensemble Aggregation Implementation Plan

## Context & Problem Statement

The current ensemble evaluation in `correlation_analysis.py` is fundamentally flawed - it averages individual model baseline scores instead of actually aggregating predictions and scoring the ensemble predictions. This means we see identical performance for MEAN vs MEDIAN aggregation strategies, making the comparison pointless.

We need to implement **real ensemble prediction aggregation** where we:
1. Take actual individual model predictions for each question
2. Apply mean/median aggregation to create ensemble predictions  
3. Score those ensemble predictions against community predictions
4. Compare the actual performance differences between aggregation strategies

## Current Architecture (✅ What We Have)

### Community Benchmarking Framework
- **File**: `community_benchmark.py` 
- **Key principle**: "Run a benchmark that compares your forecasts against the community prediction" (line 174)
- **Community as ground truth**: Community predictions are treated as approximate ground truth for scoring
- **Scoring infrastructure**: `forecasting_tools.Benchmarker` already calculates baseline scores vs community predictions

### Data Availability  
- **Individual predictions**: `benchmark.forecast_reports[].prediction` (actual prediction objects)
- **Community predictions**: `benchmark.forecast_reports[].question.community_prediction`
- **Baseline scores**: `benchmark.forecast_reports[].expected_baseline_score` (already calculated)
- **Question context**: `benchmark.forecast_reports[].question` (for scoring context)

### Current Ensemble Infrastructure
- **File**: `metaculus_bot/correlation_analysis.py`
- **EnsembleCandidate**: Class with `aggregation_strategy` field ✅
- **find_optimal_ensembles()**: Tests both MEAN and MEDIAN for each model combination ✅
- **_evaluate_ensemble()**: Currently averages individual stats (❌ needs fix)
- **_simulate_ensemble_performance()**: Currently doesn't exist (❌ needs implementation)

## Implementation Plan

### Phase 1: Create Comprehensive Markdown Plan ✅
**Status**: Current task
- Document complete implementation strategy
- Identify all components and data flows
- Create reference for implementation

### Phase 2: Extract Real Prediction Data
**Target**: `_simulate_ensemble_performance()` method in `correlation_analysis.py`

**Current Issue**: Using `self.predictions` which only contains summary data (float values)
**Solution**: Use `benchmark.forecast_reports` which contains actual prediction objects

```python
def _simulate_ensemble_performance(self, models: List[str], aggregation_strategy: str) -> float:
    # Group data by question from benchmark reports
    question_data = {}
    
    for benchmark in self.benchmarks:
        model_name = self._extract_model_name(benchmark)
        if model_name in models:
            for report in benchmark.forecast_reports:
                q_id = report.question.id_of_question
                if q_id not in question_data:
                    question_data[q_id] = {
                        'individual_preds': {},
                        'community_pred': report.question.community_prediction,
                        'question': report.question  # For scoring context
                    }
                
                # Store actual prediction object (not just float)
                question_data[q_id]['individual_preds'][model_name] = report.prediction
```

### Phase 3: Implement Proper Aggregation by Question Type
**Target**: Handle binary, numeric, and multiple choice questions appropriately

**Binary Questions**: 
- Direct mean/median of prediction probabilities
- `ensemble_pred = np.mean([pred1, pred2, pred3])` or `np.median([...])`

**Numeric Questions**:
- Need to aggregate percentile distributions
- Options: (a) Extract median values and aggregate, (b) Aggregate full distributions
- Start with approach (a) using existing `_extract_prediction_value()`

**Multiple Choice Questions**:
- Aggregate probability distributions across options
- Ensure probabilities sum to 1.0 after aggregation

```python
# Extract prediction values for aggregation
pred_values = []
for model in models:
    pred_obj = data['individual_preds'][model]
    pred_values.append(self._extract_prediction_value(pred_obj))

# Apply aggregation strategy  
if aggregation_strategy == "mean":
    ensemble_pred_value = np.mean(pred_values)
elif aggregation_strategy == "median":
    ensemble_pred_value = np.median(pred_values)
```

### Phase 4: Find and Apply Real Baseline Scoring Function
**Target**: Use the same scoring function that `forecasting_tools.Benchmarker` uses

**Research needed**: 
- Find baseline scoring function in `forecasting_tools` 
- Understand signature: `calculate_baseline_score(prediction, community_prediction, question)`
- Import and use the same function

**Fallback approach**: 
- Reverse engineer from existing baseline scores in benchmark data
- Implement simplified baseline scoring if direct access not available

```python
# Calculate baseline score for ensemble prediction
ensemble_score = calculate_baseline_score(
    prediction_value=ensemble_pred_value,
    community_prediction=data['community_pred'],
    question=data['question']
)
```

### Phase 5: Replace Current Implementation
**Target**: Update `_evaluate_ensemble()` to use real ensemble performance

**Current code**:
```python
# WRONG: Averages individual model stats
avg_performance = np.mean([model_stats[m]["avg_performance"] for m in models])
```

**Fixed code**:
```python
# RIGHT: Simulates actual ensemble aggregation
ensemble_performance = self._simulate_ensemble_performance(models, aggregation_strategy)
```

### Phase 6: Test Real Aggregation Strategy Performance
**Target**: Verify we see actual performance differences

**Expected results**:
```
r1-0528 + glm-4.5-air (MEAN): Score 23.4
r1-0528 + glm-4.5-air (MEDIAN): Score 24.1  ← Different!
```

Instead of current (incorrect):
```
r1-0528 + glm-4.5-air (MEAN): Score 23.61
r1-0528 + glm-4.5-air (MEDIAN): Score 23.61  ← Identical!
```

## Technical Implementation Details

### Data Flow
1. **Input**: List of model names + aggregation strategy
2. **Process**: 
   - Extract predictions from `benchmark.forecast_reports`
   - Group by question ID
   - Apply aggregation strategy to get ensemble predictions
   - Score ensemble predictions against community predictions
   - Average ensemble scores across all questions
3. **Output**: Real ensemble performance score

### Question Type Handling Strategy
- **Phase 1**: Use existing `_extract_prediction_value()` for all question types
- **Phase 2**: Implement proper distribution aggregation for numeric/MC questions
- **Validation**: Ensure aggregated predictions make sense (probabilities ∈ [0,1], etc.)

### Error Handling
- **Missing predictions**: Skip questions where not all models have predictions
- **Invalid community predictions**: Skip questions without community predictions  
- **Scoring errors**: Log errors but continue with available data

### Performance Considerations
- **Caching**: Results should be cached since aggregation is expensive
- **Memory**: Avoid loading all prediction objects into memory simultaneously
- **Speed**: Focus on correctness first, optimize later

## Expected Outcomes

### Immediate Benefits
1. **Real performance differences** between mean and median aggregation
2. **Actionable insights** for ensemble tuning (which strategy works better)
3. **Proper ensemble evaluation** based on actual prediction aggregation

### Long-term Benefits  
1. **Foundation for advanced aggregation strategies** (weighted averages, etc.)
2. **Support for ensemble backtesting** against real outcomes
3. **Confidence in ensemble recommendations** based on real performance

### Validation Criteria
1. **Different scores** for mean vs median on same model combinations
2. **Reasonable score ranges** similar to individual model scores  
3. **Consistent behavior** across different ensemble sizes
4. **Logical relationships** (better individual models → better ensembles)

## Files to Modify

### Primary Changes
- **`metaculus_bot/correlation_analysis.py`**:
  - Update `_evaluate_ensemble()` method
  - Implement `_simulate_ensemble_performance()` method
  - Import baseline scoring function from forecasting_tools

### Supporting Changes
- **Tests**: Update tests to verify real aggregation behavior
- **Documentation**: Update docstrings to reflect real aggregation

### No Changes Needed
- **`community_benchmark.py`**: Already provides correct data ✅
- **`analyze_correlations.py`**: Output formatting already updated ✅  
- **`EnsembleCandidate`**: Already has aggregation strategy field ✅

## Success Metrics

1. **Functional**: Mean and median show different performance scores
2. **Logical**: Ensemble scores are reasonable relative to individual model scores  
3. **Actionable**: Clear recommendations on which aggregation strategy to use
4. **Scalable**: Implementation works for different ensemble sizes and model combinations

This implementation will provide the real aggregation strategy comparison needed for effective ensemble tuning.