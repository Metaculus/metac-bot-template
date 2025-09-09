# Spagbot 1.0: An AI Ensemble Forecaster

Date: 08 September 2025
Contact: kevin.wyjad@gmail.com
Written by GPT-5

---

## Spagbot: How It Works (A Clear, Non-Coder's Tour)

### 1) What Spagbot Is Trying to Do
Spagbot is a general-purpose forecasting bot designed to participate in Metaculus AI forecasting tournaments. For each active question, it:
* [cite_start]Gathers context to create a "research report"[cite: 38].
* [cite_start]Asks an ensemble of LLMs (ChatGPT, Claude, Gemini, Grok) to produce a forecast in a strict format[cite: 39].
* [cite_start]Optionally runs a game-theoretic bargaining simulation (GTMC1) for strategic topics like elections or conflicts[cite: 40].
* [cite_start]Combines all evidence using a Bayesian/Monte-Carlo aggregator to produce a final forecast[cite: 41].
* [cite_start]Generates logs and CSVs, and can submit the forecast to Metaculus if enabled[cite: 42].

[cite_start]Think of the process as: research brief -> multi-model panel -> (strategic simulation) -> Bayesian aggregator -> forecast + logs[cite: 43].

### 2) The Main Script That Orchestrates Everything
[cite_start]The main script is **spagbot.py** (the "conductor")[cite: 45]. [cite_start]When you run `python spagbot.py --mode test_questions`, it loads configurations and API keys, picks questions, and runs a pipeline for each one[cite: 47, 48, 49].

The pipeline per question includes:
* [cite_start]**A) Research brief**: An LLM generates a short, structured brief with reference classes, recent timeline items, and drivers[cite: 52]. [cite_start]Spagbot can use AskNews to pull recent articles or rely on general knowledge[cite: 53, 54]. [cite_start]It also appends a "Market Consensus Snapshot" showing the current community view from Metaculus and Manifold[cite: 54].
* **B) Panel forecasts (LLM ensemble)**: Spagbot queries the LLM panel. [cite_start]The models must adhere to a strict output format (e.g., `Final: ZZ%` for binary questions, or specific percentile lines for numeric questions)[cite: 56, 57, 58]. [cite_start]Only these final lines are parsed, not the free-text reasoning[cite: 59, 60].
* [cite_start]**C) Optional GTMC1 strategic simulation**: If the question's title contains keywords like "election" or "conflict," Spagbot asks an LLM to extract a table of actors[cite: 62]. [cite_start]This table is fed to GTMC1, a bargaining simulator that returns a probability-like signal[cite: 63]. [cite_start]This signal is then added as evidence, and the panel is re-prompted[cite: 64].
* **D) Bayesian / Monte-Carlo aggregator**: This "decider" layer combines the evidence. [cite_start]For binary questions, it uses a Beta prior/posterior[cite: 66]. [cite_start]For numeric questions, it fits a mixture of normals[cite: 68]. [cite_start]This approach treats each forecaster as a survey contributing fractional evidence and maintains disagreement[cite: 70].
* [cite_start]**E) Outputs**: The system creates several files, including `forecasts.csv` (final results), `forecasts_by_model.csv` (per-model outcomes), and a detailed log file `forecast_logs/<RUNID>_reasoning.log` which includes the research brief, GTMC1 details, and raw panel reasoning[cite: 72, 73, 75].

### 3) The Supporting Modules
* [cite_start]**bayes_mc.py**: The statistical combiner for panel outputs[cite: 78]. [cite_start]It handles Beta-Binomial for binary questions, Dirichlet-Multinomial for MCQ, and a mixture of normals for numeric questions[cite: 79, 81, 82].
* [cite_start]**GTMC1.py**: The game-theoretic Monte-Carlo simulator for strategic cases[cite: 84]. [cite_start]It takes an actor table and simulates repeated pairwise challenges, returning a probability signal and diagnostics[cite: 85, 86, 88].
* [cite_start]**update_calibration.py**: A helper script that reads `forecasts.csv`, fetches resolution data from Metaculus, computes calibration metrics (Brier, ECE, CRPS), and writes a memo to `data/calibration_advice.txt`[cite: 209, 210].
* [cite_start]**calibration_advice.txt**: A small text file containing lightweight guidance that is prefixed to every forecasting prompt to help the models correct biases based on past performance[cite: 89, 218, 224].

### 4) The Prompts
Prompts are designed to be long and structured, guiding the LLMs to use a specific thought process:
* [cite_start]Start with a base rate[cite: 93].
* [cite_start]Compare the case to the base rate[cite: 94].
* [cite_start]Evaluate evidence as likelihoods[cite: 95].
* [cite_start]Do a Bayesian update sketch[cite: 96].
* [cite_start]Red-team the conclusion[cite: 97].
* [cite_start]End with strict output lines that Spagbot can parse[cite: 98].

---

## Configuration & Environment
[cite_start]You can configure Spagbot by setting environment variables in your `.env` file[cite: 100, 111]. Key toggles and parameters include:
* [cite_start]`USE_OPENROUTER`, `USE_GOOGLE`, `ENABLE_GROK`, `SUBMIT_PREDICTION`[cite: 100].
* [cite_start]`SPAGBOT_DISABLE_RESEARCH_CACHE` to bypass caching[cite: 101].
* [cite_start]Model IDs like `OPENROUTER_GPT5_THINK_ID`, `GEMINI_MODEL`, `XAI_GROK_ID`[cite: 103].
* [cite_start]Timeouts for API calls[cite: 104].
* [cite_start]The `TOURNAMENT_ID` and `METACULUS_TOKEN`[cite: 105].
* [cite_start]`NUM_RUNS_PER_QUESTION` (default is 3)[cite: 108].

---

## Inputs and Outputs
### Inputs
* [cite_start]`.env`: API keys and toggles[cite: 111].
* [cite_start]`run_bot_on_*.yaml`: Run configurations[cite: 112].
* [cite_start]`pyproject.toml` / `poetry.lock`: Python dependencies[cite: 113].

### Core Code
* [cite_start]`spagbot.py`: The main orchestrator[cite: 115].
* [cite_start]`bayes_mc.py`: The statistical combiner[cite: 116].
* [cite_start]`GTMC1.py`: The bargaining simulator[cite: 117].
* [cite_start]`update_calibration.py`: The calibration helper[cite: 118].

### Artifacts (created on run)
* [cite_start]`forecasts.csv`: Final forecast per question[cite: 120].
* [cite_start]`forecasts_by_model.csv`: Per-model parse outcomes[cite: 121].
* [cite_start]`forecasts_mcq_wide.csv`: MCQ results in a fixed format[cite: 122].
* [cite_start]`forecast_logs/<RUNID>_reasoning.log`: Detailed log with research, GTMC1 info, and raw panel reasoning[cite: 123].
* [cite_start]`logs/spagbot_run_<timestamp>.txt`: Console log[cite: 124].
* [cite_start]`gtmc_logs/*`: Per-run GTMC1 data[cite: 125].

---

## How the Pieces Fit Together (Mental Model)
* [cite_start]Treat each LLM as a noisy expert speaking in a standard format[cite: 127].
* [cite_start]Treat the GTMC1 output as a clue for strategic cases, not the final word[cite: 128].
* [cite_start]Treat the Bayes-MC layer as a statistical fuse that uses weak priors, is conservative, and preserves uncertainty[cite: 129].

---

## What to Open First After a Run
1.  [cite_start]`forecast_logs/<RUNID>_reasoning.log`: Read the Research and GTMC1 sections[cite: 131].
2.  [cite_start]`forecasts.csv`: Check the final probabilities or percentiles[cite: 132].
3.  [cite_start]`forecasts_by_model.csv`: Check if any model failed to parse[cite: 133].
4.  [cite_start]`gtmc_logs/*_runs.csv`: For strategic questions, check the distribution of final medians[cite: 134].

---

## Calibration Loop
1.  [cite_start]Run Spagbot, which writes to `forecasts.csv`[cite: 222].
2.  [cite_start]After some questions have resolved, run `python update_calibration.py`[cite: 223]. [cite_start]This script computes calibration metrics and writes a memo to `data/calibration_advice.txt`[cite: 210, 223].
3.  [cite_start]On the next Spagbot run, the Calibration loader reads this memo and injects it into the prompts, nudging the panel to correct biases[cite: 224].