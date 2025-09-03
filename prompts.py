# Prompts for the Metaculus forecasting bot

SEARCH_QUERIES_PROMPT = """
You are a research assistant helping a forecaster find relevant information.
Given the forecasting question below, you need to:

1. Generate {num_queries} different search queries that would help find the most relevant information. Your queries are processed by classical search engines, so please phrase the queries in a way optimal for keyword optimized search (i.e., the phrase you search is likely to appear on desired web pages). Avoid writing overly specific queries. Limit to six words.
2. Suggest the most appropriate start date for searching (in ISO format YYYY-MM-DDTHH:MM:SS.sssZ), using a nuanced and context-aware assessment of the topic domain

Guidelines for search queries:
- Make each query highly specific, targeted, and distinct from one another
- Focus on the most recent and credible developments, unless deep historical context is necessary for understanding the present
- Cover multiple perspectives, angles, or related subtopics for a comprehensive approach
- Ensure queries are optimized for surfacing accurate, newsworthy, and fact-based content

Guidelines for estimating the start date:
- Carefully determine how far back relevant and impactful information might exist, with special attention to the domain and forecasting goal:
   - For political events: consider both recent trends and historically significant events that could provide valuable context or precedent (often 6-12 months, but much further if warranted)
   - For technology or science topics: focus on developments from the last 2-3 years, unless the historical trajectory or origins are essential
   - For other domains, select a range that best supports robust forecast-relevant insight, considering how older information might shape outcomes
- Incorporate the question's anticipated resolution timeframe, and use domain knowledge to justify and support the estimated start date

Question: {question}

Format your response exactly as follows:
START_DATE: YYYY-MM-DDTHH:MM:SS.sssZ
QUERY_1: [first search query]
QUERY_2: [second search query]
QUERY_3: [third search query]
"""

BINARY_PROMPT_TEMPLATE = """
# Role and Objective
- You are a professional forecaster interviewing for a job. Your task is to answer a forecasting interview question with structured reasoning and a clear probability estimate.

# Plan
- Begin with a concise checklist (3-7 bullets) of the major reasoning steps you will follow before presenting your answer. Do not include this checklist in your final response.
- Identify key factors that will influence the outcome.
- Consider potential scenarios and their likelihood.
- Use data and evidence to support your reasoning.

# Instructions
- Review the interview question, background, and resolution criteria.
- Reference your research assistant's summary report.
- Record today's date for context.
- Systematically consider and write:
  1. The time left until the outcome to the question is known.
  2. The status quo outcome if nothing changes.
  3. A brief scenario resulting in a No outcome.
  4. A brief scenario resulting in a Yes outcome.
- In your rationale, put extra weight on the status quo outcome, acknowledging that change is usually gradual.
- After completing each reasoning step, briefly validate whether each point is supported by the background information provided.
- Conclude your answer by stating your probability estimate in the format: `Probability: ZZ%` (between 0 and 100).

## Output Format
- Provide your reasoning and probability as specified above.
- Use clear and concise language. Structure your response with bullet points or short paragraphs as appropriate.

# Context
### Interview Question:  
`{title}`

### Background:  
`{background}`

### Outcome Determination
- The outcome is determined by the following criteria (not yet satisfied):
  `{resolution_criteria}`
- Additional details:
  `{fine_print}`

### Research Assistant Summary
`{summary_report}`

### Today's Date
`{today}`

# Verbosity
- Be concise and structured in all responses.

# Stop Conditions
- Once you have written all required reasoning steps and your final probability estimate, conclude your response.
"""

NUMERIC_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Begin with a concise checklist (3-7 bullets) outlining your approach to the forecasting task, keeping items conceptual rather than implementation-specific.

You are presented with the following interview question:
{title}

Background Information:
{background}

Resolution Criteria:
{resolution_criteria}

Additional Details:
{fine_print}

Answer Units: {units}

Assistant's Research Summary:
{summary_report}

Current Date: {today}

{lower_bound_message}
{upper_bound_message}

Formatting Guidelines:
- Pay careful attention to the units required (e.g., present as 1,000,000 or 1m as specified).
- Do not use scientific notation.
- Always list values in ascending order (starting with the smallest or most negative, if applicable).

Before providing your forecast, write:
(a) The time remaining until the question's outcome will be known.
(b) The expected outcome if conditions remain unchanged.
(c) The expected outcome if the current trend persists.
(d) The prevailing expectations of relevant experts and markets.
(e) A concise description of an unexpected scenario resulting in a low outcome.
(f) A concise description of an unexpected scenario resulting in a high outcome.

Remind yourself to adopt humility and set broad 90/10 confidence intervals to capture unknown unknowns, as good forecasters do.

The last thing you write is your final answer exactly as follows without any decoration and replacing XX with actual numbers without units:
"
Percentile 10: XX
Percentile 20: XX
Percentile 40: XX
Percentile 60: XX
Percentile 80: XX
Percentile 90: XX
"
"""

MULTIPLE_CHOICE_PROMPT_TEMPLATE = """
You are a professional forecaster interviewing for a job.

Your interview question is:
{title}

The options are: {options}

Background:
{background}

{resolution_criteria}

{fine_print}

Your research assistant says:
{summary_report}

Today is {today}.

Before answering you write:
(a) The time left until the outcome to the question is known.
(b) The status quo outcome if nothing changed.
(c) A description of an scenario that results in an unexpected outcome.

You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

The last thing you write is your final probabilities for the N options in this order {options} as follows without any decoration and replace Option_X with the actual option names:
Option_A: Probability_A
Option_B: Probability_B
...
Option_N: Probability_N
"""

RATIONALE_SUMMARY_PROMPT_TEMPLATE = """
You are analyzing multiple forecasting rationales for the same question to create a consolidated summary. 

Question: {question_title}
Question Type: {question_type}
Final Aggregated Prediction: {final_prediction}
Number of Rationales: {num_rationales}

All Rationales:
{combined_rationales}

Please create a comprehensive summary that includes:

1. **Key Consistent Themes**: What points do most rationales agree on?
2. **Main Supporting Evidence**: What are the strongest pieces of evidence mentioned across rationales?
3. **Contradictions & Disagreements**: Where do the rationales disagree and why?
4. **Confidence Factors**: What factors increase or decrease confidence in the prediction?
5. **Critical Assumptions**: What key assumptions are the rationales based on?
6. **Final Justification**: How do the combined insights justify the final aggregated prediction?

Keep the summary concise but comprehensive (300-500 words). Focus on insights that would be valuable to a professional forecaster reviewing this analysis.

Consolidated Summary:"""