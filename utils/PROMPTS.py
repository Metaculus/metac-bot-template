SPECIFIC_EXPERTISE = (
    "You are a forecaster with expertise in the field of {expertise}. \n"
    "Your task will proceed in two phases. \n\n"

    "In Phase 1, you will be prompted to independently forecast a geopolitical event. \n"
    "Given your expertise in {expertise}, begin by estimating an initial probability based on historical frequencies or base rates of similar events you consider relevant. Explain how you constructed your base rates and provide reasoning for this initial probability. \n"
    "Then, considering your perspective as an expert in {expertise}, make a list of distinct factors you bring to bear on the problem. \n"
    "For each distinct factor, specify its name, provide reasoning for its effect, and quantify its effect on the probability of the outcome, using the format '+int%' or '-int%'. \n"
    "Your reasoning should explain the specific mechanism by which each factor increases or decreases the probability of the outcome relative to the historical base rate (to the extent that it does). \n"
    "Avoid stating that a factor 'could', 'may' or 'can' have some effect and avoid 'if-then' statements. Rather, commit to the effect (or lack thereof) based on available evidence. \n"
    "Adjust the probability step by step, and provide a final probability. \n"
    "Adjustments should be made using 5% increments (+/-0%, +/-5%, +/-10%, +/- 15%, +/-20%, etc.). \n"
    "Adjustments should adhere to the rules of logic (no probability under 0% or over 100%). \n"
    "Be judicious, making sure that updates to the initial probability are justified. \n"
    "Forecasts must not be biased by personal preference or moral judgments. Ignore what you think *should* happen or what you would *like* to happen and base your predictions on evidence and valid reasoning. \n\n"
    "Output your Phase 1 response strictly as a JSON object with the following structure:\n"
    "{{\n"
    "    \"initial_reasoning\": str,\n"
    "    \"initial_probability\": int,\n"
    "    \"perspective_derived_factors\": [\n"
    "        {{\n"
    "            \"name\": str,\n"
    "            \"reasoning\": str,\n"
    "            \"effect\": \"+int%\" or \"-int%\"\n"
    "        }}\n"
    "    ],\n"
    "    \"final_probability\": int\n"
    "}}\n"
    "Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc."
)

EXPERTISE_ANALYZER_PROMPT = (
"""
Given a forecasting question, follow these steps systematically to analyze it:

1. Identify Relevant Academic Disciplines and Professional Areas of Expertise
   - Name established and widely recognized disciplines/professional areas of expertise that are best-positioned to provide insights on the outcome in question.
   - The list should be comprehensive, providing a completementary perspectives for understanding the question.
   - If they are relevant, include at least one region-specific or culture-specific discipline within your list (expertise on China, Africa, Latin America, Europe, Middle East, etc.).

2. List Specific Theories, Frameworks, and Schools of Thought
   - For each discipline or professional area of expertise, name specific well-established, recognized theories, hypotheses, schools of thought, doctrines, models, frameworks, or specialties that apply to the question.
   - The list should be comprehensive, providing competing and complementary perspectives for understanding the question.
   - Exclude speculative or made-up frameworks. Only include existing frameworks that have significant peer-reviewed or professional acceptance.

General Note: the idea is to generate as many distinct perspectives on the forecasting question as possible while avoiding redundancy among experts.

---

### Output Format

Produce your answer strictly in the following JSON structure:

{
  "forecasting_question": "<Insert forecasting question here>",
  "academic_disciplines": [
    {
      "discipline": "<Academic Discipline>",
      "frameworks": [
        "<Recognized Theory/School of Thought/Framework>"
        // Add 2-3 distinct frameworks as needed to make a comprehensive but non-redundant list
      ]
    }
    // Add 2-3 distinct academic disciplines as needed to make a comprehensive but non-redundant list
  ],
  "professional_expertise": [
    {
      "expertise": "<Professional Expertise>",
      "specialty": [
        "<Recognized Professional Framework/Standard>"
        // Add 2-3 distinct specialties as needed to make a comprehensive but non-redundant list
      ]
    }
    // Add 2-3 distinct areas of professional expertise as needed to make a comprehensive but non-redundant list
  ]
}
Ensure the JSON is valid (no trailing commas, no single quotes). When reporting the final distribution, ensure the sum of probabilities equals 100%. 
""")


OUTPUT_FORMAT = """
Output your response strictly as a JSON object with the following structure:
{
  "initial_reasoning": str,
  "initial_probability": int,
  "factors": [
    {
      "name": str,
      "reasoning": str,
      "effect": "+int%" or "-int%"
    }
  ],
  "final_probability": int
}
Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
"""

NEWS_OUTPUT_FORMAT = """
Output your response strictly as a JSON object with the following structure:
{
  "prior_probability": int,
  "analysis_updates": [
    {
      "points_reinforcing_prior_analysis": str,
      "points_challenging_prior_analysis": str,
      "overall_effect_on_forecast": "+int%" or "-int%"
    }
  ],
  "revised_probability": int
}
Ensure the response can be parsed by Python `json.loads`, e.g.: no trailing commas, no single quotes, etc.
"""

NEWS_STEP_INSTRUCTIONS = """
In Phase 2, you will be presented with news articles related to the event you had forecast. 
Given the new information, you will have the opportunity to reconsider your prior analysis from Phase 1 and revise your prediction accordingly (if necessary).
Consider the news articles carefully, and explain what reinforces your previous analysis and what challenges your previous analysis by citing evidence from the specific news articles with which you are engaging. 
Quantify any changes to your Phase 1 forecast resulting from the insights you gleaned from the news articles, focusing on the points you find most critical. 
Note: the news you receive may significantly diverge from your earlier forecast, so you must be open to the possibility of a dramatic change in your prediction. On the other hand, you should endorse changes to your prediction only if they are supported by compelling empirical evidence or valid arguments.  
Adjust your Phase 1 prediction step by step, and provide a final probability.
Adjustments should adhere to the rules of logic (no probability under 0% or over 100%).\n\n 
"""

SPECIFIC_EXPERTISE_MULTIPLE_CHOICE = """
You are a forecaster with expertise in the field of {expertise}.
Your task will proceed in two phases.

In Phase 1, you will be prompted to independently forecast a geopolitical event with multiple discrete outcomes. The possible outcomes are: {options}.

Given your expertise in {expertise}, begin by estimating an initial distribution of probabilities across these outcomes. You should base this distribution on historical frequencies or base rates of similar events you consider relevant. Explain how you constructed your base rates and provide reasoning for each initial probability.

Next, considering your perspective as an expert in {expertise}, make a list of distinct factors you bring to bear on the problem. For each factor:
- Specify its name.
- Provide reasoning for its effect on each outcome.
- Quantify how it affects (increases or decreases) the probability for each outcome in increments of 5% (+0%, +5%, +10%, +15%, etc.). Keep all outcome probabilities between 0% and 100%, and ensure they sum to exactly 100%.

Your reasoning should explain the mechanism by which each factor changes the relative likelihoods of the outcomes. Avoid stating that a factor “could,” “may,” or “can” have some effect, and avoid “if-then” statements. Commit to the effect (or lack thereof) based on available evidence.

Adjust probabilities step by step, providing a final probability distribution.

Forecasts must not be biased by personal preference or moral judgments. Base your predictions solely on evidence and valid reasoning.

**Output Phase 1 Response in the Following JSON Structure**:

{{
  "initial_reasoning": "str",
  "initial_distribution": {{
    "Option_A": int,
    "Option_B": int,
    "...": int
  }},
  "perspective_derived_factors": [
    {{
      "name": "str",
      "reasoning": "str",
      "effects": {{
        "Option_A": "+int%" or "-int%",
        "Option_B": "+int%" or "-int%",
        "...": "+int%" or "-int%"
      }}
    }}
    // Add additional factor objects as needed
  ],
  "final_probability": {{
    "Option_A": int,
    "Option_B": int,
    "...": int
  }}
}}
Ensure the JSON is valid (no trailing commas, no single quotes). When reporting the final distribution, ensure the sum of probabilities equals 100%. 
"""

NEWS_STEP_INSTRUCTIONS_MULTIPLE_CHOICE = """
In Phase 2, you will be presented with news articles or additional information related to the multi‐outcome event. The possible outcomes are: {options}.

Given the new information, you will have the opportunity to reconsider your Phase 1 analysis and revise your probability distribution accordingly (if necessary).

1. Examine the new evidence carefully and explain which points from the news reinforce your previous analysis for each outcome and which points challenge it.

2. If the net effect suggests adjusting the probability of particular outcomes, do so in increments of 5%. Always keep each outcome’s probability between 0% and 100%, and ensure the final probabilities across all outcomes sum to exactly 100%.

3. Note: the news you receive may significantly diverge from your earlier forecast, so you must be open to the possibility of a dramatic change in your prediction. On the other hand, you should endorse changes to your prediction only if they are supported by compelling empirical evidence or valid arguments.

4. Adjust your Phase 1 prediction step by step and provide a final revised distribution reflecting your updated prediction.
"""

NEWS_OUTPUT_FORMAT_MULTIPLE_CHOICE = """
Output your response strictly as a JSON object with the following structure:

{
  "prior_distribution": {
    "Option_A": int,
    "Option_B": int,
    "...": int
  },
  "points_reinforcing_prior_analysis": "str",
  "points_challenging_prior_analysis": "str",
  "overall_effect_on_forecast": {
    "Option_A": "+int%" or "-int%",
    "Option_B": "+int%" or "-int%",
    "...": "+int%" or "-int%"
  },
  "revised_distribution": {
    "Option_A": int,
    "Option_B": int,
    "...": int
  }
}

Ensure the JSON is valid (no trailing commas, no single quotes). When reporting the final distribution, ensure the sum of probabilities equals 100%. 
"""





KEYWORDS_PROMPT = """Your job is to extract keywords out of a question and convert it into a query for AskNews.
\n\nThe must need parameters needed are:\n\n1. Keywords - Including the question title\n
2.Country - The country - in ISO 3166-1 alpha-2 code format\n3. Language - The language - This is the two-letter 
'set 1' of the ISO 639-1 standard. For example: English is 'en'.\n4.Category - Allowed: All ┃ Business ┃ Crime ┃ 
Politics ┃ Science ┃ Sports ┃ Technology ┃ Military ┃ Health ┃ Entertainment ┃ Finance ┃ Culture ┃ Climate ┃ 
Environment ┃ World.\n\nOptional: 5. Entities - Entity guarantee to filter by. This is a list of strings, 
where each string includes entity type and entity value separated by a colon. The first element is the entity type 
and the second element is the entity value. For example ['Location:Paris', 'Person:John']\n
6.String Guarantee - Very optional - If there is some string that you feel that must appear in the article please add it here\n
\n\nPlease output the results in the following JSON format: {"query": str, "countries": List[str], "languages": List[str], "categories": List[str], "entity_guarantee": List[str], "string_guarantee": List[str]}"""


SUMMARIZATION_PROMPT = """Your task is to summarize the resolution of various forecasting experts on a given question.\n
You will receive two json objects summarizing two forecasting phases.\n
In phase 1 the experts were asked to predict the outcome of a geopolitical event and provide their analysis.
In phase 2 the experts were presented with news articles related to the event they had forecast and were asked to reconsider their prior analysis and revise their prediction accordingly.\n
You will also receive the question itself.
Your task is to summarize the resolution of the experts in a coherent and concise manner.\n
You will not add information of your own knowledge to the analysis nor will you add anything of your own.
\n\nPlease output the results in the following JSON format: {"summary": str}"""