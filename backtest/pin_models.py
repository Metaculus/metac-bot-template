#!/usr/bin/env python3
"""Activate the pinned `llms=` block in main.py before the bot runs.

Why this exists
---------------
forecasting-tools picks default models from whichever provider env vars are
present. With only OPENROUTER_API_KEY set it assigns the researcher role to
`openai/gpt-4o-search-preview`, a model OpenRouter no longer serves, so every
question dies with:

    litellm.NotFoundError: OpenrouterException -
    {"error":{"message":"No endpoints found for openai/gpt-4o-search-preview","code":404}}

main.py ships the fix as a commented-out block. This rewrites it into an active
one, so the template stays a clean fork (no edits to main.py in git) while the
run gets models that exist.

Changing models
---------------
Put one OpenRouter model id per line in `backtest/models.txt`:

    default:    openrouter/<model>     # the forecaster - this is the one that matters
    researcher: openrouter/<model>
    summarizer: openrouter/<model>
    parser:     openrouter/<model>     # must support structured outputs

Any role you leave out falls back to the defaults below. Delete the file to go
back to the defaults entirely. No workflow edits needed.

Cross-provider fallback
------------------------
Free tiers cap by the day, not just by the minute (OpenRouter: 50 free-model
requests/24h per account). A model that is fine at 9am can 429 every question
by 6pm, and the run does not recover until the provider's own daily reset -
retrying the same account is a no-op.

Each of OPENROUTER_API_KEY, GEMINI_API_KEY and GROQ_API_KEY that is present
activates one backend of FALLBACK_CHAIN. When at least one is, every role -
including parser - is wrapped in backtest.fallback_llm.FallbackLlm, which tries
the role's own model first and falls through to the active backends in chain
order (OpenRouter, Gemini, Groq), only on an actual failure (rate limit,
timeout, provider outage) - never as a quality preference. Each backend keeps
its own timeout/allowed_tries, and a model appears once per chain. If none of
the three keys is set, behavior is unchanged: a single GeneralLlm per role (a
bare model string for the parser), same as before fallback existed.

The parser is included even though it is a few hundred tokens per call and
was never at risk of a per-model or per-minute limit: OpenRouter's free tier
caps the whole account per day (50 requests/24h), not each model separately,
so once that account-wide cap is hit every role sharing the OPENROUTER_API_KEY
dies at once - parser included. Excluding it would leave the exact failure
mode this feature exists for (an OpenRouter-wide outage) able to kill the run
through the one role left unprotected.

AskNews research
----------------
With an AskNews credential present at patch time (ASKNEWS_API_KEY, or
ASKNEWS_CLIENT_ID + ASKNEWS_SECRET) the researcher role leads with
`asknews/news-summaries`, a GeneralLlm forecasting-tools routes to its
AskNewsSearcher, and the LLM researcher above moves to the first fallback
position: real news first, the model's own knowledge if the feed fails. No
credential, no change. See the note above ACTIVE_ASKNEWS.

Forecaster ensemble
-------------------
main.py asks for five predictions per question, and until now all five came
from one model -- five correlated samples whose shared blind spots survive the
average. The "default" role is therefore wrapped in a BalancedLlm holding one
fallback chain per PRIMARY model, so the five calls spread across the distinct
providers instead. Copied from FutureSearch, currently first in this same
tournament, who publish the result in "Run agents twice for fun and profit"
(2026-05-01): "Ensembling across two Opus 4.6 runs and other frontier models
cuts Brier score".

No new provider and no spend: these are the three models the fallback chain
already carries. Only the forecaster gets it -- summarizer and researcher are
called once per question, where choosing a model at random is a lottery rather
than an ensemble, and the parser is restricted to backends that emit
schema-constrained JSON.

Why the defaults are what they are
----------------------------------
Metaculus's own FutureEval writeup found model choice to be the single largest
differentiator between winning and losing bots, and that plain one-shot bots on
frontier models placed top-5. So the forecaster, the role whose output is
scored, leads on the strongest model the key can reach. The comment above
DEFAULTS has the current choice, its prices, and why: the OpenRouter key that
Metaculus funded on 2026-09-10 cannot reach nvidia/* models at all.

That matters because the first version of these defaults ran on free Nvidia
models. nemotron-3-ultra-550b, the obvious pick, was measured and rejected: on
OpenRouter's free tier it queued past 60s and timed out on 8 of 9 test
questions. A model that does not answer inside the window scores zero, however
good it is. nemotron-3.5-lightning, the free model built for latency, was the
primary for three roles until the sponsored key, which cannot reach it, took
over.

The parser has to emit structured output reliably, which the free Nvidia endpoints
do not advertise, so that role uses a cheap paid model - at roughly a thousandth of
a cent per question. With no fallback key present it stays a bare string, exactly
as the official template ships it: parsing is a few hundred tokens and never came
close to the default timeout, so there is nothing to gain by varying it. With one
it is wrapped like every other role, as described above.

Preflight
---------
`python backtest/pin_models.py --check` additionally sends one tiny completion to
every distinct model and exits 1 if any role is left with no working model at
all; a dead primary with a live fallback behind it is not a failure, because
absorbing that is what the fallback is for. That is what caught the dead
`gpt-4o-search-preview` id: a model that 404s costs a whole tournament run, and
the run is the scarce resource, not the token.

Exits 1 if the anchor block is missing, so an upstream template change fails
loudly instead of silently restoring the broken defaults.
"""

from __future__ import annotations

import ast
import pathlib
import sys

# Metaculus funded the OpenRouter key on 2026-09-10 ($100, incremental: strong
# MiniBench performance raises it toward ~$805). The sponsored key restricts
# which providers it may reach:
#
#   "your account's allowed-providers setting permits only:
#    openai, anthropic, google-ai-studio"
#
# which 404s every nvidia/* model. nemotron-3.5-lightning, the previous primary
# for three roles, therefore failed 47 times out of 47 in run 34455087832 -- a
# dead first hop on every question. It is gone, not demoted: unlike gpt-4o-mini
# (unfunded but permitted) there is no future in which this key can reach it.
#
# Roles are now split by what each one is FOR, at prices read from the
# catalogue (run 34457071209) against 34 LLM calls per question measured from
# production logs:
#
#   claude-opus-4.6    $5/$25 per Mtok   ~$1.02/question if used alone
#   claude-haiku-4.5   $1/$5             ~$0.20/question
#   gpt-4o-mini        cheap, verified 200 OK on this key
#
# Forecaster moved to Opus 5.5 on 2026-10-02. Read from OpenRouter's /models
# API that day: id anthropic/claude-opus-5.5, 12 endpoints, $4/$20 per Mtok,
# 1M context, 128K max output -- 20% cheaper than Opus 4.6 on both meters.
#
#   claude-opus-5.5    $4/$20 per Mtok   ~$0.82/question if used alone
#                      (an ESTIMATE: the $1.02 Opus 4.6 figure scaled by the
#                      price ratio, not a measurement of this model's output
#                      length)
#
# The litellm this repo locks (1.80.10) prices this id at exactly those
# numbers, but only in the cost map it downloads at import; its bundled copy
# predates the model. bot_helpers.register_model_prices fills that gap, so
# cost tracking does not silently read $0 on a run whose download failed.
#
# The forecaster leads on Opus because Metaculus' own FutureEval writeup found
# model choice to be the single largest differentiator, and it is the role
# whose output is scored. Researcher and summarizer get Haiku: they run once
# per question and summarise rather than judge. All five forecast calls go to
# the forecaster (see FORECAST_GRADE for why the cheap fallbacks no longer
# vote), which costs more per question and is where the score is decided.
DEFAULTS = {
    "default": "openrouter/anthropic/claude-opus-5.5",
    "researcher": "openrouter/anthropic/claude-haiku-4.5",
    "summarizer": "openrouter/anthropic/claude-haiku-4.5",
    # Claude since 2026-10-22. gpt-4o-mini led the parser until it lost
    # question 46224 (three of five numeric answers parsed as 1/1000 of what
    # the forecasters wrote, Test Bot run 39565436454) and one of five
    # forecasts on question 46191 (run 39596139823); every other model answered
    # all 379 calls of the 60 runs before it. Haiku 4.5 was already the
    # parser's verified fallback. gpt-4o-mini stays behind it, for outages
    # only (PARSER_EXTRA_CHAIN).
    "parser": "openrouter/anthropic/claude-haiku-4.5",
}

# Fallback chain behind each role's own model, tried in order only on
# failure (rate limit, timeout, outage). Each needs its own API key as an
# env var (OPENROUTER_API_KEY / GEMINI_API_KEY / GROQ_API_KEY) - present in
# the workflow only if the corresponding secret is set. Absent keys are
# skipped, never an error: fallback is an upgrade, not a requirement.
# Every entry here was verified end-to-end by research/smoke_test_providers.py
# (workflow run 32293364075): the provider's own /models catalogue was listed,
# then a real completion returned 200 OK through BOTH raw HTTP and litellm.
# Model names are NOT taken from litellm's price registry - that is a price
# table, not an entitlement check, and three of three guesses taken from it
# turned out to be unavailable to these accounts.
#
# Deliberately excluded, both verified as unusable at EUR 0:
#   Cerebras  402 "Payment required to access this resource" - no free tier
#   Metaculus 400 "You don't have an allowance for model <gpt-4o-mini>"
FALLBACK_CHAIN = [
    # Haiku leads the fallbacks: same provider and key as the primary, so it
    # is the closest substitute when Opus fails. It no longer votes in the
    # forecaster ensemble; see FORECAST_GRADE.
    ("openrouter/anthropic/claude-haiku-4.5", "OPENROUTER_API_KEY"),
    ("gemini/gemini-3.5-flash-lite", "GEMINI_API_KEY"),
    ("groq/openai/gpt-oss-120b", "GROQ_API_KEY"),
]

# Which of those can serve the PARSER. The parser is the one role whose output
# must be machine-readable: structure_output() requests a JSON schema and
# raises when two samples disagree, so a backend that answers in prose turns a
# survived outage into a discarded prediction.
#
# Measured, not assumed (smoke-test run 32293364075):
#   gemini/gemini-3.5-flash-lite  returned {"probability": 0.42}          OK
#   groq/openai/gpt-oss-120b      json_validate_failed, failed_generation="" 
#
# That second line was read for a month as "Groq cannot emit structured
# output". It says no such thing -- it is one model. A full sweep of Groq's
# catalogue (run 34409492553) found qwen/qwen3.8-27b returns exactly the
# requested {"probability": <number>}, so the provider is capable and the
# model that was tested is not. See PARSER_EXTRA_CHAIN below.
#
# groq/openai/gpt-oss-120b still stays out of the parser chain, on its own
# evidence, and stays in the reasoning chain where prose is what we want.
STRUCTURED_OUTPUT_CAPABLE = {
    "gemini/gemini-3.5-flash-lite",
    "groq/qwen/qwen3.8-27b",
    # Both verified in the catalogue sweep, run 34457071209, by the same probe
    # that qualified the two above: a strict json_schema request answered with
    # parsed_keys ["probability"].
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/openai/gpt-4o-mini",
}

# Backends added to the PARSER chain only.
#
# Why the parser needed its own list: the parser's depth is decided by
# STRUCTURED_OUTPUT_CAPABLE intersected with FALLBACK_CHAIN, and with
# OpenRouter out of credit that intersection was ONE reachable backend --
# Gemini, on a single 15 RPM credential. When Gemini rate-limited, the parser
# had nowhere to go, structure_output() got no JSON, and the prediction was
# discarded. That cost five questions outright on 2026-09-08 (runs
# 34176403744 and 34193820860, 9 and 60 llm_exhausted respectively). In the
# 06:14 run the REASONING was healthy -- nemotron answered 25 times -- so
# nothing was wrong with the forecast except that it could not be parsed.
#
# Adding this to FALLBACK_CHAIN instead would have fixed the parser AND
# silently reshaped the forecaster ensemble from three primaries to four. The
# ensemble was validated end-to-end (run 34115592907); a parser fix has no
# business changing it.
#
# MEASURED, not assumed, to the same standard as the Gemini entry above.
# research/smoke_test_providers.py --structured-sweep asked Groq what it serves
# and probed all 9 chat models with a strict json_schema response_format
# (run 34409492553):
#
#   qwen/qwen3.8-27b     OK    parsed_keys ["probability"], raw {"probability": 0.42}
#   openai/gpt-oss-120b  FAIL  BAD_REQUEST      <- the one model tested in 2026-08
#   openai/gpt-oss-20b   FAIL  BAD_REQUEST
#   qwen/qwen3.6-27b     FAIL  BAD_REQUEST
#   + 5 more, all BAD_REQUEST
#
# So "Groq cannot emit structured output" was true of the model tested and
# false of the provider. One measurement of one model had been carried forward
# as if it settled the question.
PARSER_EXTRA_CHAIN = [
    # The parser's primary until 2026-10-22 (see DEFAULTS); behind Haiku now,
    # so an OpenRouter-side Anthropic outage still has a JSON backend on the
    # same key.
    ("openrouter/openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    # Qualified during the OpenRouter outage and kept afterwards: it is the one
    # Groq model that emits schema-constrained JSON, and a third reachable
    # parser backend is exactly what was missing when five questions were lost
    # on 2026-09-08. It sits AFTER the funded backends because it rate-limited
    # 43 times in run 34412589779 when it had to carry the role alone.
    ("groq/qwen/qwen3.8-27b", "GROQ_API_KEY"),
]


def _parser_primary(configured: str) -> str:
    """Which backend the parser TRIES FIRST.

    Now simply the configured primary. This function was introduced on
    2026-09-10 to route AROUND gpt-4o-mini, which returned 402 on every call
    while the OpenRouter account had no credit -- 308 guaranteed failures in
    two days. Metaculus funded the key hours later and it answers 200 OK again
    (run 34457071209), so the reason to route around it is gone.

    Kept rather than inlined because the situation recurs: the $100 is finite,
    and if it runs out this is the one place that decides what leads the chain.
    The Groq model that covered the outage stays in PARSER_EXTRA_CHAIN, so the
    chain is now deeper than before the outage rather than merely restored.
    """
    return configured


# forecasting-tools only applies a custom timeout to roles given as GeneralLlm;
# a bare model string silently gets its 60s default. The free tier queues past
# that, which showed up as `litellm.Timeout ... timeout passed=60.0` on 8 of 9
# test questions. Hence: every role is a GeneralLlm, and the budget is generous.
TIMEOUT_SECONDS = 180

# Which fallback keys are actually available *right now*. Read once, at patch
# time, via os.environ - not deferred to the bot's own runtime - so the
# generated main.py is an honest record of what ran, and --check can verify
# exactly the chain that will be used.
import os

ACTIVE_FALLBACKS = [
    (model, env_var) for model, env_var in FALLBACK_CHAIN if os.getenv(env_var)
]

# Parser-only extras whose credential is present RIGHT NOW. Resolved here, at
# import time, for the same reason ACTIVE_FALLBACKS is: every other env read in
# this module happens once at patch time, so the generated main.py is an honest
# record of what ran. Reading os.environ inside _fallbacks_for() instead would
# make that one function's result depend on when it is called, and would make
# its tests depend on the caller's environment.
ACTIVE_PARSER_EXTRA = [
    (model, env_var) for model, env_var in PARSER_EXTRA_CHAIN if os.getenv(env_var)
]

# ---------------------------------------------------------------- AskNews
# Live news for the researcher, gated on a credential exactly like the
# fallback keys above.
#
# Until 2026-10-02 the researcher was a model working from its own training
# data, so anything that happened after its cutoff was invisible to the
# forecast. forecasting_tools already routes the model string below to
# AskNewsSearcher (GeneralLlm._call_asknews): a GeneralLlm built on it is a
# real backend that returns formatted news articles, so it can lead the
# researcher's FallbackLlm chain with no new class and no change to main.py's
# run_research, which already handles a GeneralLlm researcher.
#
# What the chain looks like with a credential present:
#
#   researcher: [asknews/news-summaries, <configured researcher>, *fallbacks]
#
# so an AskNews failure of any kind -- a credential that reaches the pin step
# but not the run step, a rate limit, an outage -- costs one attempt and the
# research is produced by the LLM exactly as it was before. This holds even
# with NO fallback key present: the chain is then [asknews, <configured
# researcher>], and the FallbackLlm import is emitted for it. With no AskNews credential nothing here runs and the generated
# block is byte-for-byte what it was.
#
# Only the researcher: AskNews returns news, not reasoning, so it has no
# business in the forecaster ensemble, the summarizer or the parser.
#
# AskNewsSearcher accepts ASKNEWS_API_KEY, or ASKNEWS_CLIENT_ID together with
# ASKNEWS_SECRET. Read once, at import time, for the reason ACTIVE_FALLBACKS
# is (see above). test_pin_models_chains manages all three names.
ASKNEWS_MODEL = "asknews/news-summaries"
ASKNEWS_ENV_VARS = ("ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET")


def _asknews_credential_present(environ) -> bool:
    """Would AskNewsSearcher() authenticate from this environment?

    Mirrors its constructor: an API key, or the OAuth pair, and not both --
    it refuses the combination. Half a pair, or both styles at once, keeps
    the gate shut rather than leading every question with a backend that is
    certain to raise. test_asknews_research pins this against the real
    constructor for every combination of the three variables.
    """
    has_api_key = bool(environ.get("ASKNEWS_API_KEY"))
    has_oauth = bool(environ.get("ASKNEWS_CLIENT_ID") and environ.get("ASKNEWS_SECRET"))
    return has_api_key != has_oauth


ACTIVE_ASKNEWS = _asknews_credential_present(os.environ)

# Model strings that backtest.rate_limiter.DEFAULT_LIMITS registers a limiter
# for. Duplicated here because this file runs as a bare script -- the workflow
# calls `python backtest/pin_models.py`, so the repo root is not on sys.path
# and `from backtest.rate_limiter import ...` would raise ModuleNotFoundError
# before the bot ever starts. tests/test_pin_models_chains.py asserts the two
# never drift apart.
#
# It matters because `DEFAULT_LIMITS.get(model, ProviderLimits())` returns an
# UNTHROTTLED limiter for an unknown key: an ensemble bucket key missing from
# the registry would look controlled and rate-limit nothing, quietly removing
# Groq's measured 8000 TPM cap.
RATE_LIMITED_MODELS = frozenset(
    {
        "asknews/news-summaries",
        "gemini/gemini-3.5-flash-lite",
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.8-27b",
        "openrouter/anthropic/claude-haiku-4.5",
        "openrouter/anthropic/claude-opus-4.6",
        "openrouter/anthropic/claude-opus-5.5",
        "openrouter/nvidia/nemotron-3.5-lightning:free",
        "openrouter/openai/gpt-4o-mini",
        "openrouter/openai/gpt-5.6-sol",
    }
)

OVERRIDE_FILE = pathlib.Path(__file__).with_name("models.txt")

ANCHOR = '''        # llms={
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
'''

# All roles now get fallback-wrapped when active (see docstring: OpenRouter's
# daily cap is account-wide, so parser shares the exact failure mode this
# feature exists to protect against).
REASONING_ROLES = ("default", "summarizer", "researcher", "parser")


# A provider inside a fallback chain gets exactly ONE attempt. GeneralLlm.invoke
# is wrapped by RetryableModel with wait_random_exponential(min=5, max=60), so
# allowed_tries=3 on a chain member means up to ~2 minutes of backoff before the
# next provider is even tried - and a daily-quota 429 cannot succeed on retry
# anyway. Retrying is the fallback's job now, and it retries by switching
# provider, not by asking the same exhausted account again.
#
# Standalone (no fallback keys present) keeps allowed_tries=3, so the
# no-fallback path stays byte-for-byte identical to before this feature.
TRIES_IN_CHAIN = 1
TRIES_STANDALONE = 3


# MORE REASONING FOR THE FORECAST CALL -- 2026-10-15.
#
# Opus 5.5 thinks adaptively and defaults to effort "medium". Metaculus'
# Spring 2026 survey found the high-reasoning variant beating the standard
# one in 8 of 8 model pairs, and the forecast is the call the score depends
# on. So the forecaster role asks for "high" -- through OpenRouter's own
# `reasoning: {effort}` request field -- on the models listed here only.
#
# Sent as extra_body, NOT litellm's reasoning_effort: litellm 1.80.10 (the
# locked version) raises UnsupportedParamsError for reasoning_effort on any
# openrouter/ model, which would have failed every Opus call over to Haiku.
# extra_body reaches the request verbatim (checked against a local stand-in
# for the OpenRouter endpoint). The cheap fallbacks and the other roles are left
# as they were. Thinking takes longer, hence the larger timeout.
FORECASTER_REASONING = {
    "openrouter/anthropic/claude-opus-5.5": "high",
    "openrouter/openai/gpt-5.6-sol": "high",
}
FORECASTER_TIMEOUT_SECONDS = 420

# IS OPUS ACTUALLY REASONING? -- 2026-10-23.
#
# At "high" the median Opus forecast call takes 14.7 s and costs ~$0.058,
# which leaves room for only ~300-500 hidden reasoning tokens. spend.py now
# records the reasoning tokens each call bills; FORECASTER_REASONING_OPUS lets
# a Test Bot run ask for another effort (test_bot.yaml's opus_reasoning input)
# without a commit, so "high" and "xhigh" can be compared on the same
# questions. Unset -- as in every production workflow -- means no change.
#
# Effort NAMES only. A token budget (reasoning.max_tokens) returns 400 on
# Opus 5.5, which would fail every Opus call over to Sol or Haiku without an
# error anywhere, so anything outside this list is ignored with a warning.
# Read once, at import time, for the reason ACTIVE_FALLBACKS is.
OPUS_FORECASTER = "openrouter/anthropic/claude-opus-5.5"
OPUS_EFFORTS = ("low", "medium", "high", "xhigh")
_OPUS_EFFORT_OVERRIDE = os.environ.get("FORECASTER_REASONING_OPUS", "").strip().lower()
if _OPUS_EFFORT_OVERRIDE in OPUS_EFFORTS:
    FORECASTER_REASONING[OPUS_FORECASTER] = _OPUS_EFFORT_OVERRIDE
elif _OPUS_EFFORT_OVERRIDE:
    print(
        "::warning::FORECASTER_REASONING_OPUS={0!r} is not one of {1}; Opus keeps "
        "effort {2!r}".format(_OPUS_EFFORT_OVERRIDE[:20], ", ".join(OPUS_EFFORTS),
                              FORECASTER_REASONING[OPUS_FORECASTER]),
        file=sys.stderr,
    )


def _backend_expr(
    model: str, allowed_tries: int = TRIES_STANDALONE, role: str | None = None
) -> str:
    """One GeneralLlm(...) call, single line, for use inside a FallbackLlm list
    or standalone."""
    effort = FORECASTER_REASONING.get(model) if role == "default" else None
    if effort:
        return (
            f'GeneralLlm(model="{model}", temperature=0.3, '
            f"timeout={FORECASTER_TIMEOUT_SECONDS}, allowed_tries={allowed_tries}, "
            f'extra_body={{"reasoning": {{"effort": "{effort}"}}}})'
        )
    return (
        f'GeneralLlm(model="{model}", temperature=0.3, '
        f"timeout={TIMEOUT_SECONDS}, allowed_tries={allowed_tries})"
    )


def _fallbacks_for(role: str) -> list[str]:
    """Active fallback models this role may actually use.

    Every role gets the full chain except the parser, which only gets backends
    verified to emit schema-constrained JSON."""
    models = [model for model, _env in ACTIVE_FALLBACKS]
    if role != "parser":
        return models
    capable = [m for m in models if m in STRUCTURED_OUTPUT_CAPABLE]
    # Parser-only extras, appended after the shared chain so the existing
    # order is untouched and a credential that is absent contributes nothing.
    for model, _env_var in ACTIVE_PARSER_EXTRA:
        if model not in capable:
            capable.append(model)
    # The primary leads the chain in _chain_expr; listing it again here would
    # emit the same backend twice and burn two attempts on one dead provider.
    primary = _parser_primary(DEFAULTS["parser"])
    return [m for m in capable if m != primary]


# ---------------------------------------------------------------- ensemble
# MODEL DIVERSITY FOR THE FORECASTER ROLE.
#
# main.py asks for predictions_per_research_report=5, so the "default" role is
# invoked five times per question and the five answers are aggregated. Until
# now all five came from the SAME model at temperature 0.3, which averages five
# highly correlated samples: the shared prior, and the shared blind spots,
# survive the mean untouched.
#
# FutureSearch (currently first in the same tournament) publishes the finding
# this copies, in "Run agents twice for fun and profit" (2026-05-01):
#
#     "Running the same forecasting agent more than once and averaging beats
#      any single run."  ... "Ensembling across two Opus 4.6 runs AND OTHER
#      FRONTIER MODELS cuts Brier score"
#
# The second half is the part we were missing. Their Polymarket writeup states
# the same shape concretely: "six research agents and three forecasting models
# per market".
#
# So the default role becomes a BalancedLlm over one chain PER PRIMARY MODEL.
# Each of the five calls enters the least-loaded chain, which spreads them over
# the distinct primaries instead of hammering one.
#
# What this is NOT:
#   * not a new provider, not a new credential, not a euro of spend - these are
#     the same three models the fallback chain already carries, all three
#     verified end-to-end by research/smoke_test_providers.py (run 32293364075)
#   * not a change to how many predictions exist, nor to aggregation, nor to
#     the success threshold - BalancedLlm is one call in, one string out
#   * not applied to summarizer/researcher (called once per question, so
#     "diversity" there is a lottery over which model answers, not an ensemble)
#     nor to parser (restricted to STRUCTURED_OUTPUT_CAPABLE backends, the
#     ones verified to emit schema-constrained JSON)
#
# Bucket keys are the primary model names themselves, which is deliberate:
# those are the exact strings backtest.rate_limiter.DEFAULT_LIMITS registers,
# so the balancer measures load on the SAME limiters the backends acquire from,
# rather than keeping a parallel set of books. test_pin_models_chains asserts
# every emitted bucket key is registered - an unregistered key silently
# resolves to an UNTHROTTLED limiter, which would quietly disable Groq's 8000
# TPM cap.
# WHICH FALLBACKS MAY ALSO FORECAST -- corrected 2026-10-10.
#
# The ensemble above used to promote EVERY active fallback to a primary. With
# all three keys present that was Opus + Haiku 4.5 + Gemini Flash Lite +
# gpt-oss-120b, and the five forecast calls spread over four chains: one or
# two came from Opus, the rest from models chosen for being cheap and fast.
# The binary median of five is then usually a weak model's number. Measured on
# 141 resolved questions: mean peer -0.81, and binary direction 75.9% against
# 74.1% for answering NO to everything.
#
# What FutureSearch ensembles is "other FRONTIER models"; what Metaculus'
# Spring 2026 survey found predictive was the frontier model doing the final
# forecast, and high-reasoning variants beating standard ones 8 of 8. Cheap
# fallbacks earn their place as fallbacks, not as voters.
#
# So a fallback joins the ensemble only if it is listed here. Empty today:
# every forecast comes from the configured default (Opus), and the fallbacks
# still catch it on failure. To add a second frontier voter (e.g. a GPT-5.x
# high-reasoning model), verify it through research/smoke_test_providers.py
# first -- the sponsored key only reaches openai/anthropic/google -- then add
# it to FALLBACK_CHAIN and here.
FORECAST_GRADE: frozenset[str] = frozenset()


# SECOND-VENDOR FORECAST VOTERS -- 2026-10-18.
#
# Voters that are NOT fallbacks: they join the forecaster ensemble only, so the
# once-per-question roles (researcher, summarizer, parser) keep exactly the
# chains they had. Every top-10 respondent in Metaculus' Spring 2026 survey had
# a GPT-5.x model in the ensemble, and the open top bots (nostreambot, Ora)
# mix vendors; five Opus samples over one shared research text make the same
# mistakes together.
#
# GPT-5.6 Sol: smoke-tested on the sponsored key (run 39073967369: http,
# litellm and structured output all OK), and priced like Opus 5.5 ($4/$20 per
# M tokens, openrouter.ai/api/v1/models) -- splitting the five calls between
# them costs what five Opus calls did. Gemini 3.1 Pro was tried in the same
# run and failed with an upstream Google quota error, so it is not here.
FORECAST_VOTERS = [
    ("openrouter/openai/gpt-5.6-sol", "OPENROUTER_API_KEY"),
]
ACTIVE_VOTERS = [(m, e) for m, e in FORECAST_VOTERS if os.getenv(e)]


def _ensemble_primaries(model: str) -> list[str]:
    """The distinct primary models the forecaster ensembles over.

    ``model`` (the configured default, overridable via models.txt) leads, then
    every active fallback that is FORECAST_GRADE and not already it, in the
    production order. With none, this is a single entry and the forecaster is
    one fallback chain led by ``model``.
    """
    primaries = [model]
    for fb_model, _env in ACTIVE_FALLBACKS:
        if fb_model in FORECAST_GRADE and fb_model not in primaries:
            primaries.append(fb_model)
    for voter, _env in ACTIVE_VOTERS:
        if voter not in primaries:
            primaries.append(voter)
    return primaries


def _ensemble_expr(model: str) -> str:
    """One BalancedLlm holding one FallbackLlm per primary model.

    Chain i puts primary i first and keeps the remaining models in their
    canonical production order, so every chain still degrades through the same
    ladder it always did - only its starting rung differs.
    """
    primaries = _ensemble_primaries(model)
    chains = []
    # The fallbacks that do not vote still back every chain, after the other
    # voters: a voter that fails must degrade exactly as the single-chain
    # forecaster would, not lose the cheap rungs FORECAST_GRADE left out.
    non_voting = [fb for fb, _env in ACTIVE_FALLBACKS if fb not in primaries]
    for primary in primaries:
        others = [m for m in primaries if m != primary] + non_voting
        chains.append(_chain_expr(primary, others, indent=16, role="default"))
    joined = ",\n                ".join(chains)
    keys = ", ".join(f'"{p}"' for p in primaries)
    return f"BalancedLlm([\n                {joined},\n            ], [{keys}])"


def _chain_models(model: str, fallbacks: list[str]) -> list[str]:
    """The models one FallbackLlm chain tries, in order, each exactly once.

    The primary leads and the fallbacks follow in their configured order; a
    model that would appear a second time is dropped, so the first occurrence
    wins and the order of everything else is untouched.

    This exists because FALLBACK_CHAIN begins with claude-haiku-4.5 while the
    researcher and summarizer roles use that same model as their PRIMARY. The
    chain emitted for them was [haiku, haiku, gemini, groq]: a second call to
    the backend that had just failed, with the same model, the same credential
    and the same rate-limit window. TRIES_IN_CHAIN is 1 precisely so that a
    provider gets one attempt and the next attempt goes to a different provider
    (see the comment above it), and the repeat also inflated
    COUNTERS.llm_fallback_total, which counts any attempt past position 0 as a
    fallback. The parser already excluded its own primary in _fallbacks_for;
    doing it here covers every role and every models.txt override.
    """
    return list(dict.fromkeys([model, *fallbacks]))


def _chain_expr(
    model: str, fallbacks: list[str], indent: int = 12, role: str | None = None
) -> str:
    """One FallbackLlm, in the production order OpenRouter -> Gemini -> Groq.

    The ORDER is never altered -- balancing chooses between whole chains, it
    does not reorder the links inside one. No backend is listed twice; see
    _chain_models.
    """
    ordered = _chain_models(model, fallbacks)
    backends = [_backend_expr(m, TRIES_IN_CHAIN, role) for m in ordered]
    # A chain nested inside BalancedLlm sits one level deeper, hence `indent`.
    inner = " " * (indent + 4)
    outer = " " * indent
    joined = (",\n" + inner).join(backends)
    return f"FallbackLlm([\n{inner}{joined},\n{outer}])"


def _role_expr(role: str, model: str) -> str:
    """The right-hand side for one llms={} entry. Every role, the parser
    included (see REASONING_ROLES), gets wrapped in FallbackLlm when
    _fallbacks_for(role) is not empty, which takes at least one fallback API
    key; otherwise a single GeneralLlm, unchanged from before fallback
    existed.

    The "default" role additionally ensembles across primary models -- see the
    ensemble note above _ensemble_primaries.
    """
    if role not in REASONING_ROLES:
        return _backend_expr(model, role=role)
    fallbacks = _fallbacks_for(role)
    if role == "researcher" and ACTIVE_ASKNEWS:
        # AskNews leads and the configured LLM researcher becomes the first
        # fallback, so research never dies with the news feed. Wrapped even
        # with no fallback key, which is why this precedes the bare-backend
        # return below. See the AskNews note above ACTIVE_ASKNEWS.
        return _chain_expr(ASKNEWS_MODEL, [model, *fallbacks])
    if role == "default" and len(_ensemble_primaries(model)) > 1:
        return _ensemble_expr(model)
    if not fallbacks:
        return _backend_expr(model, role=role)
    return _chain_expr(model, fallbacks, role=role)


def _runtime_chain(role: str, model: str) -> list[str]:
    """The models one role actually tries at runtime, in order, each once.

    The single source for everything that reasons about a role's chain
    without generating it: the evaluator in _eval_llms_dict and the --check
    preflight. Derived from the same helpers _role_expr uses, so the three
    cannot disagree about which backend leads. For the forecaster ensemble the
    answer is the first chain's order; the other chains hold the same models
    with a different lead (asserted separately in _eval_llms_dict).
    """
    if role not in REASONING_ROLES:
        return [model]
    fallbacks = _fallbacks_for(role)
    if role == "researcher" and ACTIVE_ASKNEWS:
        return _chain_models(ASKNEWS_MODEL, [model, *fallbacks])
    if role == "default":
        # The voters run on every question too, so --check must prove they
        # answer: first chain's order, primaries then non-voting fallbacks.
        return _chain_models(model, [*_ensemble_primaries(model)[1:], *fallbacks])
    return _chain_models(model, fallbacks)


TEMPLATE_HEADER = '''        llms={
'''
TEMPLATE_FOOTER = '''        },
'''


def build_block(models: dict[str, str]) -> str:
    models = dict(models)
    models["parser"] = _parser_primary(models["parser"])
    lines = [TEMPLATE_HEADER.rstrip("\n")]
    for role in ("default", "summarizer", "researcher", "parser"):
        if role == "parser" and not ACTIVE_FALLBACKS:
            # Unwrapped: stay a bare model string, exactly as before fallback
            # existed, so the no-fallback path is byte-for-byte unchanged.
            lines.append(f'            "parser": "{models["parser"]}",')
        else:
            lines.append(f'            "{role}": {_role_expr(role, models[role])},')
    lines.append(TEMPLATE_FOOTER.rstrip("\n"))
    return "\n".join(lines) + "\n"


def read_overrides(path: pathlib.Path) -> dict[str, str]:
    """Parse `role: model` lines. Unknown roles are an error, not a silent no-op."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path.name}:{lineno}: expected 'role: model', got {raw!r}")
        role, model = (part.strip() for part in line.split(":", 1))
        if role not in DEFAULTS:
            raise ValueError(
                f"{path.name}:{lineno}: unknown role {role!r}; "
                f"expected one of {sorted(DEFAULTS)}"
            )
        if not model:
            raise ValueError(f"{path.name}:{lineno}: empty model for role {role!r}")
        out[role] = model
    return out


IMPORT_ANCHOR = "silence_noisy_dependencies()\n"
IMPORT_LINE = "from backtest.fallback_llm import FallbackLlm\n"
BALANCED_IMPORT_LINE = "from backtest.balanced_llm import BalancedLlm\n"


def _wants_fallback_wrapper() -> bool:
    """Does the block about to be generated name FallbackLlm at all?

    True with any fallback key, and true with an AskNews credential alone,
    because the researcher is then a two-link chain even when nothing else is
    wrapped. The import and the staleness check in patch() both ask this
    rather than ACTIVE_FALLBACKS directly, so neither can emit FallbackLlm(
    without importing it.
    """
    return bool(ACTIVE_FALLBACKS) or ACTIVE_ASKNEWS


def _ensure_fallback_import(src: str, wants_balanced: bool = False) -> str:
    """Add the FallbackLlm import once, only when the block names FallbackLlm
    (a fallback key, or an AskNews credential -- see _wants_fallback_wrapper).
    Idempotent and independent of whether the llms= block itself is patched
    yet, so re-running after an env change (a key added or removed) converges
    to the right import state on the next invocation either way.

    ``wants_balanced`` says whether the block ABOUT to be generated actually
    names BalancedLlm, which today only the forecaster ensemble does. It is
    passed in rather than recomputed here because only the caller knows the
    resolved model names, and the ensemble's shape depends on them (see
    _ensemble_primaries)."""
    has_import = IMPORT_LINE in src
    wants_import = _wants_fallback_wrapper()
    if wants_import and not has_import:
        if IMPORT_ANCHOR not in src:
            raise LookupError(
                "import anchor not found in main.py; upstream template changed"
            )
        # Assign rather than return: the balanced import below must be
        # considered in the SAME pass, or the first patch would emit
        # BalancedLlm(...) without importing it and only a second run would
        # fix it -- which also breaks the idempotency the selftest asserts.
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_LINE, 1)
    elif not wants_import and has_import:
        src = src.replace(IMPORT_LINE, "", 1)

    # The balanced import follows the same converge-either-way rule, so adding
    # or removing a provider key and re-running lands in the right state.
    has_balanced = BALANCED_IMPORT_LINE in src
    needs_balanced = wants_balanced and bool(ACTIVE_FALLBACKS)
    if needs_balanced and not has_balanced:
        if IMPORT_LINE not in src:
            raise LookupError(
                "FallbackLlm import missing; cannot anchor the balanced import"
            )
        src = src.replace(IMPORT_LINE, IMPORT_LINE + BALANCED_IMPORT_LINE, 1)
    elif not needs_balanced and has_balanced:
        src = src.replace(BALANCED_IMPORT_LINE, "", 1)
    return src


def _block_uses_balanced(models: dict[str, str]) -> bool:
    """Does the block we are about to emit name BalancedLlm anywhere?

    Asked of the generator itself rather than re-deriving the condition, so the
    import can never disagree with the code it has to support -- which is the
    one way this file can produce a main.py that raises NameError at startup.
    """
    return any(
        "BalancedLlm(" in _role_expr(role, models[role])
        for role in ("default", "summarizer", "researcher", "parser")
    )


# main.py's own module docstring contains a *documentation example* of an
# llms={...} block, at the same indentation, to show users the syntax. A bare
# text search for "llms={" matches that example first and silently patches
# nothing real. Anchoring the search to start after the actual bot
# instantiation call is what makes this unambiguous.
BOT_INIT_ANCHOR = "template_bot = SummerTemplateBot2026(\n"


def _llms_block_span(src: str) -> tuple[int, int] | None:
    """Locate the active (uncommented) llms={...} block inside the real bot
    instantiation - not the docstring example above it - by brace counting.
    Generated code has a known shape, not arbitrary Python, so this is safe."""
    init_at = src.find(BOT_INIT_ANCHOR)
    if init_at == -1:
        return None  # upstream renamed/restructured the bot init; caller falls through to ANCHOR/LookupError
    start = src.find("        llms={", init_at)
    if start == -1 or src[: start + 8].rstrip().endswith("#"):
        return None
    depth = 0
    i = src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                # include the trailing comma + newline the template emits
                end = j + 1
                if src[end : end + 2] == ",\n":
                    end += 2
                return start, end
    return None  # pragma: no cover - malformed input, caller re-raises via LookupError


def patch(src: str, models: dict[str, str]) -> str:
    src = _ensure_fallback_import(src, wants_balanced=_block_uses_balanced(models))

    span = _llms_block_span(src)
    if span is not None:
        current_block = src[span[0] : span[1]]
        wants_fallback = _wants_fallback_wrapper()
        has_fallback = "FallbackLlm(" in current_block
        # Same test for AskNews: a credential added or removed since the block
        # was written must rebuild it, or the researcher keeps a news feed it
        # cannot authenticate, or never gains one it could.
        has_asknews = ASKNEWS_MODEL in current_block
        if wants_fallback == has_fallback and ACTIVE_ASKNEWS == has_asknews:
            return src  # already active and already matches the current credential state
        # Stale: keys were added/removed since this block was written.
        # Rebuild it in place rather than leaving an import/usage mismatch.
        return src[: span[0]] + build_block(models) + src[span[1] :]

    if ANCHOR not in src:
        raise LookupError("anchor block not found in main.py; upstream template changed")
    return src.replace(ANCHOR, build_block(models))


def _eval_llms_dict(generated: str, active_fallbacks: list, models: dict[str, str] | None = None) -> dict:
    """Actually execute the generated `llms={...}` block against stand-ins for
    GeneralLlm/FallbackLlm and return the resulting dict. ast.parse only proves
    the code is syntactically valid Python - it does NOT prove `llms={{...}}`
    (doubled braces, a leftover from an earlier .format()-based template)
    parses as a dict rather than a set-containing-a-dict, which is valid
    syntax but crashes at runtime with "unhashable type: dict". This caught
    exactly that bug; keep it, do not downgrade it back to just ast.parse."""
    models = models if models is not None else DEFAULTS
    span = _llms_block_span(generated)
    assert span is not None, "generated code has no llms={...} block to evaluate"
    block_src = generated[span[0] : span[1]].strip().rstrip(",")

    class _StubGeneralLlm:
        def __init__(self, model, **kw):
            self.model = model

    class _StubFallbackLlm:
        def __init__(self, backends):
            self.backends = backends

    class _StubBalancedLlm:
        def __init__(self, chains, bucket_keys):
            self.chains = chains
            self.bucket_keys = bucket_keys

    namespace = {
        "GeneralLlm": _StubGeneralLlm,
        "FallbackLlm": _StubFallbackLlm,
        "BalancedLlm": _StubBalancedLlm,
    }
    exec(f"result = {block_src[len('llms='):]}", namespace)  # noqa: S102 - trusted, self-generated code
    result = namespace["result"]
    assert isinstance(result, dict), f"llms= evaluated to {type(result)}, not dict"
    assert set(result) == {"default", "summarizer", "researcher", "parser"}
    for role in ("default", "summarizer", "researcher", "parser"):
        # Chain length is per-role now: the parser only accepts backends
        # verified to emit schema-constrained JSON, so it can legitimately be
        # shorter than the reasoning chain (or absent entirely, if the only
        # active fallback cannot parse).
        expected_fallbacks = _fallbacks_for(role) if active_fallbacks else []
        # The chain as the generator emits it: primary first, each backend once.
        primary = _parser_primary(models[role]) if role == "parser" else models[role]
        expected_chain = _runtime_chain(role, primary)
        # Wrapped exactly when the generator wraps: a fallback behind the
        # primary, or AskNews in front of the researcher (which wraps even
        # with no fallback key at all).
        # The forecaster also wraps whenever it ensembles: a FORECAST_VOTERS
        # entry is active on its own key even when no fallback is.
        wrapped = (
            bool(expected_fallbacks)
            or (role == "researcher" and ACTIVE_ASKNEWS)
            or (role == "default" and len(_ensemble_primaries(primary)) > 1)
        )
        if role == "researcher" and ACTIVE_ASKNEWS:
            assert expected_chain[0] == ASKNEWS_MODEL
            # Unless models.txt itself names AskNews as the researcher, in
            # which case there is no separate LLM to stand behind it.
            assert primary == ASKNEWS_MODEL or expected_chain[1:2] == [primary], (
                f"{role}: the configured LLM must be the first fallback behind "
                f"AskNews, got {expected_chain}"
            )
        if wrapped:
            entry = result[role]
            if role == "default" and len(_ensemble_primaries(models[role])) > 1:
                # One chain per PRIMARY model. Every chain must still carry the
                # full set of primaries -- only the starting rung differs -- so
                # a chain that begins on a saturated provider can still degrade
                # to the other two rather than losing the prediction.
                primaries = _ensemble_primaries(models[role])
                assert isinstance(entry, _StubBalancedLlm), (
                    f"{role} should ensemble across {len(primaries)} models"
                )
                assert entry.bucket_keys == primaries, (
                    f"{role}: bucket keys {entry.bucket_keys} != primaries {primaries}"
                )
                assert len(entry.chains) == len(primaries), (
                    f"{role}: {len(entry.chains)} chains for {len(primaries)} primaries"
                )
                for chain, primary in zip(entry.chains, primaries):
                    assert isinstance(chain, _StubFallbackLlm)
                    served = [b.model for b in chain.backends]
                    assert served[0] == primary, (
                        f"{role}: chain leads with {served[0]}, expected {primary}"
                    )
                    assert sorted(served[:len(primaries)]) == sorted(primaries), (
                        f"{role}: chain {served} does not cover every primary "
                        f"{primaries}; a saturated lead would lose the call"
                    )
                    # Behind the voters come the non-voting fallbacks, in
                    # production order, so a voter that fails degrades just as
                    # the single-chain forecaster would.
                    expected_tail = [m for m in _fallbacks_for(role) if m not in primaries]
                    assert served[len(primaries):] == expected_tail, (
                        f"{role}: chain tail {served[len(primaries):]} != "
                        f"non-voting fallbacks {expected_tail}"
                    )
                # An unregistered limiter key resolves to an UNTHROTTLED
                # limiter, which would silently disable Groq's measured 8000
                # TPM cap. Checked against RATE_LIMITED_MODELS, which
                # test_pin_models_chains pins to rate_limiter.DEFAULT_LIMITS.
                for key in entry.bucket_keys:
                    assert key in RATE_LIMITED_MODELS, (
                        f"{role}: {key!r} is not a rate-limiter-registered "
                        "model, so its limiter would enforce nothing"
                    )
                continue
            assert isinstance(entry, _StubFallbackLlm), f"{role} should be wrapped"
            served = [b.model for b in entry.backends]
            assert served == expected_chain, (
                f"{role}: chain is {served}, expected {expected_chain}"
            )
        elif active_fallbacks and role == "parser":
            # Fallbacks exist but none can parse: the parser stays unwrapped
            # rather than gaining a backend that would answer in prose.
            assert isinstance(result[role], _StubGeneralLlm), (
                "parser should stay a single backend when no fallback can parse"
            )
        elif role == "parser":
            assert isinstance(result[role], str), "parser should stay a bare model string"
        else:
            assert isinstance(result[role], _StubGeneralLlm), f"{role} should be a plain backend"
    return result


def selftest() -> None:
    """Runs on every invocation. A patch step that silently no-ops is worse than
    one that crashes, so prove the three behaviours that matter before touching
    the real file. Must not assume anything about this process' own environment -
    ACTIVE_FALLBACKS reflects whatever keys are actually set when this module was
    imported, so every check below reads that real value rather than hardcoding [].
    """
    global ACTIVE_FALLBACKS, ACTIVE_ASKNEWS
    stub = IMPORT_ANCHOR + "\n" + BOT_INIT_ANCHOR + "        a=1,\n" + ANCHOR + "    )\n"
    once = patch(stub, DEFAULTS)
    assert "# llms={" not in once, "comment markers survived the patch"
    assert DEFAULTS["default"] in once
    assert DEFAULTS["parser"] in once
    ast.parse(once)                                   # the result still has to be valid Python
    _eval_llms_dict(once, ACTIVE_FALLBACKS, DEFAULTS)           # ...and evaluate to a real dict, not a set
    assert patch(once, DEFAULTS) == once              # idempotent

    try:
        patch("bot = Bot()\n", DEFAULTS)
    except LookupError:
        pass
    else:  # pragma: no cover - guards against a silently permissive matcher
        raise AssertionError("missing anchor should raise")

    # Regression guard: a real main.py has a documentation example of
    # `llms={...}` inside its module docstring, at the same indentation,
    # BEFORE the real bot instantiation. A naive whole-file text search for
    # "llms={" finds that example first and reports success while patching
    # nothing - this is what actually happened during development.
    doc_example = ANCHOR.replace("        # ", "        ")  # same block, not commented
    docstring_trap = (
        chr(39)*3 + "\n    Example:\n    ```python\n" + doc_example + "    ```\n    " + chr(39)*3 + "\n"
        + IMPORT_ANCHOR + "\n"
        + BOT_INIT_ANCHOR + "        a=1,\n" + ANCHOR + "    )\n"
    )
    trapped = patch(docstring_trap, DEFAULTS)
    # The docstring's own example model (gpt-4o) must survive untouched,
    # and the real block below it must be the one that got patched.
    assert 'model="openrouter/openai/gpt-4o"' in trapped  # docstring example untouched
    # Structural, not a string count. These assertions used to count how many
    # times DEFAULTS["default"] appeared, which silently assumed default,
    # researcher and summarizer all shared one model string. They stopped
    # sharing one on 2026-09-10 (Opus forecasts, Haiku researches) and the
    # count broke for a reason that had nothing to do with what it guarded.
    # Evaluating the block says the same thing without the assumption.
    evaluated = _eval_llms_dict(trapped, ACTIVE_FALLBACKS, DEFAULTS)
    assert set(evaluated) == {"default", "summarizer", "researcher", "parser"}

    real_block_start = trapped.index(BOT_INIT_ANCHOR)
    assert 'model="openrouter/openai/gpt-4o"' not in trapped[real_block_start:], (
        "the real block still has the old example model - docstring text leaked into the match"
    )

    # Fallback wiring: simulate both keys present, independent of this
    # process's real environment, and check the generated code is what it
    # claims to be - the import appears, and now every role including parser
    # gets every backend, since parser shares OpenRouter's account-wide daily
    # cap with the reasoning roles.
    saved_fallbacks = ACTIVE_FALLBACKS
    try:
        # Exercise the real configured chain rather than a hardcoded copy, so
        # this check cannot drift away from FALLBACK_CHAIN.
        ACTIVE_FALLBACKS = list(FALLBACK_CHAIN)
        parser_fallbacks = _fallbacks_for("parser")
        reasoning_fallbacks = _fallbacks_for("default")
        # The forecaster ensembles over one chain per primary model, and every
        # chain names every primary exactly once (same models, different lead).
        # So each model string appears: once for researcher, once for
        # summarizer, n_ens times for default, plus once more if the parser can
        # use it. Derived from _ensemble_primaries rather than hardcoded, so
        # adding a provider to FALLBACK_CHAIN cannot silently invalidate this.
        n_ens = len(_ensemble_primaries(DEFAULTS["default"]))
        with_fb = patch(stub, DEFAULTS)
        assert IMPORT_LINE in with_fb
        n_ens = len(_ensemble_primaries(DEFAULTS["default"]))
        # Same reasoning as above: assert the SHAPE, not how many times each
        # model string happens to occur. The shape is what production depends
        # on -- one ensemble over the primaries for the forecaster, a single
        # chain for the roles called once, and a parser restricted to backends
        # that emit schema-constrained JSON.
        evaluated = _eval_llms_dict(with_fb, ACTIVE_FALLBACKS, DEFAULTS)
        assert set(evaluated) == {"default", "summarizer", "researcher", "parser"}
        assert with_fb.count("BalancedLlm([") == (1 if n_ens > 1 else 0), (
            "the forecaster ensemble must emit exactly one BalancedLlm"
        )
        if n_ens > 1:
            assert BALANCED_IMPORT_LINE in with_fb
        # default contributes n_ens chains; researcher and summarizer one each.
        wrapped_roles = n_ens + 2 + (1 if parser_fallbacks else 0)
        assert with_fb.count("FallbackLlm([") == wrapped_roles, (
            "expected {0} FallbackLlm chains, got {1}".format(
                wrapped_roles, with_fb.count("FallbackLlm(["))
        )
        parser_primary = _parser_primary(DEFAULTS["parser"])
        if parser_fallbacks:
            assert "FallbackLlm([\n                GeneralLlm(model=\"" + parser_primary in with_fb, (
                "parser chain does not lead with {0}".format(parser_primary)
            )
        ast.parse(with_fb)
        _eval_llms_dict(with_fb, ACTIVE_FALLBACKS, DEFAULTS)  # real dict, correct backend count per role
        assert patch(with_fb, DEFAULTS) == with_fb  # idempotent with fallback active too

        # Import must disappear again if the chain goes back to empty (keys
        # removed) -- unless an AskNews credential still needs it for the
        # researcher's two-link chain.
        ACTIVE_FALLBACKS = []
        reverted = patch(with_fb, DEFAULTS)
        assert (IMPORT_LINE in reverted) == ACTIVE_ASKNEWS
    finally:
        ACTIVE_FALLBACKS = saved_fallbacks

    # AskNews wiring, simulated both ways independent of this process's
    # environment. With a credential the researcher leads with the news feed
    # and the configured LLM stands directly behind it -- with no fallback
    # key present too, which is the case that needs the import on its own.
    # Without one, nothing in the block mentions AskNews.
    saved_asknews = ACTIVE_ASKNEWS
    try:
        for fallbacks in ([], list(FALLBACK_CHAIN)):
            ACTIVE_FALLBACKS = fallbacks
            ACTIVE_ASKNEWS = True
            with_news = patch(stub, DEFAULTS)
            assert IMPORT_LINE in with_news
            evaluated = _eval_llms_dict(with_news, ACTIVE_FALLBACKS, DEFAULTS)
            served = [b.model for b in evaluated["researcher"].backends]
            assert served[:2] == [ASKNEWS_MODEL, DEFAULTS["researcher"]], served
            for role in ("default", "summarizer", "parser"):
                assert ASKNEWS_MODEL not in _role_expr(role, DEFAULTS[role]), (
                    f"AskNews leaked into the {role} role"
                )
            assert patch(with_news, DEFAULTS) == with_news  # idempotent
            # Credential removed: the block converges back, in place.
            ACTIVE_ASKNEWS = False
            without_news = patch(with_news, DEFAULTS)
            assert ASKNEWS_MODEL not in without_news
            assert without_news == patch(stub, DEFAULTS), (
                "removing the AskNews credential must give the same block as "
                "never having had one"
            )
    finally:
        ACTIVE_FALLBACKS = saved_fallbacks
        ACTIVE_ASKNEWS = saved_asknews

    tmp = pathlib.Path("/tmp/_pin_models_selftest.txt")
    tmp.write_text("parser: x/y  # trailing comment\n\n# whole-line comment\n")
    assert read_overrides(tmp) == {"parser": "x/y"}
    tmp.write_text("nonsense: z\n")
    try:
        read_overrides(tmp)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown role should raise")
    tmp.unlink()


def _ping(model: str) -> str | None:
    """One minimal completion. Returns None on success, the error text on failure."""
    if model.startswith("asknews/"):
        return _ping_asknews(model)
    import litellm

    try:
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=16,
            timeout=60,
        )
        print(f"  ok    {model}")
        return None
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        head = str(exc).replace("\n", " ")[:300]
        print(f"  FAIL  {model}: {head}")
        return head


def _ping_asknews(model: str) -> str | None:
    """One real news search through the same path the bot uses at runtime.

    litellm has no AskNews provider, so the generic ping above would report
    this model dead on every run and say nothing about the credential. This
    asks AskNewsSearcher directly, which is what GeneralLlm._call_asknews
    does, so a FAIL here is the failure the researcher's first link would hit
    -- and a FAIL here alone never fails the role, because the configured LLM
    stands behind it in the chain.
    """
    import asyncio

    from forecasting_tools import AskNewsSearcher

    try:
        asyncio.run(
            AskNewsSearcher().call_preconfigured_version(
                model, "What is the latest news about the Metaculus forecasting platform?"
            )
        )
        print(f"  ok    {model}")
        return None
    except Exception as exc:  # noqa: BLE001 - any failure here is disqualifying
        head = str(exc).replace("\n", " ")[:300]
        print(f"  FAIL  {model}: {head}")
        return head


def check_models(models: dict[str, str]) -> list[str]:
    """Ping every distinct model that could actually be called at runtime -
    primaries and every fallback backend - and return the roles that have NO
    working backend at all. A role with a dead primary but a live fallback is
    NOT a failure: that is exactly the scenario fallback exists for. Testing
    only `models` (the primaries) would make --check block on a transient
    OpenRouter outage that FallbackLlm would have silently absorbed."""
    # The chain each role tries at runtime, from the same helper the
    # evaluator uses, so --check pings AskNews when the researcher leads with
    # it, and pings the parser-only extra (PARSER_EXTRA_CHAIN), which the
    # previous primaries-plus-ACTIVE_FALLBACKS set never reached. When
    # fallback is active, parser is wrapped exactly like the reasoning roles
    # (see REASONING_ROLES / docstring). With no fallback keys present every
    # chain is its primary alone and this degrades to "dead primary = dead
    # role", the pre-fallback behavior.
    chains = {role: _runtime_chain(role, models[role]) for role in DEFAULTS}
    all_models = sorted({m for chain in chains.values() for m in chain})
    results = {m: _ping(m) for m in all_models}

    dead_roles = []
    for role in ("default", "researcher", "summarizer", "parser"):
        if all(results[m] is not None for m in chains[role]):
            dead_roles.append(role)
    return dead_roles


def main() -> int:
    selftest()

    path = pathlib.Path("main.py")
    if not path.exists():
        print("main.py not found; run this from the repo root", file=sys.stderr)
        return 1

    models = dict(DEFAULTS)
    try:
        overrides = read_overrides(OVERRIDE_FILE)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    models.update(overrides)

    src = path.read_text()
    try:
        patched = patch(src, models)
    except LookupError as exc:
        print(exc, file=sys.stderr)
        return 1

    if patched == src:
        print("llms block already active, nothing to do")
    else:
        ast.parse(patched)
        path.write_text(patched)
        for role in ("default", "researcher", "summarizer", "parser"):
            mark = " (from models.txt)" if role in overrides else ""
            print(f"  {role:<11} {models[role]}{mark}")
        if _OPUS_EFFORT_OVERRIDE in OPUS_EFFORTS:
            print(f"  opus forecaster reasoning effort {FORECASTER_REASONING[OPUS_FORECASTER]}"
                  " (FORECASTER_REASONING_OPUS)")

    # Deliberately outside the branch above: an already-patched tree still has to
    # prove its models answer, otherwise --check silently passes on a rerun.
    if "--check" in sys.argv:
        print("checking every model answers before the bot burns a run on them")
        if ACTIVE_FALLBACKS:
            chain = " -> ".join([m for m, _e in [(models["default"], None)] + ACTIVE_FALLBACKS])
            print(f"  fallback chain per role (default/researcher/summarizer/parser): {chain}")
        dead_roles = check_models(models)
        if dead_roles:
            print(f"role(s) with no working backend at all: {', '.join(dead_roles)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
