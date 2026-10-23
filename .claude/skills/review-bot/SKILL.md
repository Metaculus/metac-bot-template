---
name: review-bot
description: Review how this forecasting bot actually performed on resolved Metaculus questions — build the outcome table, read the reports it posted, work out why the worst questions went wrong, and write a review. Use when asked to review bot performance, analyse scores, or work out why a forecast went wrong.
---

# Review this bot's forecasts

Diagnose *why* the bot scored the way it did on questions that have resolved. Everything here
is read-only Metaculus API traffic. No forecasting, no publishing, no API spend.

## Hard rules

- **Never run the bot.** `python main.py` spends money and publishes comments. This skill
  only reads.
- **Never read a whole report.** Pull one section of one question with `show`.
- **Ask before changing code.** Diagnose first, propose fixes in the review.
- Metaculus rate-limits at roughly 5 requests/second and the client does not retry. A 429
  means rerun the command.

## 1. Build the data

`bot-review` comes from the optional `metaculus-bot-review` package. `uv run --group integrations`
installs it on first use.

```bash
uv run --group integrations bot-review review --tournament <slug-or-id> \
    --output review.json --summary review-summary.md
```

Also `--post <POST_ID> ...` for specific questions, `--resolved-since <DAYS>` for anything
that resolved recently, `--from-json review.json` to re-render without refetching. Roughly one
request per post.

Read the summary first, then `review.json` for per-question detail.

## 2. Pick questions to investigate

Around ten, fewer once findings repeat. Most of the budget goes on the worst scores. Also:

- **Forecasted but unscored while resolved** — annulled, or something odder.
- **`traces` non-empty but `trace` null** — every run came after `spot_scoring_time`, so the
  forecast was never counted.
- **Fewer forecasters than the bot's usual** — compare `len(trace.forecasters)` across the
  table.
- **`trace.truncated` true** — the comment hit Metaculus's size cap and only its opening
  survives. Report it as a lost trace; there is nothing to diagnose.
- **No trace at all** — the bot never forecast, or never published a comment.

Those four are mechanical: no reading needed. When one of them matches many questions, that is
a single finding with a count, not a dozen questions to read.

Rank on whichever score the leaderboard uses — the summary names it. Peer scores are relative
to other forecasters, so negative means you did worse than the field, and the tournament score
is the sum.

Well-scored questions are not routine reading; they rarely produce a fix. Read one only to
test a hypothesis the losses raised, and only after checking whether the table can test it for
free — a claimed bias is usually answerable by comparing forecasts to resolutions across every
question. A list of losses looks like a systematic bias whether or not one exists, so test the
patterns you think you see.

## 3. Read the report, cheapest first

Stop as soon as the question is answered.

**a. The trace** — free, already in `review.json`. Every forecaster's prediction, the count,
the run time. A tight cluster that all missed points at a shared input or blind spot; a wide
spread that landed badly points at aggregation; one model alone and right is worth naming.

**b. Search the research** — near-free. You know what the question resolved to, so search for
it rather than reading:

```bash
uv run --group integrations bot-review show <POST_ID> --comment <COMMENT_ID> \
    --section research | grep -niE "<the deciding fact>" | head
```

Found → the research had it and the forecasters underused it. Absent → likely a research gap.

**c. One or two rationales** — the outlier, or one from the cluster if they all agreed.

```bash
uv run --group integrations bot-review show <POST_ID> --comment <COMMENT_ID> --forecaster R1:F3
```

**d. The whole research section** — last resort, when the searches came up empty and you need
to know what the bot actually had. A research section dwarfs a rationale, so these are the
reads to ration.

Take `<COMMENT_ID>` from the question's `trace.comment_id`: on a question forecast more than
once it names the run that was standing when the question was scored, which is the run whose
reasoning affected the score. Without it you get the latest run. Forecaster keys
(`R<report>:F<forecaster>`) and their predictions are in `trace.forecasters`; a key identifies
a forecaster across questions, so it is what to use when naming a repeat outlier.

Also worth asking: did the final prediction match where the reasoning pointed, and did the bot
answer the question the resolution criteria actually asked?

## 4. Write the review

Write `review.md`:

- **Summary** — score, rank, and the two or three things that actually moved it.
- **Per question** — title and link, score, and what went wrong in your own words, with the
  quote or absence from the report that shows it. One or two sentences each.
- **Patterns** — causes that recur, and any model repeatedly out on its own.
- **Suggestions** — concrete: a prompt change, a research gap, a scheduling fix. Name the
  questions each would have helped and drop anything you cannot tie to one.

"Well forecast, unlucky outcome" is a real finding. Do not manufacture a process failure for a
question the bot handled well. State plainly where the evidence is thin — three well-supported
findings beat ten guesses.
