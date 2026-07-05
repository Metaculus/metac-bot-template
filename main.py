import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import dotenv
import requests

from digest_mode import (
    extract_tournament_identifier,
    load_tournament_identifiers,
    matrix_send_message,
    run_digest,
)
from tournament_update import select_questions_for_tournament_update
from forecasting_tools import GeneralLlm, MetaculusClient

from metaculus_bot import MetaculusBot
from retrospective_mode import run_retrospective
from weekly_retrospective_mode import run_weekly_retrospective


def _notify_matrix_on_submit(
    *, run_mode: str, forecast_reports: list[object]
) -> None:
    if run_mode not in {"tournament", "tournament_update", "metaculus_cup"}:
        return
    successful_reports = [
        report
        for report in forecast_reports
        if not isinstance(report, BaseException)
    ]
    failed_reports = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]

    notify_always = (
        os.getenv("MATRIX_NOTIFY_ALWAYS", "").strip().lower()
        in {"1", "true", "yes", "y"}
    )
    if not notify_always and not successful_reports and not failed_reports:
        return

    now_iso = datetime.now(timezone.utc).isoformat()
    lines = [
        f"Metaculus bot run ({now_iso[:10]}): {len(successful_reports)} submitted, {len(failed_reports)} failed"
    ]
    for report in successful_reports[:10]:
        question = getattr(report, "question", None)
        if question is None:
            continue
        question_text = getattr(question, "question_text", "(unknown question)")
        page_url = getattr(question, "page_url", None) or f"post:{getattr(question, 'id_of_post', '')}"
        lines.append(f"- {question_text} ({page_url})")
    if len(successful_reports) > 10:
        lines.append(f"... and {len(successful_reports) - 10} more")
    if failed_reports:
        lines.append("")
        lines.append("Failures (first 5):")
        for err in failed_reports[:5]:
            lines.append(f"- {repr(err)[:300]}")
    matrix_send_message("\n".join(lines))


def _validate_and_normalize_metaculus_token() -> None:
    token_raw = os.getenv("METACULUS_TOKEN", "")
    token = token_raw.strip()
    if token != token_raw:
        os.environ["METACULUS_TOKEN"] = token

    if not token:
        message = (
            "METACULUS_TOKEN is missing. Set it as a GitHub Actions Repository secret, "
            "or (if you stored it as an Environment secret) add `environment: <name>` to the workflow job."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)

    lowered = token.lower()
    if lowered.startswith("token ") or lowered.startswith("bearer "):
        message = (
            "METACULUS_TOKEN should be the raw token value (no 'Token ' / 'Bearer ' prefix). "
            "Update the secret/env var to only the token string."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)

    if any(ch.isspace() for ch in token):
        message = (
            "METACULUS_TOKEN contains whitespace. Re-copy the token (no spaces/newlines) and update the secret."
        )
        matrix_send_message(f"Metaculus bot error: {message}")
        raise SystemExit(message)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            f"Ignoring invalid integer for {name}: {raw!r}"
        )
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logging.getLogger(__name__).warning(f"Ignoring invalid boolean for {name}: {raw!r}")
    return default


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        raw = value.strip()
        try:
            if raw.endswith("Z"):
                raw = f"{raw[:-1]}+00:00"
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


def _first_question_datetime(question: object, attr_names: tuple[str, ...]) -> datetime | None:
    for attr_name in attr_names:
        value = getattr(question, attr_name, None)
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _question_identity(question: object) -> tuple[object, ...]:
    id_of_post = getattr(question, "id_of_post", None)
    id_of_question = getattr(question, "id_of_question", None)
    group_option = getattr(question, "group_question_option", None)
    conditional_type = getattr(question, "conditional_type", None)
    if id_of_post or id_of_question:
        return (id_of_post, id_of_question, group_option, conditional_type)
    return (
        getattr(question, "page_url", None),
        getattr(question, "question_text", None),
        group_option,
        conditional_type,
    )


def _question_url(question: object) -> str:
    page_url = getattr(question, "page_url", None)
    if page_url:
        return str(page_url)
    id_of_post = getattr(question, "id_of_post", None)
    if id_of_post:
        return f"https://www.metaculus.com/questions/{id_of_post}"
    return "(unknown url)"


def _log_question_window(question: object, *, tournament_id: str) -> None:
    logger = logging.getLogger(__name__)
    open_time = _first_question_datetime(
        question,
        (
            "open_time",
            "published_at",
            "publish_time",
            "created_time",
            "created_at",
            "scheduled_open_time",
        ),
    )
    close_time = _first_question_datetime(
        question,
        (
            "close_time",
            "scheduled_close_time",
            "actual_close_time",
            "scheduled_resolution_time",
        ),
    )
    now = datetime.now(timezone.utc)
    duration_hours = (
        (close_time - open_time).total_seconds() / 3600
        if open_time is not None and close_time is not None
        else None
    )
    remaining_hours = (
        (close_time - now).total_seconds() / 3600
        if close_time is not None
        else None
    )
    question_text = str(getattr(question, "question_text", "(unknown question)"))
    logger.info(
        "Open question window: tournament=%s url=%s already_forecasted=%s "
        "open_utc=%s close_utc=%s duration_hours=%s remaining_hours=%s text=%s",
        tournament_id,
        _question_url(question),
        getattr(question, "already_forecasted", None),
        open_time.isoformat() if open_time is not None else "(unknown)",
        close_time.isoformat() if close_time is not None else "(unknown)",
        f"{duration_hours:.2f}" if duration_hours is not None else "(unknown)",
        f"{remaining_hours:.2f}" if remaining_hours is not None else "(unknown)",
        question_text[:180],
    )


def _collect_open_questions_from_tournaments(
    *, client: MetaculusClient, tournaments: list[str]
) -> tuple[list[Any], list[BaseException]]:
    logger = logging.getLogger(__name__)
    questions: list[Any] = []
    failures: list[BaseException] = []
    seen: set[tuple[object, ...]] = set()

    for tournament_id in tournaments:
        try:
            tournament_questions = client.get_all_open_questions_from_tournament(
                tournament_id
            )
        except BaseException as e:
            logger.error("Failed to scan tournament '%s': %s", tournament_id, e)
            failures.append(e)
            continue

        logger.info(
            "Scanned tournament '%s': %s open question(s)",
            tournament_id,
            len(tournament_questions),
        )
        for question in tournament_questions:
            _log_question_window(question, tournament_id=tournament_id)
            identity = _question_identity(question)
            if identity in seen:
                logger.info(
                    "Skipping duplicate open question from tournament '%s': %s",
                    tournament_id,
                    _question_url(question),
                )
                continue
            seen.add(identity)
            questions.append(question)

    logger.info(
        "Open question scan complete: tournaments=%s unique_open_questions=%s failures=%s",
        ",".join(tournaments),
        len(questions),
        len(failures),
    )
    return questions, failures


if __name__ == "__main__":
    dotenv.load_dotenv()
    try:
        openclaw_env_path = Path.home() / ".openclaw" / ".env"
        if openclaw_env_path.exists():
            dotenv.load_dotenv(openclaw_env_path, override=False)
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(description="Run the Metaculus forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tournament",
            "tournament_update",
            "metaculus_cup",
            "test_questions",
            "digest",
            "retrospective",
            "weekly_retrospective",
        ],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--tournaments-file",
        type=str,
        default=None,
        help="Path to a file containing tournament URLs/slugs (one per line).",
    )
    parser.add_argument(
        "--tournament",
        action="append",
        default=[],
        help="Tournament slug/id or URL (repeatable). Extends --tournaments-file.",
    )
    parser.add_argument(
        "--researcher",
        type=str,
        default=None,
        help=(
            "Override research strategy/model. Examples: "
            "'no_research', 'free/nasdaq-eps', 'asknews/news-summaries', "
            "'asknews/deep-research/low-depth', 'smart-searcher/<model>', "
            "'tavily-searcher/<model>'."
        ),
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=None,
        help=(
            "Override the default LLM model (Litellm format), e.g. "
            "'openrouter/openai/gpt-oss-120b:free' or 'openrouter/openai/gpt-4o'."
        ),
    )
    parser.add_argument(
        "--parser-model",
        type=str,
        default=None,
        help=(
            "Override the parser LLM model used for structured output, e.g. "
            "'openrouter/openai/gpt-oss-120b:free'."
        ),
    )
    parser.add_argument(
        "--submit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Submit forecasts to Metaculus. Defaults: tournament/metaculus_cup=True, "
            "test_questions=False, digest=False."
        ),
    )
    args = parser.parse_args()
    run_mode: Literal[
        "tournament",
        "tournament_update",
        "metaculus_cup",
        "test_questions",
        "digest",
        "retrospective",
        "weekly_retrospective",
    ] = (
        args.mode
    )

    _validate_and_normalize_metaculus_token()

    default_submit = run_mode in {"tournament", "tournament_update", "metaculus_cup"}
    if run_mode == "test_questions":
        default_submit = False
    publish_to_metaculus = default_submit if args.submit is None else bool(args.submit)
    if run_mode == "digest" and publish_to_metaculus:
        logging.getLogger(__name__).warning(
            "--submit ignored in digest mode (no submission)."
        )
        publish_to_metaculus = False
    if run_mode == "retrospective" and publish_to_metaculus:
        logging.getLogger(__name__).warning(
            "--submit ignored in retrospective mode (no submission)."
        )
        publish_to_metaculus = False
    if run_mode == "weekly_retrospective" and publish_to_metaculus:
        logging.getLogger(__name__).warning(
            "--submit ignored in weekly_retrospective mode (no submission)."
        )
        publish_to_metaculus = False

    if run_mode == "retrospective":
        client = MetaculusClient()
        if args.tournaments_file or args.tournament:
            tournaments, unsupported = load_tournament_identifiers(
                args.tournaments_file, args.tournament
            )
            for item in unsupported:
                logging.getLogger(__name__).warning(
                    f"Unsupported collection (not a tournament): {item}"
                )
            if not tournaments:
                raise SystemExit(
                    "No valid tournaments configured via --tournaments-file/--tournament."
                )
        else:
            market_pulse_env = os.getenv("MARKET_PULSE_TOURNAMENT", "").strip()
            default_raw = market_pulse_env or client.CURRENT_MARKET_PULSE_ID
            default_id = extract_tournament_identifier(default_raw)
            if not default_id or default_id.startswith("index:"):
                raise SystemExit(
                    "No tournament specified. Set MARKET_PULSE_TOURNAMENT or pass --tournament."
                )
            tournaments = [default_id]

        out_dir = Path("reports") / "retrospective"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for tournament_id in tournaments:
            safe_id = tournament_id.replace("/", "_").replace(":", "_")
            out_path = out_dir / f"{safe_id}_{stamp}.md"
            markdown = asyncio.run(
                run_retrospective(
                    client=client,
                    tournament_id=tournament_id,
                    out_path=out_path,
                )
            )
            print(markdown)
        raise SystemExit(0)

    if run_mode == "weekly_retrospective":
        client = MetaculusClient()
        if args.tournaments_file or args.tournament:
            tournaments, unsupported = load_tournament_identifiers(
                args.tournaments_file, args.tournament
            )
            for item in unsupported:
                logging.getLogger(__name__).warning(
                    f"Unsupported collection (not a tournament): {item}"
                )
            if not tournaments:
                raise SystemExit(
                    "No valid tournaments configured via --tournaments-file/--tournament."
                )
        else:
            market_pulse_env = os.getenv("MARKET_PULSE_TOURNAMENT", "").strip()
            market_raw = market_pulse_env or client.CURRENT_MARKET_PULSE_ID
            market_id = extract_tournament_identifier(market_raw)

            aib_raw = os.getenv("AIB_TOURNAMENT", "").strip() or client.CURRENT_AI_COMPETITION_ID
            aib_id = extract_tournament_identifier(aib_raw)

            minibench_id = extract_tournament_identifier(client.CURRENT_MINIBENCH_ID)

            cup_raw = (
                os.getenv("METACULUS_CUP_TOURNAMENT", "").strip()
                or client.CURRENT_METACULUS_CUP_ID
            )
            cup_id = extract_tournament_identifier(cup_raw)

            tournaments = [
                t
                for t in [market_id, aib_id, minibench_id, cup_id]
                if t and not t.startswith("index:")
            ]
            if not tournaments:
                raise SystemExit(
                    "No tournaments configured for weekly_retrospective. Set MARKET_PULSE_TOURNAMENT/AIB_TOURNAMENT/METACULUS_CUP_TOURNAMENT or pass --tournament."
                )

        out_dir = Path("reports") / "retrospective_weekly"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = out_dir / f"weekly_{stamp}.md"
        state_path = Path(".state") / "weekly_retrospective_state.json"
        markdown = asyncio.run(
            run_weekly_retrospective(
                client=client,
                tournaments=tournaments,
                out_path=out_path,
                state_path=state_path,
            )
        )
        print(markdown)
        raise SystemExit(0)

    llms: dict[str, object] | None = None
    if args.researcher or args.default_model or args.parser_model:
        # Start from the bot defaults so partial CLI overrides (e.g.
        # `--researcher ...`) don't accidentally drop required LLM config and
        # fall back to forecasting-tools' built-in defaults.
        llms = {
            k: v
            for k, v in MetaculusBot._llm_config_defaults().items()
            if v is not None
        }
    if args.researcher and llms is not None:
        llms["researcher"] = args.researcher

    max_tokens = _env_int("BOT_MAX_TOKENS", 0)
    llm_max_tokens_kwargs: dict[str, int] = (
        {"max_tokens": max_tokens} if max_tokens > 0 else {}
    )
    if args.default_model and llms is not None:
        default_llm = GeneralLlm(
            model=args.default_model,
            temperature=0.3,
            timeout=60,
            allowed_tries=2,
            **llm_max_tokens_kwargs,
        )
        llms["default"] = default_llm
        llms["summarizer"] = GeneralLlm(
            model=args.default_model,
            temperature=0.0,
            timeout=60,
            allowed_tries=2,
            **llm_max_tokens_kwargs,
        )
    if args.parser_model and llms is not None:
        parser_llm = GeneralLlm(
            model=args.parser_model,
            temperature=0.0,
            timeout=60,
            allowed_tries=2,
            **llm_max_tokens_kwargs,
        )
        llms["parser"] = parser_llm
        llms["router"] = parser_llm

    research_reports_per_question = _env_int("BOT_RESEARCH_REPORTS_PER_QUESTION", 1)
    predictions_per_research_report = _env_int(
        "BOT_PREDICTIONS_PER_RESEARCH_REPORT", 1
    )
    if research_reports_per_question <= 0:
        raise SystemExit("BOT_RESEARCH_REPORTS_PER_QUESTION must be >= 1")
    if predictions_per_research_report <= 0:
        raise SystemExit("BOT_PREDICTIONS_PER_RESEARCH_REPORT must be >= 1")

    summarize_env_is_set = os.getenv("BOT_ENABLE_SUMMARIZE_RESEARCH") is not None
    enable_summarize_research = _env_bool("BOT_ENABLE_SUMMARIZE_RESEARCH", True)
    if not summarize_env_is_set and args.researcher:
        if args.researcher.strip().lower() == "no_research":
            enable_summarize_research = False

    bot = MetaculusBot(
        research_reports_per_question=research_reports_per_question,
        predictions_per_research_report=predictions_per_research_report,
        use_research_summary_to_forecast=False,
        llms=llms,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=(run_mode == "tournament"),
        extra_metadata_in_explanation=True,
        enable_summarize_research=enable_summarize_research,
    )

    client = MetaculusClient()
    forecast_reports: list[object] = []

    try:
        if run_mode == "tournament":
            if args.tournaments_file or args.tournament:
                tournaments, unsupported = load_tournament_identifiers(
                    args.tournaments_file, args.tournament
                )
                for item in unsupported:
                    logging.getLogger(__name__).warning(
                        f"Unsupported collection (not a tournament): {item}"
                    )
                if not tournaments:
                    raise SystemExit(
                        "No valid tournaments configured via --tournaments-file/--tournament."
                    )
                questions, scan_failures = _collect_open_questions_from_tournaments(
                    client=client,
                    tournaments=tournaments,
                )
                forecast_reports.extend(scan_failures)
                if questions:
                    forecast_reports.extend(
                        asyncio.run(
                            bot.forecast_questions(
                                questions, return_exceptions=True
                            )
                        )
                    )
            else:
                default_tournaments = [
                    client.CURRENT_AI_COMPETITION_ID,
                    client.CURRENT_MINIBENCH_ID,
                ]
                questions, scan_failures = _collect_open_questions_from_tournaments(
                    client=client,
                    tournaments=default_tournaments,
                )
                forecast_reports.extend(scan_failures)
                if questions:
                    forecast_reports.extend(
                        asyncio.run(
                            bot.forecast_questions(
                                questions, return_exceptions=True
                            )
                        )
                    )

        elif run_mode == "tournament_update":
            bot.skip_previously_forecasted_questions = False

            if args.tournaments_file or args.tournament:
                tournaments, unsupported = load_tournament_identifiers(
                    args.tournaments_file, args.tournament
                )
                for item in unsupported:
                    logging.getLogger(__name__).warning(
                        f"Unsupported collection (not a tournament): {item}"
                    )
                if not tournaments:
                    raise SystemExit(
                        "No valid tournaments configured via --tournaments-file/--tournament."
                    )
            else:
                market_pulse_env = os.getenv("MARKET_PULSE_TOURNAMENT", "").strip()
                default_raw = market_pulse_env or client.CURRENT_MARKET_PULSE_ID
                default_id = extract_tournament_identifier(default_raw)
                if not default_id or default_id.startswith("index:"):
                    raise SystemExit(
                        "No tournament specified. Set MARKET_PULSE_TOURNAMENT or pass --tournament."
                    )
                tournaments = [default_id]

            for tournament_id in tournaments:
                questions_to_forecast, counts = select_questions_for_tournament_update(
                    client=client,
                    tournament_id=tournament_id,
                )
                logging.getLogger(__name__).info(
                    "Tournament update scan (%s): total_open=%s, queued_unforecasted=%s, queued_cp_changed=%s, queued_diverged_from_cp=%s, skipped_missing_cp=%s, skipped_missing_my_forecast=%s",
                    tournament_id,
                    counts.get("total_open"),
                    counts.get("queued_unforecasted"),
                    counts.get("queued_cp_changed"),
                    counts.get("queued_diverged_from_cp"),
                    counts.get("skipped_missing_cp"),
                    counts.get("skipped_missing_my_forecast"),
                )
                if not questions_to_forecast:
                    continue
                forecast_reports.extend(
                    asyncio.run(
                        bot.forecast_questions(
                            questions_to_forecast, return_exceptions=True
                        )
                    )
                )

        elif run_mode == "metaculus_cup":
            bot.skip_previously_forecasted_questions = False
            cup_override_raw = os.getenv("METACULUS_CUP_TOURNAMENT", "").strip()
            cup_override = (
                extract_tournament_identifier(cup_override_raw)
                if cup_override_raw
                else None
            )
            if cup_override and cup_override.startswith("index:"):
                logging.getLogger(__name__).warning(
                    f"Ignoring METACULUS_CUP_TOURNAMENT={cup_override_raw!r} because it is an index, not a tournament."
                )
                cup_override = None

            metaculus_cup_id = cup_override or client.CURRENT_METACULUS_CUP_ID
            forecast_reports = asyncio.run(
                bot.forecast_on_tournament(
                    metaculus_cup_id, return_exceptions=True
                )
            )

        elif run_mode == "test_questions":
            # Quick test: only 1 question by default (set BOT_TEST_QUESTION_COUNT for more)
            ALL_EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Binary
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Numeric
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Multiple Choice
                "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Discrete
            ]
            test_count = _env_int("BOT_TEST_QUESTION_COUNT", 1)
            EXAMPLE_QUESTIONS = ALL_EXAMPLE_QUESTIONS[:test_count]
            bot.skip_previously_forecasted_questions = False
            questions = [
                client.get_question_by_url(question_url)
                for question_url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                bot.forecast_questions(questions, return_exceptions=True)
            )

        elif run_mode == "digest":
            tournaments_file = args.tournaments_file or "tracked_tournaments.txt"
            tournaments, unsupported = load_tournament_identifiers(
                tournaments_file, args.tournament
            )
            for item in unsupported:
                logging.getLogger(__name__).warning(
                    f"Unsupported collection (not a tournament): {item}"
                )
            if not tournaments:
                raise SystemExit(
                    "No tournaments configured. Add URLs/slugs to tracked_tournaments.txt or pass --tournament."
                )

            bot.publish_reports_to_metaculus = False
            bot.skip_previously_forecasted_questions = False
            state_path = Path(".state") / "digest_state.json"
            out_dir = Path("reports") / "digest"
            asyncio.run(
                run_digest(
                    bot=bot,
                    tournaments=tournaments,
                    state_path=state_path,
                    out_dir=out_dir,
                )
            )
            forecast_reports = []
    except requests.exceptions.HTTPError as e:
        if "Invalid token" in str(e):
            message = (
                "Metaculus API rejected your METACULUS_TOKEN (Invalid token). "
                "Double-check you copied the token from Metaculus account settings, "
                "and that GitHub Actions is actually receiving it (Repository secret vs Environment secret)."
            )
            matrix_send_message(f"Metaculus bot error: {message}")
            raise SystemExit(message) from e
        raise

    fail_on_errors = _env_bool("BOT_FAIL_ON_ERRORS", run_mode == "test_questions")
    try:
        bot.log_report_summary(  # type: ignore[arg-type]
            forecast_reports, raise_errors=fail_on_errors
        )
    finally:
        if publish_to_metaculus:
            _notify_matrix_on_submit(
                run_mode=run_mode, forecast_reports=forecast_reports
            )
