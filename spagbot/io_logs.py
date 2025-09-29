# io_logs.py
# =============================================================================
# Spagbot logging utilities:
# - Append-only forecasts CSV
# - Human-readable run logs (.md or .txt)
# - Optional git commit/push of changed logs (works nicely in GitHub Actions)
#
# HOW IT WORKS (high level):
# 1) All logs live under: forecast_logs/
#       - forecasts.csv          (master, append-only CSV of all forecasts)
#       - runs/<run_id>.md       (or .txt) human-readable run log files
#
# 2) Appending to forecasts.csv:
#       - First write creates the file and header row.
#       - Subsequent writes append rows; no rotation or per-run file creation.
#
# 3) Human logs:
#       - Create a new file per run ID (you can use your timestamp/id).
#       - You can write multiple times to the same run file.
#
# 4) Auto-commit & push to the repo:
#       - If running in GitHub Actions (GITHUB_ACTIONS=="true"), we auto-commit
#         and push logs unless DISABLE_GIT_LOGS is set to "1"/"true".
#       - Locally, we only commit/push if COMMIT_LOGS is set to "1"/"true".
#
# 5) Config via environment (all optional):
#       LOGS_BASE_DIR        default: "forecast_logs"
#       HUMAN_LOG_EXT        default: "md"   (use "txt" if you prefer)
#       DISABLE_GIT_LOGS     default: ""     (set "1"/"true" to skip commits in CI)
#       COMMIT_LOGS          default: ""     (set "1"/"true" locally to commit)
#       GIT_LOG_MESSAGE      default: "chore(logs): update Spagbot logs"
#       GIT_REMOTE_NAME      default: "origin"
#       GIT_BRANCH_NAME      default: (auto-detect; falls back to "main")
#
# This module is intentionally standalone and does not import other spagbot
# modules to avoid circular imports.
# =============================================================================

from __future__ import annotations

import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

# ------------------------------
# Paths & basic configuration
# ------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    """Return True if the env var is a truthy string ("1", "true", "yes")."""
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class LogPaths:
    base_dir: Path
    forecasts_csv: Path
    runs_dir: Path
    human_ext: str  # "md" or "txt"


def get_log_paths() -> LogPaths:
    """
    Resolve base logging directories and filenames.

    Behavior:
    - If FORECASTS_CSV_PATH is set, use exactly that path for the CSV and ensure its parent
      directory exists. Human logs still go under LOGS_BASE_DIR/runs by default.
    - Otherwise, use LOGS_BASE_DIR/forecasts.csv (default LOGS_BASE_DIR="forecast_logs").
    - HUMAN_LOG_EXT controls .md/.txt for human logs.
    """
    # Human log base & extension
    base = Path(os.getenv("LOGS_BASE_DIR", "forecast_logs")).resolve()
    human_ext = os.getenv("HUMAN_LOG_EXT", "md").strip().lower()
    if human_ext not in {"md", "txt"}:
        human_ext = "md"

    # CSV location: prefer explicit FORECASTS_CSV_PATH if provided
    csv_env = os.getenv("FORECASTS_CSV_PATH", "").strip()
    if csv_env:
        forecasts_csv = Path(csv_env).resolve()
        forecasts_csv.parent.mkdir(parents=True, exist_ok=True)
        # Keep human logs alongside default base unless user also tweaks LOGS_BASE_DIR
        runs = base / "runs"
        runs.mkdir(parents=True, exist_ok=True)
    else:
        base.mkdir(parents=True, exist_ok=True)
        runs = base / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        forecasts_csv = base / "forecasts.csv"

    return LogPaths(
        base_dir=base,
        forecasts_csv=forecasts_csv,
        runs_dir=runs,
        human_ext=human_ext,
    )



# ------------------------------
# CSV append utilities
# ------------------------------

def _write_csv_header_if_needed(csv_path: Path, fieldnames: Sequence[str]) -> None:
    """Create CSV with header row if file does not exist or is empty."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_forecast_row(row: Dict[str, object], field_order: Optional[Sequence[str]] = None) -> Path:
    """
    Append a single forecast row to forecasts.csv.
    - Creates the file and writes headers on first use.
    - If field_order is provided, columns will be written in that order; any
      extra keys in `row` will be appended to the end (stable order).
    - Returns the path to forecasts.csv

    Example row keys you might include (free-form):
      {
        "run_id": "...",
        "timestamp_ist": "...",
        "metaculus_qid": 123,
        "post_id": 456,
        "question": "...",
        "type": "binary|numeric|mcq",
        "final_prob_yes": 0.37,
        "final_point_estimate": 42.1,
        "confidence": 0.8,
        "providers_used": "gpt-4o, claude-3.7, ...",
        "gtmc1_used": True,
        "notes": "any text"
      }
    """
    paths = get_log_paths()

    # Determine fieldnames deterministically
    keys = list(row.keys())
    if field_order:
        ordered = list(field_order) + [k for k in keys if k not in field_order]
    else:
        ordered = keys

    # Ensure header exists
    _write_csv_header_if_needed(paths.forecasts_csv, ordered)

    # If the existing CSV has a different header, we extend to superset
    # (helps when runs evolve and add new columns).
    # This is a simple approach: read first line only.
    with paths.forecasts_csv.open("r", encoding="utf-8") as existing:
        reader = csv.reader(existing)
        try:
            current_header = next(reader)
        except StopIteration:
            current_header = []

    # Build superset header if needed
    superset = list(current_header)
    for k in ordered:
        if k not in superset:
            superset.append(k)

    if superset != current_header:
        # Re-write CSV with new header (migrate existing rows).
        # This is rare, but keeps a single file over time as schema evolves.
        tmp_path = paths.forecasts_csv.with_suffix(".tmp.csv")
        # Read all rows as dicts with current header
        existing_rows = []
        with paths.forecasts_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)
        # Write with expanded header
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=superset)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
        tmp_path.replace(paths.forecasts_csv)

    # Finally append the new row
    with paths.forecasts_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=superset)
        writer.writerow(row)

    return paths.forecasts_csv


# ------------------------------
# Human-readable run logs
# ------------------------------

def human_log_path(run_id: str) -> Path:
    """
    Return the path for the human-readable log for a given run_id.
    - You can pass any unique id, e.g., IST timestamp "2025-09-13T18-04-22+03:00"
    - Extension is controlled via HUMAN_LOG_EXT ("md" default; "txt" also supported)
    """
    p = get_log_paths()
    filename = f"{run_id}.{p.human_ext}"
    return p.runs_dir / filename


def write_human_log(run_id: str, content: str, mode: str = "w") -> Path:
    """
    Write or append to the human-readable run log.
    - mode="w" overwrites, mode="a" appends
    - Returns the file path
    """
    path = human_log_path(run_id)
    with path.open(mode, encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")
    return path


# ------------------------------
# Git commit & push utilities
# ------------------------------

def _run(cmd: Sequence[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with basic error transparency."""
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=True)


def _find_repo_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from `start` (or CWD) to find a .git directory."""
    cur = (start or Path.cwd()).resolve()
    for _ in range(10):  # don’t traverse forever
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _current_branch(repo: Path) -> str:
    """
    Try to detect the current git branch.
    In GitHub Actions, prefer GITHUB_REF_NAME. Otherwise use `git rev-parse --abbrev-ref HEAD`.
    Fallback to 'main'.
    """
    gh_branch = os.getenv("GITHUB_REF_NAME")
    if gh_branch:
        return gh_branch
    try:
        out = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()
        return out or "main"
    except Exception:
        return "main"


def _ensure_git_identity(repo: Path) -> None:
    """Set git user if not set (helps in CI)."""
    try:
        name = _run(["git", "config", "--get", "user.name"], cwd=repo, check=False).stdout.strip()
        email = _run(["git", "config", "--get", "user.email"], cwd=repo, check=False).stdout.strip()
        if not name:
            # Prefer GitHub actor if available
            gh_actor = os.getenv("GITHUB_ACTOR", "spagbot-bot")
            _run(["git", "config", "user.name", gh_actor], cwd=repo, check=False)
        if not email:
            # Use GitHub Actions bot email if none provided
            _run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], cwd=repo, check=False)
    except Exception:
        # Non-fatal; commit will still usually work
        pass


def commit_and_push_logs(changed_paths: Iterable[Path], commit_message: Optional[str] = None) -> bool:
    """
    Commit and push the given log files to the current repo branch.
    Returns True if a commit was made (and push attempted), False if skipped.
    Skips entirely if:
      - Not in a git repo, OR
      - CI auto-commit disabled, OR
      - Local auto-commit not opted-in
    """
    repo = _find_repo_root()
    if not repo:
        return False  # Not a git repo; nothing to do

    in_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    ci_disabled = _bool_env("DISABLE_GIT_LOGS", default=False)
    local_enabled = _bool_env("COMMIT_LOGS", default=False)

    # Decide if we proceed
    if in_ci:
        if ci_disabled:
            return False
        # In CI, default to committing logs unless explicitly disabled.
        should_commit = True
    else:
        # Locally, only commit if COMMIT_LOGS=true
        should_commit = local_enabled

    if not should_commit:
        return False

    # Prepare paths relative to repo root
    rel_paths = []
    for p in changed_paths:
        try:
            rel = p.relative_to(repo)
        except ValueError:
            # If the path is outside the repo (e.g., different drive), skip it
            continue
        rel_paths.append(rel)

    if not rel_paths:
        return False

    _ensure_git_identity(repo)

    # git add
    try:
        _run(["git", "add"] + [str(p) for p in rel_paths], cwd=repo, check=True)
    except subprocess.CalledProcessError as e:
        # If 'add' fails, just bail out quietly
        return False

    # If there are no changes, "git diff --cached --quiet" returns 0
    try:
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--exit-code"],
            cwd=str(repo),
            text=True,
        )
        if diff.returncode == 0:
            return False  # Nothing staged; skip
    except Exception:
        # If this check fails, attempt commit anyway.
        pass

    message = commit_message or os.getenv("GIT_LOG_MESSAGE", "chore(logs): update Spagbot logs")
    try:
        _run(["git", "commit", "-m", message], cwd=repo, check=True)
    except subprocess.CalledProcessError:
        # Commit may fail if nothing to commit; treat as no-op
        return False

    # Push
    remote = os.getenv("GIT_REMOTE_NAME", "origin")
    branch = os.getenv("GIT_BRANCH_NAME", _current_branch(repo))
    try:
        _run(["git", "push", remote, branch], cwd=repo, check=True)
    except subprocess.CalledProcessError as push_err:
        # Remote has likely advanced. Try to rebase onto the latest remote state
        # and push again. If anything in this recovery flow fails we still
        # consider the commit successful so the caller can continue.
        print(
            f"[git] initial push failed ({push_err.returncode}); attempting pull --rebase",
            file=sys.stderr,
        )
        try:
            _run(
                ["git", "pull", "--rebase", "--autostash", remote, branch],
                cwd=repo,
                check=True,
            )
        except subprocess.CalledProcessError as pull_err:
            print(
                f"[git] pull --rebase failed ({pull_err.returncode}); leaving commit local",
                file=sys.stderr,
            )
            return True

        try:
            _run(["git", "push", remote, branch], cwd=repo, check=True)
        except subprocess.CalledProcessError as retry_err:
            print(
                f"[git] retry push failed ({retry_err.returncode}); leaving commit local",
                file=sys.stderr,
            )
            return True

    return True


# ------------------------------
# Convenience: one-shot helper
# ------------------------------

def finalize_and_commit(*args, **kwargs) -> None:
    """Finalize a run's logs and (optionally) commit them to git.

    Backward compatible with:
      - finalize_and_commit(run_id, message=...) or ... commit_message=...
      - finalize_and_commit(message=...) with no run_id (we infer default)
    """
    run_id = None
    if len(args) >= 1 and isinstance(args[0], str):
        run_id = args[0]

    if run_id is None:
        run_id = kwargs.pop("run_id", None)
    message = kwargs.pop("message", None) or kwargs.pop("commit_message", None)

    # Ignore legacy extras silently
    kwargs.pop("forecast_rows_written", None)
    kwargs.pop("extra_paths", None)

    if not message:
        message = os.getenv("GIT_LOG_MESSAGE", "chore(logs): append forecasts & run logs")

    # Default run_id if not provided
    try:
        from .config import ist_stamp
        default_rid = ist_stamp()
    except Exception:
        import datetime as _dt
        default_rid = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rid = run_id or default_rid

    # Collect files to commit
    lp = get_log_paths()
    run_dir = lp.runs_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    paths_to_commit = []
    if lp.forecasts_csv.exists():
        paths_to_commit.append(lp.forecasts_csv)
    # human logs (md/txt depending on HUMAN_LOG_EXT)
    paths_to_commit.extend(run_dir.rglob(f"*.{lp.human_ext}"))

    if not paths_to_commit:
        return

    # Delegate to the robust helper that already handles CI/local policy,
    # git identity, branch detection, and push.
    commit_and_push_logs(paths_to_commit, commit_message=message)

# -----------------------------------------------------------------------------
# Compatibility shims for older/newer CLI codepaths
# -----------------------------------------------------------------------------

def ensure_unified_csv() -> None:
    """
    Older CLI expects this no-op initializer. We ensure the CSV directory exists
    by touching header if needed with an empty field set.
    """
    paths = get_log_paths()
    if not paths.forecasts_csv.exists():
        # create an empty CSV with a trivial header so append later just works
        _write_csv_header_if_needed(paths.forecasts_csv, ["run_id"])

def write_unified_row(row: Dict[str, object]) -> None:
    """
    Older/newer CLI calls this to append one wide row. We delegate to
    append_forecast_row(...) which handles schema evolution safely.
    """
    append_forecast_row(row)

# --- begin patch: backward-compatible write_human_markdown ---
def write_human_markdown(*args, **kwargs) -> str:
    """
    Write a per-question human-readable markdown file.

    Backward compatible with both:
      - write_human_markdown(run_id, question_id, content)
      - write_human_markdown(question_id=..., content=..., run_id=...)
    and any mixture of positional/keyword args without "multiple values" errors.

    Returns the absolute path written, as a string.
    """
    question_id = None
    content = None
    run_id = None

    # -------- Back-compat positional handling --------
    # Possible legacy forms:
    #   (run_id, question_id, content)
    #   (question_id, content)  -> implies default/current run_id
    if len(args) >= 1 and isinstance(args[0], str):
        if len(args) >= 2 and str(args[1]).isdigit():
            run_id = args[0]
            question_id = int(args[1])
            if len(args) >= 3 and isinstance(args[2], str):
                content = args[2]
    elif len(args) >= 1 and str(args[0]).isdigit():
        question_id = int(args[0])
        if len(args) >= 2 and isinstance(args[1], str):
            content = args[1]

    # -------- Keyword args (authoritative) --------
    if question_id is None:
        qi = kwargs.pop("question_id", None)
        if qi is not None:
            question_id = int(qi)
    if content is None:
        content = kwargs.pop("content", None)
    if run_id is None:
        run_id = kwargs.pop("run_id", None)

    if question_id is None or content is None:
        raise TypeError("write_human_markdown() requires question_id and content. (run_id is optional)")

    # Default run_id if not provided
    try:
        from .config import ist_stamp
        default_rid = ist_stamp()
    except Exception:
        import datetime as _dt
        default_rid = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rid = run_id or default_rid

    # Write under forecast_logs/runs/<run_id>/human/
    paths = get_log_paths()
    md_dir = (paths.runs_dir / rid / "human")
    md_dir.mkdir(parents=True, exist_ok=True)

    md_path = md_dir / f"Q{int(question_id)}.{paths.human_ext}"
    md_path.write_text(content, encoding="utf-8")
    return str(md_path)

