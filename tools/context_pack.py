#!/usr/bin/env python3
"""Generate lightweight repository context packs for GPT-5 coding sessions."""
from __future__ import annotations

import argparse
import ast
import collections
import datetime as _dt
import json
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_IGNORE_PARTS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "migrations",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "\.tox",
}

PY_EXTENSIONS = {".py"}
TS_EXTENSIONS = {".ts", ".tsx", ".js"}
TEXT_EXTENSIONS = {
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".md",
    ".txt",
    ".ini",
    ".cfg",
}

SCHEMA_DIR = pathlib.Path("schemas")

EXPORT_REGEX = re.compile(
    r"export\s+(?:default\s+)?(?:async\s+)?"
    r"(?:(?:function|class)\s+(?P<func>[A-Za-z0-9_]+)"
    r"|(?:const|let|var|type|interface|enum)\s+(?P<const>[A-Za-z0-9_]+))"
)


class ShellError(RuntimeError):
    """Raised when a shell command cannot be executed."""


def sh(*args: str, check: bool = True, capture_stderr: bool = True) -> str:
    """Run a shell command and return its text output."""

    stderr = subprocess.STDOUT if capture_stderr else None
    try:
        out = subprocess.check_output(args, text=True, stderr=stderr)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        if not check:
            return exc.output or ""
        raise ShellError(f"Command {' '.join(args)} failed: {exc.output}") from exc
    return out


def git_root() -> pathlib.Path:
    """Return the git repository root."""

    out = sh("git", "rev-parse", "--show-toplevel")
    return pathlib.Path(out.strip())


def resolve_default_base() -> str:
    """Determine the default base revision for comparisons."""

    try:
        sh("git", "rev-parse", "--verify", "origin/main")
        return "origin/main"
    except ShellError:
        pass

    try:
        tag = sh("git", "describe", "--tags", "--abbrev=0").strip()
        if tag:
            return tag
    except ShellError:
        pass
    return "HEAD"


def changed_since(base: str) -> List[Tuple[str, str]]:
    """Return (status, path) tuples for files changed since *base*."""

    try:
        out = sh("git", "diff", "--name-status", base, "--", check=True)
    except ShellError:
        return []
    changes: List[Tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            status, path = parts
        else:
            status, path = parts[0], ""
        changes.append((status.strip(), path.strip()))
    return changes


def churn_hotspots(limit: int = 20) -> List[Tuple[str, int]]:
    """Return the files with the highest change frequency."""

    try:
        out = sh("git", "log", "--name-only", "--pretty=format:")
    except ShellError:
        return []
    counter: collections.Counter[str] = collections.Counter()
    for line in out.splitlines():
        path = line.strip()
        if path:
            counter[path] += 1
    return counter.most_common(limit)


def include_path(path: str) -> bool:
    """Check whether a path should be included in context outputs."""

    if not path:
        return False
    if any(part in REPO_IGNORE_PARTS for part in path.split("/")):
        return False
    suffix = pathlib.Path(path).suffix
    if suffix in PY_EXTENSIONS or suffix in TS_EXTENSIONS:
        return True
    if suffix in TEXT_EXTENSIONS:
        return True
    return False


def list_repo_files() -> List[str]:
    """List tracked repository files with filtering."""

    try:
        out = sh("git", "ls-files")
    except ShellError:
        return []
    files: List[str] = []
    for line in out.splitlines():
        rel = line.strip()
        if include_path(rel):
            files.append(rel)
    files.sort()
    return files


def build_tree(paths: Sequence[str]) -> str:
    """Create a textual tree representation from paths."""

    tree: Dict[str, Dict] = {}
    for path in paths:
        parts = path.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("__files__", set()).add(parts[-1])

    lines: List[str] = []

    def recurse(prefix: str, node: Dict[str, Dict], depth: int = 0) -> None:
        directories = sorted(k for k in node.keys() if k != "__files__")
        files = sorted(node.get("__files__", []))
        indent = "  " * depth
        for directory in directories:
            lines.append(f"{indent}{directory}/")
            recurse(prefix + directory + "/", node[directory], depth + 1)
        for file in files:
            lines.append(f"{indent}{file}")

    recurse("", tree, 0)
    return "\n".join(lines)


def extract_py_symbols(path: pathlib.Path) -> List[Tuple[str, str, str]]:
    """Return (name, signature, doc) for top-level Python symbols."""

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    symbols: List[Tuple[str, str, str]] = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args: List[str] = []
            pos_args = [arg.arg for arg in node.args.posonlyargs]
            if pos_args:
                args.extend(pos_args)
                if node.args.args:
                    args.append("/")
            args.extend(arg.arg for arg in node.args.args)
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            kwonly = node.args.kwonlyargs
            if kwonly:
                if not node.args.vararg:
                    args.append("*")
                args.extend(arg.arg for arg in kwonly)
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            signature = f"def {node.name}({', '.join(args)})"
        elif isinstance(node, ast.ClassDef):
            signature = f"class {node.name}"
        else:
            continue

        docstring = ast.get_docstring(node) or ""
        doc_line = docstring.strip().splitlines()[0] if docstring else ""
        symbols.append((node.name, signature, doc_line))
    return symbols


def extract_ts_exports(path: pathlib.Path) -> List[Tuple[str, str]]:
    """Return exported symbol descriptions from TS/JS files."""

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    exports: List[Tuple[str, str]] = []
    for match in EXPORT_REGEX.finditer(source):
        name = match.group("func") or match.group("const") or "<anonymous>"
        line_no = source.count("\n", 0, match.start()) + 1
        line = source.splitlines()[line_no - 1].strip()
        exports.append((name, f"L{line_no}: {line}"))
    return exports


def extract_pyarrow_schemas(path: pathlib.Path) -> List[Tuple[str, str]]:
    """Return (name, source) pairs for PyArrow schema definitions."""

    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    schemas: List[Tuple[str, str]] = []
    for node in ast.walk(module):
        target_name: str | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                target_name = tgt.id
            elif isinstance(tgt, ast.Attribute):
                target_name = ast.unparse(tgt) if hasattr(ast, "unparse") else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
        else:
            continue

        call = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "schema" and isinstance(func.value, ast.Name) and func.value.id in {"pa", "pyarrow"}:
            snippet = ast.get_source_segment(source, node) or target_name or "schema"
            preview = textwrap.shorten(" ".join(snippet.split()), width=200, placeholder="...")
            label = target_name or "schema"
            schemas.append((label, preview))
    return schemas


def find_schema_json(root: pathlib.Path) -> List[pathlib.Path]:
    if not (root / SCHEMA_DIR).exists():
        return []
    json_paths: List[pathlib.Path] = []
    for path in (root / SCHEMA_DIR).rglob("*.json"):
        if include_path(str(path.relative_to(root))):
            json_paths.append(path)
    json_paths.sort()
    return json_paths


def format_json_preview(path: pathlib.Path, max_length: int = 400) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    snippet = data.strip()
    try:
        parsed = json.loads(snippet)
        snippet = json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        pass
    if len(snippet) > max_length:
        snippet = snippet[: max_length - 3] + "..."
    return snippet


def ensure_outdir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_file(path: pathlib.Path, content: str) -> None:
    ensure_outdir(path.parent)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def build_public_api(paths: Iterable[str], root: pathlib.Path) -> str:
    py_lines: List[str] = []
    ts_lines: List[str] = []
    for rel in paths:
        file_path = root / rel
        suffix = file_path.suffix
        if suffix in PY_EXTENSIONS:
            for _name, signature, doc in extract_py_symbols(file_path):
                detail = f"`{signature}`"
                if doc:
                    detail += f" — {doc}"
                py_lines.append(f"- `{rel}`: {detail}")
        elif suffix in TS_EXTENSIONS:
            exports = extract_ts_exports(file_path)
            for name, line in exports:
                ts_lines.append(f"- `{rel}`: `{name}` — {line}")

    lines: List[str] = ["# Public APIs & Contracts"]
    lines.append("\n## Python")
    if py_lines:
        lines.extend(py_lines)
    else:
        lines.append("(none detected)")
    lines.append("\n## TypeScript / JavaScript")
    if ts_lines:
        lines.extend(ts_lines)
    else:
        lines.append("(none detected)")
    return "\n".join(lines)


def build_changeset(changes: Sequence[Tuple[str, str]], base: str) -> str:
    lines = [f"# Changes since {base}"]
    if not changes:
        lines.append("(no tracked changes)")
        return "\n".join(lines)
    for status, path in changes:
        if path:
            lines.append(f"- {status}\t{path}")
        else:
            lines.append(f"- {status}")
    return "\n".join(lines)


def build_hotspots(entries: Sequence[Tuple[str, int]]) -> str:
    lines = ["# Hotspots (top change frequency)"]
    if not entries:
        lines.append("(no history available)")
        return "\n".join(lines)
    for path, count in entries:
        lines.append(f"- {path} ({count})")
    return "\n".join(lines)


def build_codemap(paths: Sequence[str], hotspots: Sequence[Tuple[str, int]], changes: Sequence[Tuple[str, str]]) -> str:
    lines = [
        "# CODEMAP",
        "## Purpose",
        "- <Describe the overall goal of this repository>",
        "## Entrypoints",
        "- Document CLI or service entrypoints here",
        "## Modules (high level)",
        "- Summarize key modules here",
        "## Config & Secrets",
        "- List relevant config files and secret handling",
        "## Data & Schemas",
        "- Summarize data locations and schema docs",
        "## Tests",
        "- Note how to run test suites",
        "\n---\n### Auto-generated appendix",
        "#### Tree (filtered)",
        "```",
    ]
    if paths:
        tree_preview = build_repo_tree(paths).splitlines()
        lines.extend(tree_preview[:1000])
    lines.append("```")
    lines.append("#### Hotspots")
    if hotspots:
        lines.extend(f"- {path} ({count})" for path, count in hotspots)
    else:
        lines.append("- (none)")
    lines.append("#### Changed since base")
    if changes:
        lines.extend(f"- {status}\t{path}" for status, path in changes if path)
    else:
        lines.append("- (no tracked changes)")
    return "\n".join(lines)


def build_schemas(root: pathlib.Path, repo_files: Sequence[str]) -> str:
    lines = ["# Schemas"]
    json_paths = find_schema_json(root)
    if json_paths:
        lines.append("\n## JSON Schemas")
        for path in json_paths:
            rel = path.relative_to(root)
            preview = format_json_preview(path)
            lines.append(f"### `{rel}`")
            lines.append("```json")
            lines.append(preview)
            lines.append("```")
    pyarrow_entries: List[Tuple[str, str, str]] = []
    for rel in repo_files:
        file_path = root / rel
        if file_path.suffix in PY_EXTENSIONS:
            for name, snippet in extract_pyarrow_schemas(file_path):
                pyarrow_entries.append((rel, name, snippet))
    if pyarrow_entries:
        lines.append("\n## PyArrow Schemas")
        for rel, name, snippet in pyarrow_entries:
            lines.append(f"- `{rel}` defines `{name}`: {snippet}")
    if len(lines) == 1:
        lines.append("(none detected)")
    return "\n".join(lines)


def build_repo_tree(paths: Sequence[str]) -> str:
    if not paths:
        return "(repository tree unavailable)"
    return build_tree(paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a GPT context pack for the repository")
    parser.add_argument("--base", help="Base revision for CHANGESET", default=None)
    parser.add_argument("--outdir", help="Output directory (default: context/pack-<timestamp>)", default=None)
    args = parser.parse_args(argv)

    root = git_root()
    base = args.base or resolve_default_base()

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = pathlib.Path(args.outdir) if args.outdir else root / "context" / f"pack-{timestamp}"
    ensure_outdir(outdir)

    repo_files = list_repo_files()
    hotspots = churn_hotspots()
    changes = changed_since(base)

    # Write REPO_TREE.txt
    write_file(outdir / "REPO_TREE.txt", build_repo_tree(repo_files))

    # Write HOTSPOTS.md
    write_file(outdir / "HOTSPOTS.md", build_hotspots(hotspots))

    # Write PUBLIC_APIS.md
    write_file(outdir / "PUBLIC_APIS.md", build_public_api(repo_files, root))

    # Write CHANGESET.md
    write_file(outdir / "CHANGESET.md", build_changeset(changes, base))

    # Write SCHEMAS.md
    write_file(outdir / "SCHEMAS.md", build_schemas(root, repo_files))

    # Write CODEMAP.md
    write_file(outdir / "CODEMAP.md", build_codemap(repo_files, hotspots, changes))

    print(f"Context pack created at {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
