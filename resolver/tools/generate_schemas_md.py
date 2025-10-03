#!/usr/bin/env python3
"""Generate SCHEMAS.md from resolver/tools/schema.yml.

This script normalises the schema definition into a set of entities and
renders a Markdown document containing a table of contents and a table for
each entity.  It is intentionally defensive so it can work with both the
current schema shape as well as legacy variants that only describe a single
facts table with ``required``/``optional`` keys.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - handled in CI environments only
    print("Please install pyyaml to run the schema generator.", file=sys.stderr)
    sys.exit(2)


GLOBAL_ENTITY_KEYS = {"entities", "required", "optional", "enums"}
REFERENCE_FIELDS = ("fk", "ref", "reference", "references")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input", required=True, help="Path to schema.yml")
    parser.add_argument("--out", dest="output", required=True, help="Path to SCHEMAS.md")
    parser.add_argument("--title", default="Resolver Schemas", help="Title for the markdown output")
    parser.add_argument(
        "--fail-on-missing-desc",
        action="store_true",
        help="Fail if any column is missing a description.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort entities and columns alphabetically (required columns first).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}
        return data


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def looks_like_entity(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    return any(k in data for k in ("columns", "required", "optional"))


def normalise_entities(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    entity_sections = data.get("entities") if isinstance(data.get("entities"), dict) else None
    if entity_sections:
        for name, payload in entity_sections.items():
            entity = build_entity(name, payload or {}, data)
            entities.append(entity)
            seen_names.add(entity["name"])

    # Legacy style where sections live at the top level
    for key, value in data.items():
        if key in GLOBAL_ENTITY_KEYS:
            continue
        if key in seen_names:
            continue
        if isinstance(value, dict) and looks_like_entity(value):
            entities.append(build_entity(key, value, data))
            seen_names.add(str(key))

    # Bare schema (required/optional at top level)
    if not entities and looks_like_entity(data):
        default_name = str(data.get("name") or data.get("title") or "facts")
        entities.append(build_entity(default_name, data, data))

    return entities


def build_entity(name: str, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    description = str(payload.get("description", "")).strip()
    keys = ensure_list(payload.get("keys", []))
    columns = normalise_columns(payload, context)
    return {
        "name": str(name),
        "description": description,
        "keys": [str(k) for k in keys if str(k).strip()],
        "columns": columns,
    }


def normalise_columns(payload: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    columns_map: Dict[str, Dict[str, Any]] = {}

    def get_or_create(col_name: str) -> Dict[str, Any]:
        col = columns_map.get(col_name)
        if not col:
            col = {
                "name": col_name,
                "type": "",
                "required": False,
                "enum": None,
                "format": None,
                "description": "",
                "notes": [],
            }
            columns_map[col_name] = col
        return col

    columns_data = payload.get("columns")
    if isinstance(columns_data, dict):
        iterator = columns_data.items()
    elif isinstance(columns_data, list):
        iterator = []
        for entry in columns_data:
            if isinstance(entry, dict):
                name = entry.get("name") or ""
                iterator.append((name, entry))
            else:
                iterator.append((str(entry), {"name": entry}))
    else:
        iterator = []

    for name, info in iterator:
        if not name:
            continue
        col = get_or_create(str(name))
        merge_column(col, info or {})

    for required_name in ensure_list(payload.get("required", [])):
        if not required_name:
            continue
        col = get_or_create(str(required_name))
        col["required"] = True

    optional = payload.get("optional")
    if isinstance(optional, dict):
        for opt_name, opt_info in optional.items():
            col = get_or_create(str(opt_name))
            if isinstance(opt_info, dict):
                merge_column(col, opt_info)
            if "required" in col:
                col["required"] = bool(col.get("required"))
    elif isinstance(optional, list):
        for opt_entry in optional:
            if isinstance(opt_entry, dict):
                name = opt_entry.get("name")
                if not name:
                    continue
                col = get_or_create(str(name))
                merge_column(col, opt_entry)
            else:
                name = str(opt_entry)
                col = get_or_create(name)
                col["required"] = bool(col.get("required"))

    enums = context.get("enums") if isinstance(context.get("enums"), dict) else {}
    for col_name, values in enums.items() if isinstance(enums, dict) else []:
        if not col_name:
            continue
        col = get_or_create(str(col_name))
        if not col.get("enum"):
            col["enum"] = ensure_list(values)
            if not col.get("type"):
                col["type"] = "enum"

    # Ensure boolean required
    for column in columns_map.values():
        column["required"] = bool(column.get("required", False))
        if column.get("enum") and not column.get("type"):
            column["type"] = "enum"

    return list(columns_map.values())


def merge_column(target: Dict[str, Any], info: Dict[str, Any]) -> None:
    for key, value in info.items():
        if key == "name":
            continue
        if key in {"required"}:
            target[key] = bool(value)
        elif key == "enum" and value is not None:
            target[key] = ensure_list(value)
        elif key == "notes" and value:
            target.setdefault("notes", [])
            if isinstance(value, list):
                target["notes"].extend(str(v) for v in value)
            else:
                target["notes"].append(str(value))
        elif key in {"fk", "ref", "reference", "references"}:
            target.setdefault("notes", [])
            target["notes"].append(format_reference(value))
        else:
            target[key] = value if value is not None else target.get(key)


def format_reference(value: Any) -> str:
    if isinstance(value, dict):
        table = value.get("table") or value.get("entity") or value.get("name")
        column = value.get("column") or value.get("field")
        if table and column:
            return f"{table}.{column}"
        if table:
            return str(table)
    return str(value)


def slugify(text: str) -> str:
    import re

    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug


def escape_markdown(text: str) -> str:
    text = text.replace("|", "\\|")
    text = text.replace("`", "\\`")
    return text.replace("\n", "<br>")


def escape_inline_code(text: str) -> str:
    return text.replace("`", "\\`")


def format_enum_or_format(column: Dict[str, Any]) -> str:
    enum_values = column.get("enum")
    fmt = column.get("format")
    parts: List[str] = []
    if enum_values:
        if isinstance(enum_values, dict):
            enum_values = list(enum_values.values())
        if not isinstance(enum_values, list):
            enum_values = [enum_values]
        values = [str(v) for v in enum_values if str(v)]
        parts.append(chunk_join(values))
    if fmt:
        parts.append(str(fmt))
    return "<br>".join(parts)


def chunk_join(values: List[str], chunk_size: int = 5) -> str:
    if not values:
        return ""
    if len(values) <= chunk_size:
        return ", ".join(values)
    lines = []
    for i in range(0, len(values), chunk_size):
        lines.append(", ".join(values[i : i + chunk_size]))
    return "<br>".join(lines)


def render_markdown(
    entities: List[Dict[str, Any]],
    title: str,
    sort_entities: bool,
    fail_on_missing_desc: bool,
) -> Tuple[str, List[str]]:
    missing_desc: List[str] = []

    if sort_entities:
        entities = sorted(entities, key=lambda e: e["name"].lower())

    now = dt.datetime.utcnow().strftime("%Y-%m-%dZ")
    lines: List[str] = [
        f"<!--- DO NOT EDIT: generated by resolver/tools/generate_schemas_md.py on {now} -->",
        "",
        f"# {title}",
        "",
    ]

    if not entities:
        lines.append("_No schemas defined in resolver/tools/schema.yml._")
        lines.append("")
        return "\n".join(lines).rstrip() + "\n", missing_desc

    # Table of contents
    lines.append("## Table of contents")
    lines.append("")
    for entity in entities:
        anchor = slugify(entity["name"])
        lines.append(f"- [{entity['name']}](#{anchor})")
    lines.append("")

    for entity in entities:
        name = entity["name"]
        anchor = slugify(name)
        if sort_entities:
            entity_columns = sorted(
                entity["columns"],
                key=lambda c: (0 if c.get("required") else 1, c.get("name", "").lower()),
            )
        else:
            entity_columns = entity["columns"]

        lines.append(f"## {name}")
        lines.append("")
        if entity.get("description"):
            lines.append(entity["description"].strip())
            lines.append("")
        if entity.get("keys"):
            keys = ", ".join(f"`{k}`" for k in entity["keys"])
            lines.append(f"**Keys:** {keys}")
            lines.append("")

        lines.append("| Name | Type | Required | Enum/Format | Description |")
        lines.append("| --- | --- | --- | --- | --- |")

        notes: List[Tuple[str, str]] = []
        if not entity_columns:
            lines.append("| _(none)_ |  |  |  |  |")
        else:
            for column in entity_columns:
                name_cell = escape_markdown(str(column.get("name", "")))
                type_cell = escape_markdown(str(column.get("type", "")))
                required_cell = "yes" if column.get("required") else "no"
                enum_cell = escape_markdown(format_enum_or_format(column))
                description = column.get("description", "")
                if not description and fail_on_missing_desc:
                    missing_desc.append(f"{entity['name']}.{column.get('name', '')}")
                desc_cell = escape_markdown(str(description or ""))
                lines.append(
                    f"| {name_cell} | {type_cell} | {required_cell} | {enum_cell} | {desc_cell} |"
                )

                for ref_field in REFERENCE_FIELDS:
                    value = column.get(ref_field)
                    if value:
                        notes.append((str(column.get("name", "")), format_reference(value)))
                for note in column.get("notes", []) or []:
                    notes.append((str(column.get("name", "")), str(note)))

        if notes:
            lines.append("")
            for col_name, message in notes:
                code_name = escape_inline_code(col_name)
                lines.append(
                    f"- `{code_name}` → {escape_markdown(message)}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n", missing_desc


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    schema_path = Path(args.input)
    output_path = Path(args.output)

    data = load_schema(schema_path)
    entities = normalise_entities(data)
    markdown, missing = render_markdown(
        entities,
        title=args.title,
        sort_entities=bool(args.sort),
        fail_on_missing_desc=args.fail_on_missing_desc,
    )

    if missing and args.fail_on_missing_desc:
        missing_str = ", ".join(missing)
        print(
            f"Missing descriptions for: {missing_str}",
            file=sys.stderr,
        )
        return 1

    output_path.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
