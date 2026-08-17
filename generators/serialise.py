"""Events plus policy to Markdown.

A pure function of the event stream and `config/serialisation.yml` — it imports
no PIL and renders nothing. That split is deliberate (design §6): the convention
is the risky, iterate-on-it part of this design, so it can change and every
transcript re-emit in seconds without re-rendering an image.

It takes plain dicts rather than `Event` objects so it can run straight off a
stored `events.jsonl` without importing the recorder.

**It never normalises.** No whitespace collapsing, no case folding, no Unicode
folding, no dash or quote substitution. Design §5 puts all of that in the
scoring tool so that scoring policy can change without regenerating the corpus;
baking any of it in here would freeze a policy into the data.
"""

from pathlib import Path

import yaml

REQUIRED_POLICY_KEYS: tuple[str, ...] = (
    "title_style",
    "pair_separator",
    "pair_strip_trailing_colon",
    "table_style",
    "empty_cell_token",
    "cell_sub_line_join",
    "cell_newline_join",
    "split_order",
    "block_separator",
    "emphasis",
)

# Enum-valued keys, each mapped to every value this serialiser implements. A
# value outside these is a configuration error, not a silently-ignored setting.
_ALLOWED: dict[str, tuple[str, ...]] = {
    "title_style": ("atx_h1",),
    "table_style": ("pipe_with_header_rule",),
    "split_order": ("column_major",),
    "emphasis": ("none",),
}

_EXAMPLES: dict[str, str] = {
    "title_style": "atx_h1",
    "pair_separator": '": "',
    "pair_strip_trailing_colon": "true",
    "table_style": "pipe_with_header_rule",
    "empty_cell_token": '""',
    "cell_sub_line_join": '" "',
    "cell_newline_join": '" "',
    "split_order": "column_major",
    "block_separator": '"\\n\\n"',
    "emphasis": "none",
}

# Structural markers: they shape the stream but put nothing in the transcript.
_STRUCTURE = frozenset(
    {"panel_open", "panel_close", "split_open", "split_close", "column_open", "column_close"}
)


class SerialisationError(RuntimeError):
    """Raised when the serialisation policy or the event stream is unusable."""


def _err(what: str, *, path: Path | str, key: str, expected: str, recover: str) -> SerialisationError:
    """Build a four-element fail-fast diagnostic."""
    return SerialisationError(
        "Invalid serialisation policy.\n"
        f"  What:     {what}\n"
        f"  Where:    {path} -> {key}\n"
        f"  Expected: {expected}\n"
        f"  Recover:  {recover}"
    )


def load_serialisation_policy(path: Path) -> dict:
    """Load and validate the serialisation convention.

    Every key is required. Omitting one is an error, never a silent default,
    including the keys whose value is a no-op (design §4.4) — so that reading
    the file alone answers what a transcript looks like.

    Args:
        path: Path to `serialisation.yml`.

    Returns:
        The validated policy mapping.

    Raises:
        SerialisationError: The file is missing, unparseable, missing a required
            key, or gives an enum key a value this serialiser does not implement.
    """
    resolved = path.resolve()
    if not path.exists():
        raise _err(
            f"{path} does not exist.",
            path=resolved,
            key="(whole file)",
            expected="a YAML mapping declaring every key of "
            f"{list(REQUIRED_POLICY_KEYS)}, e.g.\n              title_style: atx_h1",
            recover="create config/serialisation.yml (see the copy shipped with any exported corpus).",
        )

    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise _err(
            f"the file is not valid YAML: {err}",
            path=resolved,
            key="(whole file)",
            expected="parseable YAML, e.g.\n              emphasis: none",
            recover="fix the syntax error at the line named above.",
        ) from err

    if not isinstance(policy, dict):
        raise _err(
            f"expected a mapping, got {type(policy).__name__}.",
            path=resolved,
            key="(document root)",
            expected="a top-level mapping of policy keys, e.g.\n              split_order: column_major",
            recover="wrap the settings in a top-level mapping.",
        )

    for key in REQUIRED_POLICY_KEYS:
        if key not in policy:
            raise _err(
                f"'{key}' is not declared.",
                path=resolved,
                key=key,
                expected=f"every key of {list(REQUIRED_POLICY_KEYS)} present — none has a "
                f"Python default, including no-op values, e.g.\n              {key}: "
                f"{_EXAMPLES[key]}",
                recover=f"add '{key}:' to {path}.",
            )

    for key, allowed in _ALLOWED.items():
        if policy[key] not in allowed:
            raise _err(
                f"'{key}' is {policy[key]!r}, which this serialiser does not implement.",
                path=resolved,
                key=key,
                expected=f"one of {list(allowed)}, e.g.\n              {key}: {allowed[0]}",
                recover=f"set '{key}:' to a supported value, or implement it in generators/serialise.py.",
            )

    return policy


def _join_cell(text: str, policy: dict) -> str:
    """Fold an authored line break inside a cell onto one line.

    A pipe table cell is single-line, so a header authored as
    "Date of\\nTransaction" must be joined. Which joiner is policy, not code.
    """
    return policy["cell_newline_join"].join(part for part in text.split("\n"))


def _render_table(columns: list[str], rows: list[list[str]], policy: dict) -> str:
    """Render captured rows as a pipe table with a header separator row.

    Args:
        columns: The table's column keys, in order.
        rows: Cell text per row, already padded to the column count.
        policy: The validated serialisation policy.

    Returns:
        The pipe table as one block.
    """
    width = len(columns)
    lines = []
    for index, row in enumerate(rows):
        padded = list(row) + [policy["empty_cell_token"]] * (width - len(row))
        lines.append("| " + " | ".join(padded[:width]) + " |")
        if index == 0:
            lines.append("| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def serialise(events: list[dict], policy: dict) -> str:
    """Turn one document's event stream into its Markdown transcript.

    Args:
        events: The document's events, in walk order, as plain dicts.
        policy: The validated policy from `load_serialisation_policy`.

    Returns:
        The transcript. Never normalised — exactly what was drawn, arranged
        under the policy's conventions.

    Raises:
        SerialisationError: An event kind has no serialisation rule. A new kind
            must fail loudly rather than vanish from every transcript.
    """
    blocks: list[str] = []

    columns: list[str] = []
    table_rows: list[list[str]] = []
    row: list[str] = []
    row_keys: list[str] = []
    in_table = False

    for event in events:
        kind = event["kind"]
        text = event["text"]
        meta = event.get("meta", {})

        if kind in _STRUCTURE:
            # Structure only. Column order is already `column_major` in the
            # stream because `draw_split` walks columns in DSL order, so there
            # is nothing to reorder here — see the policy's `split_order`.
            continue

        if kind == "title":
            blocks.append(f"# {text}")
        elif kind == "line":
            blocks.append(str(text))
        elif kind == "pair":
            label = str(meta["label"])
            if policy["pair_strip_trailing_colon"]:
                label = label.rstrip().removesuffix(":").rstrip()
            blocks.append(f"{label}{policy['pair_separator']}{meta['value']}")
        elif kind == "table_open":
            in_table = True
            columns = [str(key) for key in meta["columns"]]
            table_rows = []
        elif kind == "row_open":
            row, row_keys = [], []
        elif kind == "cell":
            row.append(_join_cell(str(text or policy["empty_cell_token"]), policy))
            row_keys.append(str(meta.get("column_key", "")))
        elif kind == "cell_sub_line":
            # Folded into the cell it belongs to, found by column key so a
            # sub-line under column 2 cannot land on column 0.
            key = str(meta.get("column_key", ""))
            if key in row_keys:
                position = row_keys.index(key)
                row[position] = f"{row[position]}{policy['cell_sub_line_join']}{text}"
        elif kind == "row_close":
            table_rows.append(row)
            row, row_keys = [], []
        elif kind == "table_close":
            if table_rows:
                blocks.append(_render_table(columns, table_rows, policy))
            in_table = False
            columns, table_rows = [], []
        else:
            raise SerialisationError(
                "Unknown event kind.\n"
                f"  What:     no serialisation rule for event kind '{kind}' "
                f"(seq {event.get('seq')}).\n"
                "  Where:    generators/serialise.py -> serialise()\n"
                "  Expected: a branch for every kind a primitive emits, e.g. "
                '`elif kind == "line": blocks.append(text)`.\n'
                "  Recover:  add a rule for this kind, and a matching row to the "
                "event table in the design doc's §4.3."
            )

    if in_table:
        raise SerialisationError(
            "Unbalanced event stream.\n"
            "  What:     a table_open was never closed by a table_close.\n"
            "  Where:    the document's captured event stream\n"
            "  Expected: every table_open matched by a table_close, which "
            "`draw_table` emits on every path.\n"
            "  Recover:  regenerate this document; if it recurs, a table "
            "primitive is returning early without closing its events."
        )

    return policy["block_separator"].join(blocks)
