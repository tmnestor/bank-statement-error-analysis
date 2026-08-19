"""The declared-unproducible contract, shared by the runners and by scoring.

`score` refuses a system with missing predictions, because a silent gap is
indistinguishable from a run that died half-way. A *declared* gap is different:
it is a finding, and it is scored as a total failure rather than excused, so
every system stays averaged over the same transcripts.

The file lives beside the predictions as `_unproduced.json`. `score` globs
`*.md`, so it is never mistaken for a prediction.

Standard library only: a runner must be able to write this in an environment
that holds no corpus dependencies, and `score` must be able to read it in one
that holds no parser.
"""

import json
from pathlib import Path

UNPRODUCED_FILE = "_unproduced.json"


def declare_unproduced(out_dir: Path, stems: list[str], reason: str) -> Path:
    """Declare pages a system cannot produce, so scoring can represent them.

    Declaring is deliberate and operator-driven. Auto-declaring on any failure
    would let a transient error become a permanent "unproducible", which is
    exactly the kind of quiet gap this contract exists to prevent.

    Args:
        out_dir: The system's prediction directory.
        stems: The stems this system cannot produce.
        reason: Why, in one line, for whoever reads the report.

    Returns:
        The path written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / UNPRODUCED_FILE
    path.write_text(
        json.dumps({"reason": reason, "pages": sorted(stems)}, indent=2),
        encoding="utf-8",
    )
    return path


def read_unproduced(out_dir: Path) -> set[str]:
    """Read the stems a system has declared it cannot produce.

    Args:
        out_dir: The system's prediction directory.

    Returns:
        The declared stems, empty when nothing is declared.
    """
    path = out_dir / UNPRODUCED_FILE
    if not path.exists():
        return set()
    declared = json.loads(path.read_text(encoding="utf-8"))
    return set(declared.get("pages", []))
