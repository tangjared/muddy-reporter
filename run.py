"""Tiny CLI shim: `python run.py --ticker LKNCY [...]` works without `pip install -e .`."""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap src layout so users don't need an editable install.
_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from muddy_reporter.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
