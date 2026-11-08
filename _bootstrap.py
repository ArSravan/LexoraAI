"""Make `import LexoraAI` work from the project folder, whatever the folder layout.

LexoraAI is the Python package. Usually the project folder contains a LexoraAI/ folder and nothing is needed.
In the other layout the project folder IS the package: its modules (config.py, api/, services/ ...) sit next
to app.py and the folder has an __init__.py. Python can then import the package by its folder name only if the
folder's PARENT is on sys.path. Every entry point (main.py, app.py, scripts/, evaluation/) imports this module
first, so nobody has to set PYTHONPATH by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

PACKAGE = "LexoraAI"
HERE = Path(__file__).resolve().parent


def ensure_importable(root: Path = HERE) -> bool:
    """Put the parent of the project folder on sys.path when the folder itself is the package.
    Returns True if sys.path was changed. Raises ImportError if the folder is not named LexoraAI."""
    if not (root / "__init__.py").is_file():
        return False                                    # usual layout: the project folder has a LexoraAI/ folder
    if root.name != PACKAGE:
        raise ImportError(f"This project folder is the Python package '{PACKAGE}' but it is named '{root.name}'. "
                          f"Rename the folder to '{PACKAGE}'.")
    parent = str(root.parent)
    if parent in sys.path:
        return False
    sys.path.insert(0, parent)
    return True


ensure_importable()
