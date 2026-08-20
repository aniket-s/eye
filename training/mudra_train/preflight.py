"""Check the training dependencies are installed, and say what to do if not.

Training needs PyTorch and friends, which the web app does not. Someone who has only run
``npm install`` therefore meets a bare ``ModuleNotFoundError: No module named 'numpy'``
pointing at a line deep inside `train.py` — technically accurate and useless.

This module is imported *before* any third-party import in the training entry points, so
the first thing that happens is a message naming the missing package and the command that
installs it.

Deliberately dependency-free: it cannot import the thing it exists to report on.
"""

from __future__ import annotations

import importlib.util
import sys

#: Import name → what to say it is for.
REQUIRED = {
    "numpy": "arrays",
    "torch": "training",
    "onnx": "exporting the model pack",
    "onnxruntime": "verifying the exported model",
    "sklearn": "evaluation metrics",
}

INSTRUCTIONS = """
Training dependencies are not installed: {missing}.

From the repository root:

    python3 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
    pip install -e "training[dev]"

A virtual environment is worth the extra line — recent macOS and Linux Pythons refuse a
plain `pip install` into the system interpreter, and the error they give for that is even
less helpful than this one.

You probably do not need any of this: a trained model pack ships with the repository,
and `npm run dev` uses it as-is. Training is only needed to train a *new* model.
"""


def missing_dependencies() -> list[str]:
    """Which required packages are not importable."""
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def require_dependencies() -> None:
    """Exit with instructions if anything needed for training is absent.

    :func:`sys.exit` rather than an exception: this is a command-line entry point, and a
    traceback here would bury the one useful line in noise.
    """
    missing = missing_dependencies()
    if not missing:
        return
    sys.exit(INSTRUCTIONS.format(missing=", ".join(missing)))
