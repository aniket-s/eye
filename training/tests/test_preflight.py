"""The dependency preflight.

Its whole purpose is the message it prints, so that is what gets asserted. A check that
merely returns the right boolean would still be free to emit something useless.
"""

from __future__ import annotations

import importlib.util

import pytest

from mudra_train import preflight


def test_reports_nothing_missing_in_a_working_environment() -> None:
    assert preflight.missing_dependencies() == []


def test_passes_silently_when_everything_is_installed() -> None:
    preflight.require_dependencies()


def test_names_what_is_missing_and_how_to_install_it(monkeypatch) -> None:
    """The failure this exists to replace was `ModuleNotFoundError: No module named
    'numpy'` raised from the middle of train.py — accurate and unactionable."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None if name == "torch" else real(name, *args, **kwargs),
    )

    with pytest.raises(SystemExit) as exit_info:
        preflight.require_dependencies()

    message = str(exit_info.value)
    assert "torch" in message
    assert 'pip install -e "training[dev]"' in message
    # The most useful fact of all: they probably do not need to do this.
    assert "ships with the repository" in message  # must not be split across lines


def test_recommends_a_virtual_environment(monkeypatch) -> None:
    """A plain `pip install` into a recent system Python fails with an even worse error."""
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None if name == "numpy" else real(name, *args, **kwargs),
    )
    with pytest.raises(SystemExit) as exit_info:
        preflight.require_dependencies()
    assert "venv" in str(exit_info.value)


def test_every_requirement_explains_itself() -> None:
    for name, purpose in preflight.REQUIRED.items():
        assert name and purpose
