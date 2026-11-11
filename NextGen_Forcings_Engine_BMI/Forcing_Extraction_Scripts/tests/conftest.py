import sys
from pathlib import Path

import pytest

BMI_DIR = Path(__file__).resolve().parents[2]
if str(BMI_DIR) not in sys.path:
    sys.path.insert(0, str(BMI_DIR))


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="Run tests that contact upstream data servers.")


def pytest_configure(config):
    config.addinivalue_line("markers", "live: contacts upstream data servers; enabled with --live")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="contacts upstream servers; run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
