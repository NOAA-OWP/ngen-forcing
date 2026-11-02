"""
BMI Forcings Engine standalone preprocessing mode wrapper script.

Provides ability to streamline ESMF Mesh production and forcing extraction
of operational data based on Forcing Engine configuration inputs.

example usage: python preprocessing_wrapper.py ./standard_ana_config.yml
"""

from __future__ import annotations

import argparse
import pathlib
import time

from config_models import ConfigOptionsSubset, ForcingExtractionConfig
import esmf_creation
import forcing_extraction


def execute(forcing_config_input: str | pathlib.Path) -> None:
    """
    Execute preprocessing steps for NextGen Forcings Engine in standalone mode.

    Modules executed: ESMF Mesh Conversion and Forcing Extraction.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    """
