"""
BMI Forcings Engine standalone preprocessing mode wrapper script.

Provides ability to streamline ESMF Mesh production and operational data
workflows based on Forcing Engine configuration inputs.

example usage: python preprocessing_wrapper.py ./standard_ana_config.yml
"""

import argparse
import os
import time
from types import SimpleNamespace

import yaml

import esmf_creation


def execute(forcing_config_input: str):
    """
    Execute the preprocessing steps for NextGen Forcings Engine pipeline in standalone mode.

    Modules executed: ESMF Mesh Conversion.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    :return: None
    """
    # Read in forcing engine configuration file
    with open(forcing_config_input, 'r') as forcing_config_file:
        forcing_config = yaml.safe_load(forcing_config_file)

    # Wrap config dict into simplenamespace to match esmf creation ConfigOptions format
    esmf_cfg = SimpleNamespace(
        geopackage=forcing_config['Geopackage'],
        geogrid=forcing_config['GeogridIn'],
        parquet=forcing_config.get('Parquet', forcing_config.get('parquet', None))
    )
