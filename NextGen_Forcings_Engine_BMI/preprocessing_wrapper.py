"""
BMI Forcings Engine standalone preprocessing mode wrapper script.

Provides ability to streamline ESMF Mesh production and operational data
workflows based on Forcing Engine configuration inputs.

example usage: python preprocessing_wrapper.py ./standard_ana_config.yml
"""

from __future__ import annotations

import argparse
import pathlib
import time

import esmf_creation
from config_models import ConfigOptionsSubset


def execute(forcing_config_input: str | pathlib.Path) -> None:
    """
    Execute the preprocessing steps for NextGen Forcings Engine pipeline in standalone mode.

    Modules executed: ESMF Mesh Conversion.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    """
    esmf_cfg: ConfigOptionsSubset = ConfigOptionsSubset.from_yaml_file(forcing_config_input)

    start: float = time.perf_counter()
    esmf_creation.create_mesh(esmf_cfg)
    end: float = time.perf_counter()

    print(f"Time taken for ESMF mesh production: {end - start:.6f} seconds")


def get_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "forcing_config_input",
        type=str,
        help="Path to forcing engine configuration file for forecast run",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = get_options()
    execute(forcing_config_input=args.forcing_config_input)


if __name__ == "__main__":
    main()
