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
    esmf_cfg: ConfigOptionsSubset = ConfigOptionsSubset.from_yaml_file(forcing_config_input)
    extract_cfg: ForcingExtractionConfig = ForcingExtractionConfig.from_yaml_file(forcing_config_input)

    # Start timer for ESMF Mesh production
    start: float = time.perf_counter()
    esmf_creation.create_mesh(esmf_cfg)
    end: float = time.perf_counter()
    print(f"Time taken for ESMF mesh production: {end - start:.6f} seconds")

    # Start timer for forcing extraction
    start = time.perf_counter()
    forcing_extraction.retrieve_forcing(extract_cfg)
    end = time.perf_counter()
    print(f"Time taken for forcing file extraction: {end - start:.6f} seconds")


def get_options() -> argparse.Namespace:
    """Parse command line arguments for forcing configuration path."""
    parser = argparse.ArgumentParser(description="BMI Forcings Engine preprocessing driver")
    parser.add_argument(
        "forcing_config_input",
        type=pathlib.Path,
        help="Path to forcing engine configuration file for forecast run",
    )
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args: argparse.Namespace = get_options()
    execute(forcing_config_input=args.forcing_config_input)


if __name__ == "__main__":
    main()
