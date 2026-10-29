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

    # Sanitize parquet path: enforce None if empty string or file not found
    parquet_val = forcing_config.get('parquet', forcing_config.get('Parquet', None))
    if parquet_val is not None:
        parquet_val = str(parquet_val).strip()
        if not parquet_val or not os.path.isfile(parquet_val):
            print(
                f"NOTICE: Parquet file '{parquet_val}' is empty or does not exist. "
                "Setting to None; bias correction and downscaling methods will not be applied"
                "in the NextGen Forcings Engine BMI workflow unless parquet file is prescribed."
            )
            parquet_val = None
            
    # Wrap config dict into simplenamespace to match esmf creation ConfigOptions format
    esmf_cfg = SimpleNamespace(
        geopackage=forcing_config['Geopackage'],
        geogrid=forcing_config['GeogridIn'],
        parquet=parquet_val)
    )

    # Start timer for ESMF Mesh production script
    start = time.perf_counter()

    # Create mesh file
    esmf_creation.create_mesh(esmf_cfg)

    # Stop timer for ESMF Mesh production script
    end = time.perf_counter()

    # Report the time it took for ESMF Mesh production script
    print(f"Time taken for ESMF mesh production: {end - start:.6f} seconds")


def get_options():
    """
    Function to accept and parse arguments.

    :return: An argparse.Namespace object containing the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'forcing_config_input',
        type=str,
        help='Path to forcing engine configuration file for forecast run'
    )
    return parser.parse_args()


def main():
    """
    Main function to handle command-line execution.
    """
    args = get_options()
    execute(forcing_config_input=args.forcing_config_input)


if __name__ == '__main__':
    main()
