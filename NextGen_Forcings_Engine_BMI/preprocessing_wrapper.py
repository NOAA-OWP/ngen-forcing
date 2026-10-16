"""
BMI Forcings Engine standalone preproccesing mode wrapper script.

Provides ability to streamline ESMF Mesh production and forcing extraction of operational data based on Forcing Engine configuration inputs

example usage: python preprocessing_wrapper.py ./standard_ana_config.yml
"""

import argparse
import os
import time
import tempfile
import subprocess
from datetime import datetime, timedelta

import yaml
from types import SimpleNamespace

import forcing_extraction
import esmf_creation


def execute(forcing_config_input: str):
    """
    Execute the preprocessing steps for NextGen Forcings Engine pipeline in standalone mode.

    Modules executed: ESMF Mesh Conversion and Forcing Extraction.

    This method accepts expects all the output pathways to already be hard coded into
    the forcing configuration input file, which is te requriment for this driver script.
    It handles mesh conversion and forcing extraction using the specified parameters.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    :return: None
    """

    # Read in forcing engine configuration file
    with open(forcing_config_input, 'r') as forcing_config_file:
        forcing_config = yaml.safe_load(forcing_config_file)

    # Wrap config dict into simplenamespace to match esmf creation ConfigOptions format
    esmf_cfg = SimpleNamespace(geopackage=forcing_config['Geopackage'],
                               geogrid=forcing_config['GeogridIn'],
                               parquet=forcing_config.get('parquet',None))

    # Look for optional arguments with user controls on file
    # downloading mechanisms or just wanting to check data availability
    try: 
        max_download_attempts = forcing_config['max_download_attempts']
    except: 
        max_download_attempts = 10
    try:
        download_attempt_interval = forcing_config['download_attempt_interval']
    except:
        download_attempt_interval = 30
    try:
        check_file_availability = forcing_config['check_file_availability']
    except:
        check_file_availability = 0
        
    # Wrap config dict into simplenamespace to match forcing extraction ConfigOptions format
    extract_cfg = SimpleNamespace(b_date_proc=datetime.strptime(forcing_config['RefcstBDateProc'], "%Y%m%d%H%M"),
                                  input_forcings=forcing_config['InputForcings'],
                                  supp_precip_forcings=forcing_config['SuppPcp'],
                                  input_force_dirs=forcing_config['InputForcingDirectories'],
                                  supp_precip_dirs=forcing_config['SuppPcpDirectories'],
                                  fcst_input_horizons=forcing_config['ForecastInputHorizons'],
                                  cfsv2EnsMember=forcing_config['cfsEnsNumber'],
                                  ana_flag=forcing_config['AnAFlag'],
                                  look_back=forcing_config['LookBack'],
                                  max_download_attempts=max_download_attempts,
                                  download_attempt_interval=download_attempt_interval,
                                  check_file_availability=check_file_availability)

    # Start timer for ESMF Mesh production script
    start = time.perf_counter()

    # Create mesh file
    esmf_creation.create_mesh(esmf_cfg)

    # Stop timer for ESMF Mesh production script
    end = time.perf_counter()

    # Report the time it took for ESMF Mesh production script 
    print(f"Time taken for ESMF mesh production: {end - start:.6f} seconds")

    # Start timer for forcing extraction script
    start = time.perf_counter()

    # Extract forcing
    forcing_extraction.retrieve_forcing(extract_cfg)

    # Stop timer for forcing extraction script
    end = time.perf_counter()

    # Report the time it took for forcing extraction script
    print(f"Time taken for forcing file extraction: {end - start:.6f} seconds")

def main():
    """
    Main function to handle command-line execution.

    This function parses command-line arguments and calls the execute() method.
    It allows the script to be run both programmatically or from the command line.

    :return: None
    """
    # Parse command-line arguments
    args = get_options()

    # Call execute with parsed arguments
    execute(forcing_config_input=args.forcing_config_input)


def get_options():
    """
    Function to accept and parse arguments.

    This function handles the command-line argument parsing and returns the parsed arguments.

    :return: An argparse.Namespace object containing the parsed arguments
    """
    # TODO keyword arguments should start with --
    parser = argparse.ArgumentParser()
    parser.add_argument('forcing_config_input',
                        type=str,
                        help='Path to forcing engine configuration file for forecast run')
    return parser.parse_args()


if __name__ == '__main__':
    main()
