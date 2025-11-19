"""
BMI Forcings Engine standalone mode wrapper script.

Provides ability to run the BMI Forcings Engine pipeline in standalone mode using a single command.

example usage: python bmi_wrapper.py short_range Gage_01011000.gpkg
"""

import argparse
import os
import tempfile
import subprocess
from datetime import datetime, timedelta

import yaml
from types import SimpleNamespace
from git_util import print_git_info_all

import forcing_extraction
import esmf_creation


def execute(forcing_config_input: str, config_input: str = None, output_path: str = None, csv_path: str = None, np: str = None):
    """
    Execute the full forcings engine BMI pipeline in standalone mode.

    Modules executed: ESMF Mesh Conversion, Forcing Extraction, Forcing Engine BMI.

    This method accepts the cycle name, hydrofabric file, configuration file path,
    output path, and number of processes to run the BMI Forcings Engine pipeline.
    It handles mesh conversion, forcing extraction, and finally the execution of the
    BMI engine using the specified parameters.

    :param forcing_config_input: Path to forcing engine configuration file for forecast run
    :param config_input: Optional path to the wrapper config file.
    :param output_path: Optional full path to specify forcing engine output location.
    :param csv_path: Optional path for CSV output, if desired.
    :param np: Optional number of processes to use.
    :return: None
    """
    print_git_info_all()

    # Read in the configuration file to access paths and settings
    if config_input:
        config_read = config_input
    else:
        config_read = './wrapper_config.yml'

    with open(config_read, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Read in forcing engine configuration file
    with open(forcing_config_input, 'r') as forcing_config_file:
        forcing_config = yaml.safe_load(forcing_config_file)

    # Wrap config dict into simplenamespace to match esmf creation ConfigOptions format
    esmf_cfg = SimpleNamespace(geopackage=forcing_config['Geopackage'],
                               geogrid=forcing_config['GeogridIn'])

    # Wrap config dict into simplenamespace to match forcing extraction ConfigOptions format
    extract_cfg = SimpleNamespace(b_date_proc=datetime.strptime(forcing_config['RefcstBDateProc'], "%Y%m%d%H%M"),
                                  input_forcings=forcing_config['InputForcings'],
                                  supp_precip_forcings=forcing_config['SuppPcp'],
                                  input_force_dirs=forcing_config['InputForcingDirectories'],
                                  supp_precip_dirs=forcing_config['SuppPcpDirectories'],
                                  fcst_input_horizons=forcing_config['ForecastInputHorizons'],
                                  cfsv2EnsMember=forcing_config['cfsEnsNumber'],
                                  ana_flag=forcing_config['AnAFlag'],
                                  look_back=forcing_config['LookBack'])

    # Create mesh file
    esmf_creation.create_mesh(esmf_cfg)

    # Extract forcing
    forcing_extraction.retrieve_forcing(extract_cfg)

    # Set the mesh file name based on the hydrofabric file
    base_geo_name = os.path.splitext(os.path.basename(forcing_config['Geopackage']))[0]
    mesh_fileName = f"{base_geo_name}_ESMF_Mesh.nc"

    # Extract paths and environment names from the configuration file
    mesh_outPath = os.path.join(config['global']['mesh_out_base_path'], mesh_fileName)
    bmi_scriptPath = config['global']['bmi_script_path']

    # Get parameters from the forcing engine config file
    refcstbdate = datetime.strptime(forcing_config['RefcstBDateProc'], "%Y%m%d%H%M")
    input_horizons = forcing_config['ForecastInputHorizons']
    input_horizons = input_horizons + [input_horizons[0]] * len(forcing_config['SuppPcp'])
    ana_flag = forcing_config['AnAFlag']
    look_back = forcing_config['LookBack']

    # Set time variables for forcing engine
    b_date_dt = refcstbdate
    b_date = b_date_dt.strftime("%Y%m%d%H%M")

    if ana_flag == 0:
        start_time_dt = b_date_dt + timedelta(hours=1)
        end_time_dt = b_date_dt + timedelta(minutes=input_horizons[0])
    if ana_flag == 1:
        end_time_dt = b_date_dt - timedelta(hours=1)
        start_time_dt = b_date_dt - timedelta(minutes=(look_back))

    start_time = start_time_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_time = end_time_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Construct output path for forcing engine
    output_path = (
        output_path or tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name if csv_path
        else None
    )

    # Build command for the BMI engine
    command = []

    # Optional: mpirun prefix
    if np is not None:
        command += ["mpirun", "-np", str(np)]

    # Main Python call
    command += [
        "python3", bmi_scriptPath,
        f"-config_path={forcing_config_input}",
        f"-b_date={b_date}",
        f"-geogrid={mesh_outPath}",
    ]

    # Optional: -output_path
    if output_path:
        command.append(f"-output_path={output_path}")

    # Always add start/end time
    command += [start_time, end_time]

    # Now run using utility
    subprocess.run(command, check=True)

    if csv_path:
        # Get the directory of the current Python module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the full path to the script
        post_process_script = os.path.join(module_dir, "post_process", "netcdf_to_csv.py")

        subprocess.run(
            ["python3", post_process_script, f"{output_path}", f"{csv_path}"],
            check=True
        )


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
    execute(
        forcing_config_input=args.forcing_config_input,
        config_input=args.config_input,
        output_path=args.output_path,
        np=args.np,
        csv_path=args.csv_path
    )


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
    parser.add_argument('-output_path',
                        type=str,
                        help='Full path for nc output file. If omitted, and -csv_path is provided, output_path will be set to /tmp/temp.nc.')
    parser.add_argument('-csv_path',
                        type=str,
                        help='Path for csv output, if desired. If omitted, no csv files will be created.')
    parser.add_argument('-config_input',
                        type=str,
                        help='Path to wrapper config file. If omitted, defaults to ./wrapper_config.yml')
    parser.add_argument('-np',
                        type=int,
                        help='The number of processes to use when executing the forcing engine. If omitted, will default to one process.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
