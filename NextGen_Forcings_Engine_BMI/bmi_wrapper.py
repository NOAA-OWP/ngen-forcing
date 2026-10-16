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

def execute(forcing_config_input: str, output_path: str = None, csv_path: str = None, np: str = None):
    """
    Execute the NextGen Forcings Engine BMI pipeline in standalone mode.

    Modules executed: Forcing Engine BMI.

    This method accepts the cycle name, hydrofabric file, configuration file path,
    output path, and number of processes to run the BMI Forcings Engine pipeline.
    This handles the execution of the BMI forcing engine using the specified parameters.

    :param forcing_config_input: Path to forcing engine configuration file for forecast.
    :param output_path: Optional full path to specify forcing engine output location.
    :param csv_path: Optional path for CSV output, if desired.
    :param np: Optional number of processes to use.
    :return: None
    """

    # Read in forcing engine configuration file
    with open(forcing_config_input, 'r') as forcing_config_file:
        forcing_config = yaml.safe_load(forcing_config_file)


    # Extract ESMF Mesh file pathway from forcing engine configuration file
    mesh_outPath = forcing_config['GeogridIn']

    # Get parameters from the forcing engine configuration  file
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

    # Look for $PYTHON, fallback to 'python3' if it isn't set
    python_bin = os.environ.get('PYTHON', 'python3')
    
    # Build command for the BMI engine
    command = []

    # Optional: mpirun prefix
    if np is not None:
        command += ["mpirun", "-np", str(np)]

    # Main Python call
    command += [
        python_bin, "./run_bmi_model.py",
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
            [python_bin, post_process_script, f"{output_path}", f"{csv_path}"],
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
    parser.add_argument('-np',
                        type=int,
                        help='The number of processes to use when executing the forcing engine. If omitted, will default to one process.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
