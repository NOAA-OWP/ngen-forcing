from NextGen_lumped_forcings_driver import NextGen_lumped_forcings_driver
from multiprocessing import freeze_support
import argparse
import os
import re
from pathlib import Path

def dir_path(string):
    """Custom function for argparse to handle and create directories."""
    path = Path(string)

    # Create the directory and any missing parent directories safely
    if not path.exists():
        print(f"Directory '{path}' does not exist. Creating it now...")
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        # Raise an error if the path exists but is actually a file
        raise argparse.ArgumentTypeError(f"'{path}' already exists and is a file, not a directory.")
    return path

def execute(args):
    
    hyfab_in=args.hydrofab_path
    
    output_dir = args.output_dir
    
    NextGen_lumped_forcings_driver(
        output_dir, 
        start_time="2013-01-01 00:00:00",
        end_time="2022-12-31 23:00:00",
        met_dataset="AORC",
        hyfabfile=hyfab_in,
        hyfabfile_parquet=None,
        met_dataset_pathway="s3://noaa-nws-aorc-v1-1-1km/",
        weights_file=None,
        netcdf=False,
        csv=True,
        bias_calibration=False,
        downscaling=False,
        CONUS=False,
        AnA=False,
        num_processes=4
    )
    
def get_options():
    '''
    Function to accept and parse arguments.
    
    Returns an argparse object.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('hydrofab_path', help='full path to hydrofabric file')
    parser.add_argument('-o', '--output_dir',type=dir_path,required=True,help="Path to the output directory (will be created if missing)")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_options()
    freeze_support()    
    execute(args) 
