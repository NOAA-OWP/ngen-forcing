from NextGen_lumped_forcings_driver import NextGen_lumped_forcings_driver
from multiprocessing import freeze_support
import argparse
import os
import re

def create_output_directory(hydrofab_path):
    """
    Creates output directory based on the hydrofabric filename.
    Returns the path to the created directory.
    """
    # Extract the filename from the path
    filename = os.path.basename(hydrofab_path)
    
    # Extract the gage number using regex
    match = re.search(r'gages-(\d+)', filename)
    if match:
        gage_number = match.group(1)
        # Create the output directory path
        output_dir = f"/srv/data/AORC_2.2/Gage_{gage_number}"
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir
    else:
        raise ValueError(f"Could not extract gage number from filename: {filename}")

def execute(args):
    
    hyfab_in=args.hydrofab_path
    
    output_dir = create_output_directory(hyfab_in)
    
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
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_options()
    freeze_support()    
    execute(args) 
