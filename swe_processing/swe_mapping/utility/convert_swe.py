import pandas as pd
import glob
import os
import re
from datetime import datetime
import numpy as np
import xarray as xr
import argparse

def read_swe_values_from_dir(directory, dates):
    """
    Extract 06Z sneqv values for specified dates from all catchments
   
    Args:
        directory (str): Path to directory containing CSV files
        dates (list): List of dates in 'YYYY-MM-DD' format
       
    Returns:
            - catchment_ids: numpy array of catchment IDs
            - times: numpy array of datetime objects
            - data: 2D numpy array (time x catchment) of swe values
    """
   
    # Convert dates to datetime and add 06z timestamp
    times = np.array([datetime.strptime(f"{date} 06:00:00", "%Y-%m-%d %H:%M:%S")
                     for date in dates])
   
    # Get all catchment files
    pattern = os.path.join(directory, "cat-*.csv")
    csv_files = glob.glob(pattern)
   
    # Extract catchment IDs from filenames

    if not csv_files:
        raise Exception(f"No CSV files found in {directory}")

    catchment_ids = np.array([
        int(match.group(1))  # Extract the number safely
        for f in csv_files
        if (match := re.search(r'cat-(\d+)', os.path.basename(f)))  # Store the match
    ])

    if catchment_ids.size == 0:
        raise Exception(f'No valid catchment files found in {directory}: {csv_files}')

    print(f"catchment_ids: {catchment_ids}")

    # Initialize data array - 2d (times, ids)
    data = np.full((len(times), len(catchment_ids)), np.nan)

    missing_swe_data = False
    # Parse SWE values from each file
    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            # Use lower() to make case-insensitive
            df.columns = df.columns.str.lower()

            if 'swe_m' not in df.columns and 'swe_mm' not in df.columns:
                print(f"SWE columns not found in {file_path}")
                missing_swe_data = True
                continue

            # Use only selected date/times    
            df['time'] = pd.to_datetime(df['time'])
            mask = df['time'].isin(times)
            if not mask.any():
                continue

            # Extract and store SWE values
            if 'swe_m' in df.columns:
                values = df.loc[mask, 'swe_m'].values
            elif 'swe_mm' in df.columns:
                values = df.loc[mask, 'swe_mm'].values / 1000  # Convert mm to meters
            data[:, idx] = values

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Check if any files were missing SWE data
    if missing_swe_data:
        raise Exception("One or more files were missing SWE data.")

    return catchment_ids, times, data

def write_to_netcdf(catchment_ids, times, data, output_file):
    """
    Write the extracted swe values to a NetCDF file with dates as strings
    """
   
    # Use xarray to construct the dataset for writing
    ds = xr.Dataset(
        data_vars={
            "swe": (["date", "catchment"], data)
        },
        coords={
            "date": [t.strftime('%Y-%m-%d') for t in times],
            "catchment": catchment_ids
        }
    )
   
    # write to netcdf output file
    ds.to_netcdf(output_file)

def get_options(args_list=None):
    """Read and pass in command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_directory', type=str, 
                        help="Path that contains csv ngen files.")
    parser.add_argument('dates', nargs='+',
                        help="Dates to process ex: '2015-12-01' '2015-12-02'")
    parser.add_argument('output', type=str,
                        help="Desired path for output file.")
    return parser.parse_args(args_list)

def main(args_list=None):
    args = get_options(args_list)
    directory = args.csv_directory
    dates = args.dates
    output = args.output
   
    catchment_ids, times, data = read_swe_values_from_dir(directory, dates)
    print(f"Converted {len(catchment_ids)} catchments")
    #print(f"Time steps: {times}")
    #print(f"Data shape: {data.shape}")
   
    write_to_netcdf(catchment_ids, times, data, output)
if __name__ == "__main__":
    main()

