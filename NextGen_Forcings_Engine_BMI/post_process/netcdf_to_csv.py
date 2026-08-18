"""NetCDF to CSV Converter for atmospheric forcing data.

Converts bmi engine .nc forcing output into multiple CSV files, for use in ngen.
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr


class NCDataProcessor:
    """Static methods for NetCDF data processing."""

    @staticmethod
    def open_netcdf(file_path):
        """Open a NetCDF file using xarray.

        Parameters
        ----------
        file_path : str
            Path to the NetCDF file

        Returns
        -------
        xarray.Dataset

        """
        dataset = xr.open_dataset(file_path, engine="netcdf4")

        return dataset

    @staticmethod
    def get_catchment_ids(dataset):
        """Extract catchment IDs from the dataset.

        Parameters
        ----------
        dataset : xarray.Dataset
            The NetCDF dataset

        Returns
        -------
        list
            List of catchment IDs

        """
        catchment_ids = dataset.ids.values.tolist()

        return catchment_ids

    @staticmethod
    def convert_time_format(time_values):
        """Convert time values to the format expected by ngen.

        Parameters
        ----------
        time_values : numpy.ndarray
            Array of time values as numpy.datetime64

        Returns
        -------
        list
            List of formatted time strings (YYYY-MM-DD HH:MM:SS)

        """
        formatted_times = []

        # process and reformat datetime64 values
        for t in time_values:
            dt = pd.Timestamp(t).to_pydatetime()
            formatted_times.append(dt.strftime("%Y-%m-%d %H:%M:%S"))

        return formatted_times

    @staticmethod
    def apply_scaling(data, scale_factor=None, add_offset=None):
        """Apply scaling factor and offset to data if specified.

        Parameters
        ----------
        data : numpy.ndarray
            Data array
        scale_factor : float, optional
            Scale factor to apply
        add_offset : float, optional
            Offset to apply

        Returns
        -------
        numpy.ndarray
            Scaled data

        """
        result = data.copy()

        if scale_factor is not None:
            result = result * scale_factor

        if add_offset is not None:
            result = result + add_offset

        return result

    @staticmethod
    def extract_variable_data(dataset, var_name, catchment_idx):
        """Extract data for a variable and apply scaling if needed.

        Parameters
        ----------
        dataset : xarray.Dataset
            The NetCDF dataset
        var_name : str
            Variable name
        catchment_idx : int
            Index of the catchment

        Returns
        -------
        numpy.ndarray
            Processed variable data

        """
        # Extract the data for the specified catchment
        var_data = dataset[var_name].values[:, catchment_idx]

        # Get scale factor and offset if they exist
        scale_factor = dataset[var_name].attrs.get("scale_factor")
        add_offset = dataset[var_name].attrs.get("add_offset")

        # Check for fill values and mask before scaling
        fill_value = dataset[var_name].attrs.get("_FillValue")
        if fill_value is not None:
            mask = var_data != fill_value
            var_data = np.where(mask, var_data, np.nan)

        # Apply scaling and offset if they exist
        scaled_data = NCDataProcessor.apply_scaling(var_data, scale_factor, add_offset)

        # Handle any NaN values created from fill values
        if fill_value is not None:
            scaled_data = np.nan_to_num(scaled_data, nan=0.0)

        return scaled_data


class CSVWriter:
    """Static methods for CSV creation and writing."""

    @staticmethod
    def create_csv_dataframe(formatted_times, variables_data):
        """Create a pandas DataFrame for CSV output.

        Parameters
        ----------
        formatted_times : list
            List of formatted time strings
        variables_data : dict
            Dictionary of variable names and their data

        Returns
        -------
        pandas.DataFrame
            DataFrame ready for CSV output

        """
        data = {"Time": formatted_times}
        data.update(variables_data)
        return pd.DataFrame(data)

    @staticmethod
    def write_csv_file(df, output_path):
        """Write DataFrame to CSV file.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to write
        output_path : str
            Path where the CSV file will be saved

        Returns
        -------
        bool
            True if successful, False otherwise

        """
        try:
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error writing CSV: {e}")
            return False


class NetCDFtoCSVConverter:
    """Orchestrator class to manage conversion process."""

    def __init__(self, input_path, output_path):
        """Initialize converter with input and output file paths.

        Parameters
        ----------
        input_path : str
            Path to the input NetCDF file
        output_path : str
            Directory where CSV files will be saved

        """
        self.input_path = input_path
        # self.output_dir = os.path.dirname(output_path)
        self.output_dir = output_path.rstrip(os.sep)
        self.dataset = None
        self.variable_names = [
            "U2D",
            "V2D",
            "LWDOWN",
            "RAINRATE",
            "T2D",
            "Q2D",
            "PSFC",
            "SWDOWN",
        ]

    def run(self):
        """Execute the conversion process."""
        print(f"Opening NetCDF file: {self.input_path}")
        self.dataset = NCDataProcessor.open_netcdf(self.input_path)

        # Get catchment IDs and time values
        catchment_ids = NCDataProcessor.get_catchment_ids(self.dataset)
        time_values = self.dataset.Time.values
        formatted_times = NCDataProcessor.convert_time_format(time_values)

        csv_count = 0

        # Process each catchment
        for idx, catchment_id in enumerate(catchment_ids):
            # Extract data for all variables for this catchment
            variables_data = {}
            for var_name in self.variable_names:
                variables_data[var_name] = NCDataProcessor.extract_variable_data(
                    self.dataset, var_name, idx
                )

            # Create dataframe
            df = CSVWriter.create_csv_dataframe(formatted_times, variables_data)

            # Write CSV file
            output_filename = f"{catchment_id}.csv"
            output_path = os.path.join(self.output_dir, output_filename)

            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            if CSVWriter.write_csv_file(df, output_path):
                csv_count += 1

        print(f"Successfully created {csv_count} CSV files in {self.output_dir}")


def get_options():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert NetCDF atmospheric forcing data to CSV files."
    )
    parser.add_argument("input_path", help="Path to the input NetCDF file")
    parser.add_argument("output_path", help="Path for csv output files.")
    args = parser.parse_args()

    return args


def execute():
    """Parse arguments and run the converter."""
    args = get_options()
    print("netcdf_to_csv args:", vars(args))

    converter = NetCDFtoCSVConverter(args.input_path, args.output_path)
    converter.run()


if __name__ == "__main__":
    execute()
