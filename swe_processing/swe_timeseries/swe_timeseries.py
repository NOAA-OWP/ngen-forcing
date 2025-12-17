import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import glob
import os
import re
import argparse
import fsspec
import time
from swe_mapping.utility.geo_utils import GeoUtils
from swe_mapping.utility.snotel_utils import SnotelDataLoader, SnotelCalculator

class FileLoader:
    """Handles loading and retrieving files"""
    
    @staticmethod
    def get_times(csv_files):
        """
        Create an array of 06z timestamps given start and end date.

        Parameters
        ----------
        csv_files : str
            Path to the ngen csv files.

        Returns
        -------
        numpy.ndarray
            Array of 06z datetime objects for each day
        """

        file_path = csv_files[0]
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()
        df['time'] = pd.to_datetime(df['time'])

        start_date = min(df['time'])
        end_date = max(df['time'])

        # Populate with 06z timesteps from within the start and end
        times = np.arange(start_date, np.datetime64(end_date),
                         np.timedelta64(1, 'D')).astype('datetime64[ns]')
        times = times + np.timedelta64(6, 'h') 

        return times

    @staticmethod
    def get_filenames(directory):
        """
        Create a list of pathnames pointing to ngen csv output files.

        Parameters
        ----------
        directory : str
            The path where .csv files are located
            
        Returns
        -------
        list
            A list of csv filenames found in the directory provided
        """
        pattern = os.path.join(directory, "cat-*.csv")
        csv_files = glob.glob(pattern)

        if csv_files:
            return csv_files
        else:
            raise Exception(f"No csv files found in {directory}")

    @staticmethod
    def get_ids(csv_files):
        """
        Extract catchment IDs from filenames.
        
        Parameters
        ----------
        csv_files : list
            List of csv file paths
            
        Returns
        -------
        numpy.ndarray
            Array of catchment IDs
        """
        # Stop if csv_files is empty
        if not csv_files:
            raise ValueError("No CSV files found in the directory. Processing halted.")
        
        # Same code as convert_swe
        catchment_ids = np.array([
            int(match.group(1))  # Extract the number safely
            for f in csv_files
            if (match := re.search(r'cat-(\d+)', os.path.basename(f)))  # Store the match
        ])

        
        # Stop if csv_files was not empty, but no catchment_ids were parsed
        if len(catchment_ids) == 0:
            raise ValueError("No valid catchment CSV files found (files must match 'cat-{number}.csv' pattern).")
        
        return catchment_ids

class S3Loader:
    """Handles operations related to S3 storage"""
    
    @staticmethod
    def construct_s3_path(gpkg_file, direct_s3, s3_mount_point, snodas_s3_path):
        """
        Parse/construct SNODAS s3 path from gpkg filename.
        
        Parameters
        ----------
        gpkg_file : str
            geopackage path
            
        Returns
        -------
        string
            a string containing the s3 path
        """       
        filename = os.path.basename(gpkg_file)
        prefix = filename.split('.')[0]
        basin_id = re.search(r'(\d+)', prefix).group(1)
        path = snodas_s3_path

        # Construct the path using the numeric part and s3 options
        if not direct_s3:
            fs = fsspec.filesystem('local')
            s3_path = f"{s3_mount_point}/{path}/gages-{basin_id}_swe.csv"
            if os.path.exists(s3_path):
                return s3_path, basin_id
            else:
                raise FileNotFoundError(f"Could not find local csv file: {s3_path}")
        else:
            fs = fsspec.filesystem('s3')
            s3_path = f"s3://{path}/gages-{basin_id}_swe.csv"
            if fs.exists(s3_path):
                return s3_path, basin_id 
            else:
                raise FileNotFoundError(f"Could not find S3 csv file: {s3_path}")

    @staticmethod
    def read_csv_from_s3(s3_path):
        """
        Read a CSV file from an S3 bucket.
        
        Parameters
        ----------
        s3_path : str
            The s3 path to the csv file
            
        Returns
        -------
        pandas.DataFrame
            DataFrame containing the CSV data
        """
        try:
            # Use fsspec to open the file
            with fsspec.open(s3_path) as f:
                df = pd.read_csv(f)
            
            return df
        
        except Exception as e:
            print(f"Error reading S3 file {s3_path}: {e}")
            return None

class DataParser:
    """Handles parsing and processing of data"""
    
    @staticmethod
    def parse_snodas_dataframe(snodas_df, times):
        """
        Extract snodas SWE values for specified dates.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing SNODAS basin-averaged SWE data
        times : numpy.ndarray
            Array of datetime objects representing the time points to extract

        Returns
        -------
        numpy.ndarray
            1D numpy array of basin-averaged SWE values corresponding to the provided times
        """
        try:
            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(snodas_df['timestamp']):
                snodas_df['timestamp'] = pd.to_datetime(snodas_df['timestamp'])
            
            # Sort data by timestamp to ensure proper time ordering
            snodas_df = snodas_df.sort_values('timestamp')
            
            # Initialize an array for the basin average values
            snodas_data = np.full(len(times), np.nan)
            
            # Create a mask for the dates we want to extract
            mask = snodas_df['timestamp'].isin(times)
            
            if not mask.any():
                print(f"Warning: No matching timestamps within date range found in dataframe")
                return snodas_data
            
            # Extract the filtered data
            filtered_df = snodas_df[mask].copy()
            
            # Create a dictionary for quick lookup of SWE values by timestamp
            swe_dict = dict(zip(filtered_df['timestamp'], filtered_df['basin_avg_swe']))


            # Populate the basin_avg_data array with values from the dictionary
            for i, t in enumerate(times):
                if not isinstance(t, pd.Timestamp):
                    t_timestamp = pd.to_datetime(t)
                else:
                    t_timestamp = t
                if t_timestamp in swe_dict:
                    snodas_data[i] = swe_dict[t_timestamp]
            
            return snodas_data
            
        except Exception as e:
            print(f"Error processing basin average data from dataframe: {e}")
            return np.full(len(times), np.nan)

    @staticmethod
    def parse_swe_data(csv_files, catchment_ids, times):
        """
        Extract 06Z SWE values for specified dates from all catchments.

        Parameters
        ----------
        csv_files : list
            List of CSV file paths
        catchment_ids : numpy.ndarray
            Array of catchment IDs
        times : numpy.ndarray
            Array of datetime objects

        Returns
        -------
        numpy.ndarray
            2D numpy array (time x catchment) of SWE values
        """
        # Initialize data array - 2d (times, ids)
        data = np.full((len(times), len(catchment_ids)), np.nan)

        critical_error = False
    
        for idx, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                # Use lower() to make headers case-independent
                df.columns = df.columns.str.lower()
                if 'swe_m' not in df.columns and 'swe_mm' not in df.columns:
                    raise ValueError("SWE column names not found")
                
                df['time'] = pd.to_datetime(df['time'])
                    
                # Check date range - these are critical errors we want to exit on
                if max(times) > max(df['time']):
                    raise ValueError(f"End date out of range...max: {max(df['time'])}.")
                elif min(times) < min(df['time']):
                    raise ValueError(f"Start date out of range...min: {min(df['time'])}.")
               
                # Use only selected date/times                      
                mask = df['time'].isin(times)
                if not mask.any():
                    continue
                    
                # Extract and store specified values
                if 'swe_m' in df.columns:    
                    values = df.loc[mask, 'swe_m'].values
                elif 'swe_mm' in df.columns:
                    values = (df.loc[mask, 'swe_mm'].values)/1000
                    
                data[:, idx] = values
                
            except ValueError as ve:
                # For ValueError, set flag and break loop
                print(f"Critical error with {file_path}: {ve}")
                critical_error = True
                break
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if critical_error:
            raise ValueError("Processing stopped due to critical error.")
            
        return data


class CatchmentData:
    """Handles catchment geographic information"""
    
    @staticmethod
    def read_catchment_areas(gpkg_file):
        """
        Read catchment areas from geopackage file.
        
        Parameters
        ----------
        gpkg_file : str
            Path to geopackage file
            
        Returns
        -------
        dict
            Mapping of catchment IDs to their areas
        """
        try:
            # Read divides layer
            gdf = gpd.read_file(gpkg_file, layer='divides')
            
            # Extract just the catchment numbers from divide_id
            catchment_ids = pd.to_numeric(gdf['divide_id'].str.replace('cat-', 
                                                                    '', 
                                                                    regex=False))
            
            # Get areas from geometry
            areas = gdf['areasqkm']
                
            # Create dictionary with integer keys
            area_dict = dict(zip(catchment_ids, areas))
            
            return area_dict
        except Exception as e:
            error_message = (f"Error reading geopackage file: {e}")
            raise ValueError(error_message)


class SWEAnalyzer:
    """Analyzes SWE data across catchments."""
    
    @staticmethod
    def calculate_basin_average(data, catchment_ids, areas):
        """
        Calculate area-weighted basin average SWE.
        
        Parameters
        ----------
        data : numpy.ndarray
            2D array of SWE values (time x catchment)
        catchment_ids : numpy.ndarray
            Array of catchment IDs
        areas : dict
            Dictionary mapping catchment IDs to areas
            
        Returns
        -------
        numpy.ndarray
            Basin-averaged SWE values for each timestep
        """
        # Convert catchment_ids to integers if they aren't already
        catchment_ids = np.array([int(cid) for cid in catchment_ids])
        
        try:
            # Create an array of area values for each catchment
            weights = np.array([areas[int(cid)] for cid in catchment_ids])
            
            # Convert area values to percentages for weight calculations
            weights = weights / np.sum(weights)

            # Calculate weighted average across catchments for each timestep
            # Weights is converted to 2d for np operations
            basin_avg = np.sum(data * weights[np.newaxis, :], axis=1)

            return basin_avg
            
        except KeyError as e:
            print(f"Error: Cannot find area for catchment {e}")
            print(f"Catchment ID type: {type(e.args[0])}")
            print(f"Area dictionary key type: {type(list(areas.keys())[0])}")
            raise


class SWEPlotter:
    """Handles visualization of SWE data"""
    
    @staticmethod
    def add_grids(ax):
        """
        Add grid lines to the plot.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to add grid lines to
        """
        ax.grid(True, which='major', linestyle='--', 
                alpha=0.8, color='darkgray')
        ax.grid(True, which='minor', linestyle=':', 
                alpha=0.4, color='gray')

    @staticmethod
    def titles_labels(ax, basin_id):
        """
        Add title and axis labels.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to add title and labels to
        """
        ax.set_title(f"Basin Average SWE Comparison (Basin {basin_id})", fontsize=14, pad=15)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("SWE (m)", fontsize=12)

    @staticmethod
    def get_x_intervals(times):
        """
        Determine x-axis intervals and labels based on time span.
        
        Parameters
        ----------
        times : numpy.ndarray
            Array of datetime objects
            
        Returns
        -------
        tuple
            (x_major_interval, x_minor_interval, date_fmt)
        """
        # Calculate date range for dynamic interval
        time_range = np.ptp(times).astype('timedelta64[D]').item().days

        target_major_ticks = 15
        
        # Calculate major interval and round to whole days
        raw_major_interval = time_range / target_major_ticks
        x_major_interval = max(1, round(raw_major_interval))
        
        # Set minor interval to half the major interval
        x_minor_interval = max(1, x_major_interval // 2)
        
        # Determine date format based on the overall time range
        if time_range <= 60:  # Up to 2 months
            date_fmt = '%Y-%m-%d'
        else:
            date_fmt = '%Y-%m'

        return x_major_interval, x_minor_interval, date_fmt

    @staticmethod
    def calculate_y_lims(simulated_avg, snodas_avg, snotel_data=None):
        # Calculate y-axis range for dynamic intervals
        sim_y_min, sim_y_max = np.nanmin(simulated_avg), np.nanmax(simulated_avg)
        snodas_y_min, snodas_y_max = np.nanmin(snodas_avg), np.nanmax(snodas_avg)

        y_min = min(sim_y_min, snodas_y_min)
        y_max = max(sim_y_max, snodas_y_max)
        
        # Include SNOTEL data in the y-axis range if available
        if snotel_data:
            for station_id, data in snotel_data.items():
                snotel_y_min = np.nanmin(data['swe'])
                snotel_y_max = np.nanmax(data['swe'])
                y_min = min(y_min, snotel_y_min)
                y_max = max(y_max, snotel_y_max)

        y_range = y_max - y_min
        
        # Add a small buffer to the y-axis limits
        y_buffer = y_range * 0.025
        y_lim_min = max(0, y_min - y_buffer)
        y_lim_max = y_max + y_buffer

        return y_lim_min, y_lim_max, y_range

    @staticmethod
    def get_y_intervals(y_range):
        """
        Set y intervals based on the range of swe values.
        
        Parameters
        ----------
        y_range : float
            range in values between y_min and y_max

        Returns
        -------
        tuple
            (y_major_interval, y_minor_interval, y_format)
        """
        
        # Target number of major ticks
        target_major_ticks = 10
        
        # Calculate the major interval to get 10 ticks
        y_major_interval = y_range / target_major_ticks
        
        # Find the appropriate magnitude (0.001, 0.01, 0.1, etc.)
        magnitude = 10 ** np.floor(np.log10(y_major_interval))

        if magnitude == 0.001:
            y_format = '%.3f'
        elif magnitude == 0.01:
            y_format = '%.2f'
        elif magnitude == 0.1:
            y_format = '%.1f'
        else:
            y_format = '%.0f'
        
        # Round to increments of the magnitude to make prettier intervals
        if y_major_interval / magnitude <= 1:
            y_major_interval = magnitude
        elif y_major_interval / magnitude <= 2:
            y_major_interval = 2 * magnitude
        elif y_major_interval / magnitude <= 5:
            y_major_interval = 5 * magnitude
        else:
            y_major_interval = 10 * magnitude
        
        # Set minor interval to half the major interval
        y_minor_interval = y_major_interval / 2
        
        return y_major_interval, y_minor_interval, y_format

    @staticmethod
    def customize_x_axis(ax, x_major_interval, x_minor_interval, date_fmt):
        """
        Apply x-axis formatting.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to customize
        x_major_interval : int
            Interval for major tick marks
        x_minor_interval : int
            Interval for minor tick marks
        date_fmt : str
            Date format string for tick labels
        """
        ax.xaxis.set_major_formatter(DateFormatter(date_fmt))
        ax.xaxis.set_minor_locator(DayLocator(interval=x_minor_interval))
        ax.xaxis.set_major_locator(DayLocator(interval=x_major_interval))
        ax.tick_params(rotation=45)

    @staticmethod
    def customize_y_axis(ax, y_major_interval, y_minor_interval, 
                         y_lim_min, y_lim_max, y_format):
        """
        Apply y-axis formatting.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to customize
        y_major_interval : float
            Interval for major tick marks
        y_minor_interval : float
            Interval for minor tick marks
        y_lim_min : float
            Lower limit for y-axis
        y_lim_max : float
            Upper limit for y-axis
        y_format : str
            Decimal format for y-axis labels
        """
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter(y_format))
        ax.yaxis.set_major_locator(plt.MultipleLocator(y_major_interval))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(y_minor_interval))
        ax.set_ylim(y_lim_min, y_lim_max)

    @staticmethod
    def plot_basin_average(times, simulated_avg, snodas_avg, snotel_ts=None):
        """
        Create time series plot of basin-averaged SWE.
        
        Parameters
        ----------
        times : numpy.ndarray
            Array of datetime objects
        simulated_avg : numpy.ndarray
            Array of simulated basin-averaged SWE values
        snodas_avg : numpy.ndarray
            Array of SNODAS basin-averaged SWE values
        snotel_ts : dict, optional
            Dictionary of SNOTEL station data
            
        Returns
        -------
        tuple
            (fig, ax) - Figure and axis objects
        """
        # Initialize figure and axis
        fig, ax = plt.subplots(figsize=(10,6))
        
        # Plot data
        ax.plot(times, 
                simulated_avg,
                'b.-',
                markersize=5,
                linewidth=1.5, 
                label='Simulated SWE')
        ax.plot(times, 
                snodas_avg, 
                'g^-', 
                markersize=4, 
                linewidth=1.5,
                alpha=.5,
                label='SNODAS SWE')
        
        # Add SNOTEL data if available
        if snotel_ts:
            cmap = plt.cm.tab10
            
            for i, (station_id, data) in enumerate(snotel_ts.items()):
                color = cmap(i % 30)
                ax.plot(times, 
                        data['swe'], 
                        'o-', 
                        markersize=3, 
                        linewidth=1,
                        alpha=0.7,
                        color=color,
                        label=f'SNOTEL {station_id}')
        
        ax.legend()
        
        return fig, ax

    @staticmethod
    def finalize_plot(fig, output_path):
        """
        Save the plot to file.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to save
        output_path : str
            Path where the plot should be saved
        """
        # Adjust layout to prevent label cutoff
        fig.tight_layout()
        # Save plot
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class SWEProcessor:
    """Main class for processing and visualizing SWE data."""
    
    def __init__(self, csv_directory=None, gpkg_file=None, plot_output=None, csv_output=None, direct_s3=False):
        """
        Initialize SWE processor with input and output parameters.
        
        Parameters
        ----------
        csv_directory : str, optional
            Path to directory containing csv files
        gpkg_file : str, optional
            Path to geopackage file with catchment geometries
        plot_output : str, optional
            Path where plot should be saved
        csv_output : str, optional
            Path where csv data should be saved
        """
        self.csv_directory = csv_directory
        self.gpkg_file = gpkg_file
        self.plot_output = plot_output
        self.csv_output = csv_output
        self.direct_s3 = direct_s3
        
        # Initialize data-related containers
        self.times = None
        self.csv_files = None
        self.catchment_ids = None
        self.simulated_data = None
        self.snodas_data = None
        self.areas = None
        self.simulated_avg = None
        self.snodas_avg = None
        self.s3_path = None
        self.snodas_df = None
        self.basin_id = None
        self.s3_mount_point = None
        self.snodas_s3_path = None
        
        # SNOTEL-related containers
        self.basin_geometry = None
        self.snotel_data = None
        self.snotel_ts = None
        self.snotel_filenames = None
        self.basin_gdf = None
        self.stations_gdf = None
        self.snotel_df = None
        self.snotel_filesystem = None
        self.snotel_s3_path = None
        
        # Initialize visualization parameters
        self.x_major_interval = None
        self.x_minor_interval = None
        self.date_fmt = None
        self.y_major_interval = None
        self.y_minor_interval = None
        self.y_lim_min = None
        self.y_lim_max = None
        self.y_range = None
        self.y_format = None
    
    def load_data(self):
        """Load all required data"""
        self.snotel_s3_path = 'ngwpc-forcing/snotel_csv'
        self.snodas_s3_path = 'ngwpc-forcing/snodas_csv'
        self.s3_mount_point = os.getenv('S3_MOUNT_POINT', os.path.join(os.path.expanduser("~"), 's3'))
        tl0 = time.time()
        self.csv_files = FileLoader.get_filenames(self.csv_directory)
        tl1 = time.time()
        self.times = FileLoader.get_times(self.csv_files)
        tl2 = time.time()
        self.s3_path, self.basin_id = S3Loader.construct_s3_path(self.gpkg_file, 
                                                                 self.direct_s3, 
                                                                 self.s3_mount_point,
                                                                 self.snodas_s3_path)
        tl3 = time.time()
        self.snodas_df = S3Loader.read_csv_from_s3(self.s3_path)
        tl4 = time.time()
        self.catchment_ids = FileLoader.get_ids(self.csv_files)
        tl5 = time.time()
        self.simulated_data = DataParser.parse_swe_data(self.csv_files, 
                                                        self.catchment_ids, 
                                                        self.times)
        tl6 = time.time()
        self.snodas_data = DataParser.parse_snodas_dataframe(self.snodas_df,
                                                             self.times)
        tl7 = time.time()
        self.areas = CatchmentData.read_catchment_areas(self.gpkg_file)
        tl8 = time.time()
        
        #SNOTEL
        self.snotel_filenames, self.snotel_filesystem = SnotelDataLoader.list_snotel_filenames(self.s3_mount_point, 
                                                                                               self.snotel_s3_path,
                                                                                               self.direct_s3)
        self.stations_gdf = SnotelDataLoader.parse_snotel_filenames(self.snotel_filenames)
        self.basin_gdf = GeoUtils.read_geo(self.gpkg_file)
        #Get basin_geometry, but we don't need bounds, so we ignore those
        self.basin_geometry, _ = GeoUtils.get_basin_geometry(self.basin_gdf) 
        tl9 = time.time()

        print(f"\ntime in load_data: {tl9-tl0:.5f}s")
        print(f" - get_filenames time: {tl1-tl0:.5f}s")
        print(f" - get_times time: {tl2-tl1:.5f}s")
        print(f" - contruct_s3_path time: {tl3-tl2:.5f}s")
        print(f" - read_csv_from_s3 time: {tl4-tl3:.5f}s")
        print(f" - get_ids time: {tl5-tl4:.5f}s")
        print(f" - parse_swe_data time: {tl6-tl5:.5f}s")
        print(f" - parse_snodas_data time: {tl7-tl6:.5f}s")
        print(f" - read_catchment_areas time: {tl8-tl7:.5f}s")
        print(f" - snotel data loading time: {tl9-tl8:.5f}s")
        print(f"\nread_from_s3 percentage of load_time: {((tl4-tl3)/(tl9-tl0))*100:.2f}%")
        print(f"parse_swe_data percentage of load_time: {((tl6-tl5)/(tl9-tl0))*100:.2f}%\n")
    
    def analyze_data(self):
        """Perform analysis on loaded data"""
        t0 = time.time()
        self.simulated_avg = SWEAnalyzer.calculate_basin_average(self.simulated_data,
                                                                 self.catchment_ids,
                                                                 self.areas)
        self.snodas_avg = self.snodas_data
        #SNOTEL
        ta = time.time()
        self.stations_in_basin = SnotelCalculator.find_stations_in_basin(self.stations_gdf, 
                                                                         self.basin_geometry)
        ta1 = time.time()                                                                         
        self.snotel_df = SnotelDataLoader.get_snotel_timeseries(self.basin_geometry, 
                                                                self.times, 
                                                                self.stations_in_basin,
                                                                self.snotel_filesystem,
                                                                self.s3_mount_point,
                                                                self.snotel_s3_path)
        ta2 = time.time()
        self.snotel_ts = SnotelDataLoader.extract_snotel_timeseries(self.snotel_df, self.times)        
        tb = time.time()
        snotel_time = tb-ta
        analyze_time = tb-t0
        print(f"    - find_stations_in_basin time: {ta1-ta:.2f}s")
        print(f"    - get_snotel_timeseries time: {ta2-ta1:.2f}s")
        print(f"    - extract_snotel_timeseries time: {tb-ta2:.2f}s")
        print(f" - SNOTEL processing time: {snotel_time:.2f}s")
        print(f" - SNOTEL % of analyze time: {((snotel_time)/(analyze_time))*100:.2f}%")



    def prepare_visualization(self):
        """Calculate parameters for visualization"""
        (self.y_lim_min, 
         self.y_lim_max,
         self.y_range) = SWEPlotter.calculate_y_lims(self.simulated_avg,
                                                     self.snodas_avg,
                                                     self.snotel_ts)
        (self.x_major_interval, 
         self.x_minor_interval, 
         self.date_fmt) = SWEPlotter.get_x_intervals(self.times)
        (self.y_major_interval, 
         self.y_minor_interval,
         self.y_format) = SWEPlotter.get_y_intervals(self.y_range)
    
    def create_plot(self):
        """Create and save the plot"""
        snotel_ts = self.snotel_ts
        
        fig, ax = SWEPlotter.plot_basin_average(self.times, 
                                                self.simulated_avg, 
                                                self.snodas_avg,
                                                snotel_ts)
        SWEPlotter.customize_x_axis(ax, 
                                    self.x_major_interval, 
                                    self.x_minor_interval, 
                                    self.date_fmt)
        SWEPlotter.customize_y_axis(ax, 
                                    self.y_major_interval, 
                                    self.y_minor_interval, 
                                    self.y_lim_min, 
                                    self.y_lim_max,
                                    self.y_format)
        SWEPlotter.add_grids(ax)
        SWEPlotter.titles_labels(ax, self.basin_id)
        SWEPlotter.finalize_plot(fig, self.plot_output)
        print(f"Basin average SWE data plot saved to {self.plot_output}")

    def save_basin_avg_to_csv(self):
        """Save basin average SWE data to csv file"""
        if (self.csv_output is None or 
            self.simulated_avg is None or 
            self.times is None):
            return
            
        try:
            # Create basic DataFrame with times and basin average SWE values
            data_dict = {
                'timestamp': self.times,
                'Simulated_SWE': self.simulated_avg,
                'SNODAS_SWE': self.snodas_avg
            }
            
            # Add SNOTEL data if available
            if self.snotel_ts:
                for station_id, station_data in self.snotel_ts.items():
                    data_dict[f'SNOTEL_{station_id}_SWE'] = station_data['swe']
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data_dict)
            df.to_csv(self.csv_output, index=False)
            print(f"Basin average SWE data table saved to {self.csv_output}")
        except Exception as e:
            print(f"Error saving data to CSV: {e}")

    def process(self):
        """Run the processing pipeline"""
        
        t0 = time.time()
        self.load_data()
        t1 = time.time()
        #print(f"load_data time: {t1-t0:.5f}s")

        t2 = time.time()
        self.analyze_data()
        t3 = time.time()
        print(f"analyze_data time: {t3-t2:.5f}s\n")

        # Export data to csv if csv_output is provided
        if self.csv_output:
            t4 = time.time()
            self.save_basin_avg_to_csv()
            t5 = time.time()
            print(f"save_to_csv time: {t5-t4:.5f}s")
        # Generate visualization if plot_output is provided
        if self.plot_output:
            t6 = time.time()
            self.prepare_visualization()
            t7 = time.time()
            print(f"prepare_visualization time: {t7-t6:.5f}s")
            t8 = time.time()
            self.create_plot()
            t9 = time.time()
            print(f"create_plot time: {t9-t8:.5f}s")
            print(f"\nload_data percentage of total runtime: {((t1-t0)/(t9-t0))*100:.2f}%")
            print(f"analyze_data percentage of total runtime: {((t3-t2)/(t9-t0))*100:.2f}%")
            print(f"create_plot percentage of total runtime: {(t9-t8)/(t9-t0)*100:.2f}%")
            print(f"\ntotal runtime: {t9-t0:.5f}s")

def get_options(args_list=None):
    """
    Parse command line arguments.
    
    Parameters
    ----------
    args_list : list, optional
        List of command line arguments for programmatic execution
        
    Returns
    -------
    argparse.Namespace
        Parsed arguments from command line or list
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_directory', type=str, 
                       help="Path that contains ngen csv files.")
    parser.add_argument('gpkg_file', type=str,
                       help="Path to geopackage file containing catchment geometries.")
    parser.add_argument('--plot_output', type=str, default=None,
                       help="Optional output path for the simulated SWE time series PNG file.")
    parser.add_argument('--csv_output', type=str, default=None,
                       help="Optional output path for the basin average SWE data CSV file.")
    parser.add_argument('--direct_s3', action='store_true', 
                         help='Use direct S3 access instead of local mount')
    
    if args_list is not None:
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()

def execute(args):
    """
    Execute the SWE processor.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """ 
    # Create and run the SWE processor
    processor = SWEProcessor(
        csv_directory=args.csv_directory,
        gpkg_file=args.gpkg_file,
        plot_output=args.plot_output,
        csv_output=args.csv_output,
        direct_s3=args.direct_s3
    )
    processor.process()

def swe_ts(args_list=None):
    """
    Main entry point for the script.
    
    Parameters
    ----------
    args_list : list, optional
        List of command line arguments for programmatic execution
    """
    args = get_options(args_list)
    execute(args)
    
if __name__ == "__main__":
    swe_ts()

