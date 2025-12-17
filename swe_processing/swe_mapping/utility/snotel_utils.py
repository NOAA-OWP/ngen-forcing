import re
import fsspec
import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point

class SnotelDataLoader:
    @staticmethod
    def list_snotel_filenames(s3_mount_point, snotel_s3_path, direct_s3):
        """
        List SNOTEL CSV files available in the S3 bucket.
        
        Returns
        -------
        list
            List of filenames (strings) of SNOTEL CSV files in the S3 bucket or local mount
        """

        path = snotel_s3_path
        
        if not direct_s3:
            fs = fsspec.filesystem('local')
            full_path = f"{s3_mount_point}/{path}"
            objects = fs.ls(full_path)
            if not objects:
                raise FileNotFoundError(f"Local snotel files not found at {full_path}")
        else:
            fs = fsspec.filesystem('s3')
            objects = fs.ls(path)
            if not objects:
               raise FileNotFoundError(f"Snotel files not found on S3: {path}")
    
        filenames = [obj.split('/')[-1] for obj in objects if '/' in obj]
        
        # Filter out empty strings
        snotel_filenames = [f for f in filenames if f]

        return snotel_filenames, fs

    @staticmethod
    def parse_snotel_filenames(filenames):
        """
        Parse latitude and longitude from SNOTEL filenames and create a GeoDataFrame.
        
        Parameters
        ----------
        filenames : list
            List of SNOTEL filenames to parse
        
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with columns for station_id, latitude, longitude, filename,
            and geometry (Point objects)
        """
        data = []
        
        for filename in filenames:
            # Skip if not a CSV file
            if not filename.endswith('.csv'):
                continue
                
            # Use regex to extract information from the filename
            # Only works with files created by the pre-processor script
            match = re.search(r'(\d+)_LAT_([\d.-]+)_LON_([\d.-]+)\.csv', filename)
            
            if match:
                station_id = match.group(1)
                latitude = float(match.group(2))
                longitude = float(match.group(3))
                
                data.append({
                    'station_id': station_id,
                    'latitude': latitude,
                    'longitude': longitude,
                    'filename': filename,
                    'geometry': Point(longitude, latitude)
                })
        
        # Convert to GeoDataFrame for spatial operations
        stations_gdf = gpd.GeoDataFrame(data, geometry='geometry')
        
        # Converts to EPSG:4326 for the coordinates
        stations_gdf.crs = "EPSG:4326"
        
        return stations_gdf
    
    @staticmethod
    def load_snotel_data(stations_in_basin, date, fs, s3_mount_point, snotel_s3_path):
        """
        Load SNOTEL SWE data for stations within the basin for a specific date.
        Optimized for loading a single timestep. 
        
        Parameters
        ----------
        stations_in_basin : geopandas.GeoDataFrame
            GeoDataFrame of SNOTEL stations that are within the basin
        date : str
            Date string in format 'YYYY-MM-DD'
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with station information and SWE values for the specified date
        """
        if stations_in_basin.empty:
            return pd.DataFrame()
        
        # Initialize a list to store data
        snotel_data_list = []

        path = snotel_s3_path

        if 'local' in fs.protocol:
            base_path = f"{s3_mount_point}/{path}"
        elif 's3' in fs.protocol:
            base_path = f"s3://{path}"

        # Process each station in the basin
        for _, station in stations_in_basin.iterrows():
            filename = station['filename']
            s3_path = f"{base_path}/{filename}"
            try:
                # Open and read the CSV file
                with fs.open(s3_path, 'r') as file:
                    df = pd.read_csv(file)
                    # Convert the target date to datetime
                    target_date = pd.to_datetime(date)
                    
                    # Filter for rows where the date matches the target date
                    df['date'] = pd.to_datetime(df['date'])
                    df_filtered = df[df['date'].dt.date == target_date.date()]
                    
                    if not df_filtered.empty:
                        # Get the SWE value for this date
                        swe_value = df_filtered['snotel_swe'].iloc[0]
                        
                        # Create a record with station info and SWE value
                        snotel_data_list.append({
                            'station_id': station['station_id'],
                            'latitude': station['latitude'],
                            'longitude': station['longitude'],
                            'swe': swe_value
                        })
                    else:
                        print(f"No data found for station {station['station_id']} on {date}")
            except Exception as e:
                print(f"Error loading SNOTEL data for station {station['station_id']}: {e}")
        
        # Convert list to DataFrame
        snotel_df = pd.DataFrame(snotel_data_list)       
        return snotel_df
        
    @staticmethod
    def get_snotel_timeseries(basin_geometry, times, stations_in_basin, fs, s3_mount_point, snotel_s3_path):
        """
        Get time series data for SNOTEL stations within a basin.
        Optimized for loading multiple timesteps.
        
        Parameters
        ----------
        basin_geometry : shapely.geometry
            Basin geometry to use for filtering stations
        times : numpy.ndarray
            Array of datetime objects representing the time points to extract
        stations_in_basin : geopandas.GeoDataFrame
            GeoDataFrame of SNOTEL stations that are within the basin
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with station information and SWE values for each timestamp
        """
        
        if stations_in_basin.empty:
            print("No SNOTEL stations found within the basin.")
            return pd.DataFrame()
        
        start_date = pd.Timestamp(min(times))
        end_date = pd.Timestamp(max(times))

        # Initialize a list to store all station data
        all_station_data = []
        
        path = snotel_s3_path

        if 'local' in fs.protocol:
            base_path = f"{s3_mount_point}/{path}"
        elif 's3' in fs.protocol:
            base_path = f"s3://{path}"
        
        # Process each station in the basin
        for _, station in stations_in_basin.iterrows():
            filename = station['filename']
            s3_path = f"{base_path}/{filename}"
            try:
                # Open and read the CSV file
                with fs.open(s3_path, 'r') as file:
                    df = pd.read_csv(file)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Filter for dates within time range
                    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
                    
                    if not df_filtered.empty:
                        # Add station info to each row, then append to the list
                        df_filtered.loc[:, 'station_id'] = station['station_id']
                        df_filtered.loc[:, 'latitude'] = station['latitude']
                        df_filtered.loc[:, 'longitude'] = station['longitude']
                        all_station_data.append(df_filtered)
                    else:
                        print(f"No data found for station {station['station_id']} in date range")
            except Exception as e:
                print(f"Error loading SNOTEL data for station {station['station_id']}: {e}")
        
        # Combine all station data
        if all_station_data:
            snotel_df = pd.concat(all_station_data)
            return snotel_df
        else:
            return pd.DataFrame()
       
    @staticmethod
    def extract_snotel_timeseries(snotel_df, times):
        """
        Extract SNOTEL SWE values for specified dates from all stations.
        Assumes data is pre-processed to contain only 06z measurements.
        
        Parameters
        ----------
        snotel_df : pandas.DataFrame
            DataFrame with SNOTEL data for stations within the basin
        times : numpy.ndarray
            Array of datetime objects representing the time points to extract
        
        Returns
        -------
        dict
            Dictionary with station_id as keys and arrays of SWE values as values
        """
        if snotel_df.empty:
            return {}
        
        # Convert times to dates once and create lookup dictionary
        time_dates = [pd.to_datetime(t).date() if not isinstance(t, pd.Timestamp) 
                      else t.date() for t in times]
        time_to_idx = {date: idx for idx, date in enumerate(time_dates)}
        
        # Add date column
        snotel_df['date_only'] = snotel_df['date'].dt.date
        
        # Get unique station IDs
        station_ids = snotel_df['station_id'].unique()
        
        # Initialize a dictionary to store results
        snotel_data = {}
        
        for station_id in station_ids:
            # Filter data for this station
            station_df = snotel_df[snotel_df['station_id'] == station_id]
            
            # Create a time series for this station
            swe_values = np.full(len(times), np.nan)
            
            # Create date->SWE lookup dictionary
            # If multiple measurements per day, take the first one
            date_swe_dict = {}
            for _, row in station_df.iterrows():
                date = row['date_only']
                if date not in date_swe_dict:
                    date_swe_dict[date] = row['snotel_swe']
            
            # Fill SWE values using lookup dictionaries
            for date, idx in time_to_idx.items():
                if date in date_swe_dict:
                    swe_values[idx] = date_swe_dict[date]
            
            # Only add stations that have at least some valid data
            if not np.isnan(swe_values).all():
                snotel_data[station_id] = {
                    'swe': swe_values,
                    'latitude': station_df['latitude'].iloc[0],
                    'longitude': station_df['longitude'].iloc[0]
                }
            else:
                print(f"No measurements found for station {station_id} - excluding from results")
        
        return snotel_data


class SnotelCalculator:
    @staticmethod
    def find_stations_in_basin(stations_gdf, basin_geometry):
        """
        Find SNOTEL stations that fall within the basin geometry.
        
        Parameters
        ----------
        stations_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing SNOTEL station information
        basin_geometry : shapely.geometry
            Basin geometry to use for filtering stations
            
        Returns
        -------
        geopandas.GeoDataFrame
            Filtered GeoDataFrame containing only stations within the basin
        """
        # Ensure CRS match between stations and basin geometry
        if hasattr(basin_geometry, 'crs') and basin_geometry.crs != stations_gdf.crs:
            stations_gdf = stations_gdf.to_crs(basin_geometry.crs)
        
        # Filter stations within the basin
        stations_in_basin = stations_gdf[stations_gdf.intersects(basin_geometry)]

        if not stations_in_basin['station_id'].empty:
            station_return = []
            for stations in stations_in_basin['station_id']:
                station_return.append(stations)
            print(f"{len(station_return)} SNOTEL stations found in basin: {station_return}")
        
        return stations_in_basin

class SnotelPlotter:
    @staticmethod
    def add_snotel_overlay(ax, snotel_data, proj):
        """
        Add SNOTEL SWE data as text overlays on a map.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to add overlay to
        snotel_data : pandas.DataFrame
            DataFrame with station information and SWE values
        proj : cartopy.crs
            Projection to use
            
        Returns
        -------
        matplotlib.axes.Axes
            Updated axes with SNOTEL overlay
        """
        if snotel_data.empty:
            return ax

        color='#990000'

        if not snotel_data.empty:
            # Plot all stations
            ax.plot(
                snotel_data['longitude'], 
                snotel_data['latitude'], 
                'o',
                markersize=3,
                transform=proj,
                color=color,
                label='SNOTEL Stations (SWE)'
            )
            
            # Add text labels iteratively
            for _, station in snotel_data.iterrows():
                swe_value = f"{station['swe']:.2f}"
                ax.text(
                    station['longitude'] + 0.0005,
                    station['latitude'] - 0.0005,
                    swe_value,
                    fontsize=11,
                    ha='left',
                    va='top',
                    transform=proj,
                    fontweight='bold',
                    color=color
                )
            # Add the legend to the plot, using the specified label
            ax.legend(loc='upper right', fontsize=10, framealpha=0.5, bbox_to_anchor=(1.25, 1.05))
        
        return ax

