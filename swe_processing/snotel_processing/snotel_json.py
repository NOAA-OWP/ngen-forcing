import json
import argparse
import pandas as pd
import os
import time

class DataLoader:
    @staticmethod
    def load_json_file(filepath):
        """
        Load data from a JSON file.
        
        Parameters:
        -----------
        filepath : str
            Path to the JSON file to load
            
        Returns:
        --------
        list
            A list of dictionaries
        """
        
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)
            return data
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error: File '{filepath}' contains invalid JSON.")
            return None
        except Exception as e:
            print(f"Error loading JSON file: {str(e)}")
            return None
    
    @staticmethod
    def load_json_directory(directory_path):
        """
        Load all JSON files from a directory.
        
        Parameters:
        -----------
        directory_path : str
            Path to the directory containing JSON files
            
        Returns:
        --------
        dict
            Dictionary mapping filenames to loaded JSON data
        """
        if not os.path.isdir(directory_path):
            print(f"Error: Directory '{directory_path}' not found.")
            return []
            
        json_files = []
        
        # List all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                filepath = os.path.join(directory_path, filename)
                json_files.append(filepath)
        
        return json_files


class DataParser:
    @staticmethod
    def parse_swe_station(swe_data):
        """
        Extract the station triplet from SWE data without parsing all values.
        
        Parameters:
        -----------
        swe_data : list
            List containing SWE data
            
        Returns:
        --------
        str
            Station triplet identifier
        """
        if not swe_data or len(swe_data) == 0:
            return None
        
        swe_station = swe_data[0]['stationTriplet']

        return swe_station

    @staticmethod
    def parse_station_data(data):
        # Extract data and create DataFrame
        df = pd.DataFrame([
            {
                'triplet': item['stationTriplet'],
                'id': item['stationId'],
                'lat': item['latitude'],
                'lon': item['longitude'],
                'offset': item['dataTimeZone']
            }
            for item in data
        ])

        return df

    @staticmethod
    def parse_swe_data(swe_data, swe_station, local_hour, day_offset):
        """
        Parse SWE (Snow Water Equivalent) data from JSON format.
        
        Parameters:
        -----------
        swe_data : list
            List containing SWE data for a station
        swe_station : str
            Station triplet identifier
        local_hour : int
            The local time that corresponds to 06z
        day_offset: int
            Offset if 06z corresponds to a different local-time day
        Returns:
        --------
        pandas.DataFrame
            DataFrame with columns for triplet, date, and value, filtered for the specified UTC hour
        """
        # Extract all values from swe_data and create a dataframe
        values = swe_data[0]['data'][0]['values']
        df = pd.DataFrame(values)
        
        # If no entries, return empty DataFrame and error
        if df.empty:
            print("Empty swe_data returned to dataframe.")
            return pd.DataFrame(columns=['triplet', 'date', 'snotel_swe'])
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for local hour using vectorized operation
        df = df[df['date'].dt.hour == local_hour]
        
        # If no matching timesteps, return empty DataFrame and error
        if df.empty:
            print("No matching time entries in DataFrame")
            return pd.DataFrame(columns=['triplet', 'date', 'snotel_swe'])
        
        # Create new date column for 06z
        date_only = pd.to_datetime(df['date'].dt.date)
        df['utc_date'] = date_only - pd.Timedelta(days=day_offset)
        df['date'] = pd.to_datetime(df['utc_date']) + pd.Timedelta(hours=6)
        
        # Construct the final DataFrame
        swe_df = pd.DataFrame({
            'triplet': swe_station,
            'date': df['date'],
            'snotel_swe': df['value'] * 0.0254  # Convert inches to meters
        })
        
        return swe_df

    @staticmethod
    def merge_data(station_df, swe_df):
        """
        Merge station metadata with SWE time series data.
        
        Parameters:
        -----------
        station_df : pandas.DataFrame
            DataFrame containing station metadata including triplet, lat, lon, offset
        swe_df : pandas.DataFrame
            DataFrame containing SWE measurements with triplet, date, swe
            
        Returns:
        --------
        pandas.DataFrame
            Combined DataFrame with all necessary information
        """
        # Merge on the 'triplet' column which exists in both dataframes
        merged_df = pd.merge(
            swe_df,
            station_df[['triplet', 'lat', 'lon', 'offset']],  # Select only the columns we need
            on='triplet',
            how='left'  # Keep all SWE measurements, even if station not found (though this shouldn't happen)
        )
        
        # Reorder columns for clarity
        column_order = ['date', 'snotel_swe', 'triplet', 'lat', 'lon', 'offset']
        merged_df = merged_df[column_order]
        
        return merged_df

    @staticmethod
    def get_offset(swe_station, station_df):
        '''
        Extract the offset from UTC value from the station dataframe, using the station triplet 
        from the swe data json file to index.

        Parameters:
        -----------
        swe_station : string
            string that contains the stationTriplet identifier from the swe data
        station_df : pandas.DataFrame
            DataFrame containing station metadata including triplet, lat, lon, offset
        
        Returns:
        --------
        numpy.int64
            An integer value describing the offset from UTC
        '''
        station_offset = station_df[station_df['triplet'] == swe_station]['offset'].iloc[0]

        return station_offset
    
    @staticmethod
    def get_local_time_for_utc(utc_hour, offset):
        """
        Calculate the local time equivalent of a specific UTC hour.
        
        Parameters:
        -----------
        utc_hour : int
            The hour in UTC (0-23) - hardcoded in 
        offset : int
            Timezone offset in hours
            
        Returns:
        --------
        tuple
            (local_hour, day_offset) where:
            - local_hour is the hour in local time (0-23)
            - day_offset is 0 for same day, -1 for previous day, 1 for next day
        """
        # Calculate local hour
        local_hour = (utc_hour + offset) % 24
        
        # Calculate day offset
        day_offset = 0
        if utc_hour + offset < 0:
            day_offset = -1
        elif utc_hour + offset >= 24:
            day_offset = 1

        return local_hour, day_offset

class CSVWriter:
    @staticmethod
    def write_to_csv(merged_df, output_dir):
        """
        Write the merged DataFrame to a CSV file with filename encoding metadata.
        
        Args:
            merged_df (pandas.DataFrame): DataFrame containing date, swe, triplet, lat, lon
            output_dir (str): Directory path where the CSV file will be saved
        
        Returns:
            str: Path to the created CSV file
        """
        # Extract metadata from first row (assuming all rows have same station data)
        first_row = merged_df.iloc[0]
        triplet = first_row['triplet']
        lat = first_row['lat']
        lon = first_row['lon']
        
        # Parse triplet to get station info
        triplet_parts = triplet.split(':')
        station_id = triplet_parts[0]
        
        # Create filename with metadata
        filename = f"{station_id}_LAT_{lat}_LON_{lon}.csv"
        filepath = f"{output_dir}/{filename}"
        
        # Write only date and swe columns to CSV
        merged_df[['date', 'snotel_swe']].to_csv(filepath, index=False)
        
        return filepath


class JSONProcessor:
    def __init__(self, station_filepath=None, swe_filepath=None, output_dir=None):
        self.station_filepath = station_filepath
        self.swe_filepath = swe_filepath
        self.output_dir = output_dir
        self.utc_hour = 6

        self.station_data = None 
        self.station_df = None
        
        self.swe_filelist = None
        self.swe_data = None
        self.swe_df = None
        self.offset = None
        self.local_hour = None
        self.day_offset = None
        self.swe_station = None

        self.merged_df = None
        self.csv_path = None

    def process(self):
        
        t_start = time.time()

        self.station_data = DataLoader.load_json_file(self.station_filepath)
        self.station_df = DataParser.parse_station_data(self.station_data)
        self.swe_filelist = DataLoader.load_json_directory(self.swe_filepath)
        for file in self.swe_filelist:
            self.swe_data = DataLoader.load_json_file(file)
            self.swe_station = DataParser.parse_swe_station(self.swe_data)
            self.offset = DataParser.get_offset(self.swe_station, self.station_df)
            (self.local_hour, 
             self.day_offset) = DataParser.get_local_time_for_utc(self.utc_hour,
                                                                  self.offset)
            self.swe_df = DataParser.parse_swe_data(self.swe_data, 
                                                    self.swe_station,
                                                    self.local_hour, 
                                                    self.day_offset)
            self.merged_df = DataParser.merge_data(self.station_df, self.swe_df)
            self.csv_path = CSVWriter.write_to_csv(self.merged_df, self.output_dir)
            print(f"Wrote csv to: {self.csv_path}")
            
        t_end = time.time()

        total_time = t_end-t_start

        print(f"Total processing time: {total_time:.4f}s")

def get_options(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('station_filepath', type=str, 
                       help="Path that points to station metadata json file.")
    parser.add_argument('swe_filepath', type=str,
                       help="Path that points to SNOTEL swe data json files.")
    parser.add_argument('output_dir', type=str,
                       help="Path to write csv output.")

    if args_list is not None:
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()

def execute(args):
    """
    Execute the JSON processor.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """ 
    # Create and run the JSON processor
    processor = JSONProcessor(
        station_filepath=args.station_filepath,
        swe_filepath=args.swe_filepath,
        output_dir=args.output_dir
    )
    processor.process()

def proc_snotel(args_list=None):
    args = get_options(args_list)
    execute(args)

if __name__ == '__main__':
    proc_snotel()
