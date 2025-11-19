import os
import json
import requests
from urllib.parse import urljoin
import argparse

# SNOTEL REST API root
base_url = "https://wcc.sc.egov.usda.gov/awdbRestApi/"

# Data endpoint
data_endpoint = "services/v1/data"

# Station Metadata endpoint
stations_endpoint = "services/v1/stations"

# Data and Station Metadata URLs
data_url = urljoin(base_url, data_endpoint)
stations_url = urljoin(base_url, stations_endpoint)

# HTTP request timeout in seconds (5 minutes)
request_timeout = 300

# Send HTTP request to the Data REST endpoint. Requests hourly SWE (WTEQ) data for the given
# station and begin/end times.
# Returns the HTTP response
def make_data_request(station, begin_date, end_date):
    query_params = {'stationTriplets': station["stationTriplet"],
                    'elements': 'WTEQ',
                    'duration': 'HOURLY',
                    'beginDate': begin_date,
                    'endDate': end_date,
                    'periodRef': 'END',
                    'centralTendencyType': 'NONE',
                    'returnFlags': 'false',
                    'returnOriginalValues': 'false',
                    'returnSuspectData': 'false'
    }
    r = requests.get(url=data_url, params=query_params, timeout=request_timeout)

    return r

# Send HTTP request to the Station Metadata REST endpoint. Requests station metadata for all SNOTEL
# stations that produce hourly SWE (WTEQ) data for both active and non-active stations. 
# Returns the HTTP response
def make_station_metadata_request():
    query_params = {'stationTriplets': '*:*:SNTL',
                    'elements': 'WTEQ',
                    'durations': 'HOURLY',
                    'returnForecastPointMetadata': 'false',
                    'returnReservoirMetadata': 'false',
                    'returnStationElements': 'false',
                    'activeOnly': 'false'
    }
    r = requests.get(url=stations_url, params=query_params, timeout=request_timeout)

    return r

# Reads a list of SNOTEL stations from a JSON file and returns the mapped python object. The format
# needs to be a list of objects with the "stationTriplet" name/value pair element as shown in the 
# example below.
# [
#   {
#     "stationTriplet": "301:CA:SNTL"
#   }
# ]     
def get_stations_from_file(file):
    with open(file, "r") as f:
        stations = json.load(f)
    
    return stations

# Requests SNOTEL stations from the Station Metadata REST endpoint and returns the mapped python
# object.
def get_stations_from_endpoint():
    # If any error occurs here, just let the exception be thrown
    response = make_station_metadata_request()
    response.raise_for_status()
    json_data = response.json()
    
    return json_data

# Performs command line argument parsing 
def get_options():
    parser = argparse.ArgumentParser(description="Downloads hourly SNOTEL Snow Water Equivalent (SWE) observation data from the National Water and Climte Center (NWCC) \
                                     of the U.S. Department of Agriculture (USDA). NWCC's REST webservice is used to download the data. Data files are JSON. \
                                     If no begin or end dates are specified, the default period of 2009-12-01 to the current date is used because this aligns with the unmasked SNODAS dataset period. \
                                     Data is retrieved for all active SNOTEL stations that provide SWE data.")
    parser.add_argument("output_dir", help="Path to directory where the data files will be written")
    parser.add_argument("--station_file", help="Path to file that contains a list of SNOTEL stations for obtaining SWE data. Stations will be retrieved using the NWCC webservice if this option is not used.")
    parser.add_argument("--begin_date", default="2009-12-01", help="Begin date of requested data. Specify as YYYY-MM-DD (e.g. 2025-02-25)")
    parser.add_argument("--end_date", default="0", help="End date of requested data. Specify as YYYY-MM-DD (e.g. 2025-02-25) or 0 for the current date")

    args = parser.parse_args()

    return args
    

def main():
    args = get_options()
    station_file = args.station_file
    output_dir = args.output_dir
    begin_date = args.begin_date
    end_date = args.end_date

    if station_file:
        print(f"Retrieving list of stations from file {station_file}")
        stations = get_stations_from_file(station_file)
    else:
        print(f"Retrieving list of stations from endpoint {stations_url}")
        stations = get_stations_from_endpoint()

    num_stations = len(stations)
    print(f"Number of stations: {num_stations}")
    station_count = 0

    for station in stations:
        station_id = station["stationTriplet"]
        print()
        try:
            print(f"Retrieving SWE data for {station_id}")
            response = make_data_request(station, begin_date, end_date)
        except Exception as e:
            print(f"ERROR: Unable to retrieve data ... skipping station\n{e}")
            continue

        if not response.ok:
            print(f"ERROR: HTTP response status code has unexpected value: {response.status_code} ... skippking station")
            print(f"URL: {response.url}")
            continue
        
        try:
            json_data = response.json()
        except Exception as e:
            print(f"ERROR: Unable to decode JSON in HTTP response ... skipping station\n{e}")
            continue

        file_name = f"{station_id.replace(':', '-')}_snotel_hourly_SWE.json"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"Created file {file_path}")
        station_count+=1

    print(f"\nDownload Complete! Retrieved data for {station_count} out {num_stations} stations")

if __name__ == "__main__":
    main()
