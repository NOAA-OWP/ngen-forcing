#!/usr/bin/env python
import os
import datetime

import csv
import json

from warnings import warn
from gov.noaa.nwc.cwmstools import CWMSDownloader

import argparse

"""
This program downloads stream flow data from the Army Corp of Engineers (ACE) CWMS  web service.  

usage:
   python CWMS_download_current --file_format=<format> site_file output_directory

format is one of "json", "xml", or "csv" 
"""

def main():
    print( datetime.datetime.now(), end = " --- " )
    print(  'Entering CWMS_download_current.py ... ' )
    #parse the command line
    parser = argparse.ArgumentParser()

    parser.add_argument("site_file",help="Csv file that contains the gage ids, office codes, and NEDID codes of the gages to download")
    parser.add_argument("output_dir",help="The directory to store downloaded files")
    parser.add_argument("-f","--file_format",help="The format to store downloaded data [json,xml,csv]",default=json)

    args = parser.parse_args()

    output_path = args.output_dir
    csv_data = read_input_csv(args.site_file)
    output_format = args.file_format

    downloader = CWMSDownloader()

    for row in csv_data:

        if output_format == "json":
            json_data = get_data(downloader, row["office"], row["gage"], "PT-48h", "json")
            with open(output_path+"/"+row["usace_gage_id"]+".json","w") as outfile:
                json.dump(json_data,outfile,indent=2)
                print( datetime.datetime.now(), end = " --- " )
                print(  'Successfully downloaded ' + \
                                output_path+"/"+row["usace_gage_id"]+".json" + "!" )
        elif output_format == "xml":
            xml_data = get_data(downloader, row["office"], row["gage"], "PT-48h", "xml")
            with open(output_path+"/"+row["usace_gage_id"]+".xml", "w") as outfile:
                try:
                   outfile.write(xml_data)
                except Exception as e:
                   print( datetime.datetime.now(), end = " --- " )
                   print(  'WARNING: Failed writting ' + \
                                output_path+"/"+row["usace_gage_id"]+".xml" + "!" )
                else:
                   print( datetime.datetime.now(), end = " --- " )
                   print(  'Successfully downloaded ' + \
                                output_path+"/"+row["usace_gage_id"]+".xml" + "!" )



        elif output_format == "csv":
            csv_data = get_data(downloader, row["office"], row["gage"], "PT-48h", "csv")
            with open(output_path+"/"+row["usace_gage_id"]+".csv", "w") as outfile:
                outfile.write(csv_data)
                print( datetime.datetime.now(), end = " --- " )
                print(  'Successfully downloaded ' + \
                                output_path+"/"+row["usace_gage_id"]+".csv" + "!" )
        else:
            print( datetime.datetime.now(), end = " --- " )
            warn("Unexpected output format",RuntimeWarning)

#    with open( output_path + '/fetch_last_success', 'a'):
#             os.utime( output_path + '/fetch_last_success', None )
    print( datetime.datetime.now(), end = " --- " )
    print(  'Leaving CWMS_download_current.py ... ' )

def read_input_csv(name):
    csv_data = []

    with open(name) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        for row in reader:
            csv_data.append(row)
    return csv_data


def get_data(downloader, office, name, start, data_format):

    stop = False
    count = 0
    data = {}

    while not stop and count < 3:
        try:
            data = downloader.get_data(office, name, start, data_format)
            stop = True
        except:
            count += 1

    return data


if __name__ == "__main__":
    main()
