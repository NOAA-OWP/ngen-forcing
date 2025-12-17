#!/usr/bin/env python
# from datetime import datetime

import csv
import json

from warnings import warn
from gov.noaa.nwc.cwmstools import CWMSDownloader

import argparse

"""
This program downloads stream flow data from the Army Corp of Engineers (ACE) CWMS  web service.  

usage:
   python CWMS_download_current --file_format=<format> site_file output_directory start_date stop_date

format is one of "json", "xml", or "csv" 

start_date and stop_date must be formated YYYYMMDD
"""


def main():
    #parse the command line
    parser = argparse.ArgumentParser()

    parser.add_argument("site_file",help="Csv file that contains the gage ids, office codes, and NEDID codes of the gages to download")
    parser.add_argument("output_dir",help="The directory to store downloaded files")
    parser.add_argument("begin", help="The first date in the requested date range")
    parser.add_argument("end", help="The last date in the request date range")
    parser.add_argument("-f","--file_format",help="The format to store downloaded data [json,xml,csv]",default=json)

    args = parser.parse_args()

    output_path = args.output_dir
    csv_data = read_input_csv(args.site_file)
    output_format = args.file_format
    begin = args.begin
    end = args.end

    #output_path = "/home/dwj/PycharmProjects/CWMS_eval_1/output/"
    #csv_data = read_input_csv("/media/sf_Projects/Lakes_upstream_of_A2W_sites_323_total_Final.csv")
    #output_format = "xml"

    downloader = CWMSDownloader()

    for row in csv_data:

        if output_format == "json":
            json_data = get_data(downloader, row["office"], row["gage"], begin, end, "json")
            with open(output_path+row["usace_gage_id"]+".json","w") as outfile:
                json.dump(json_data,outfile)
        elif output_format == "xml":
            xml_data = get_data(downloader, row["office"], row["gage"], begin, end, "xml")
            with open(output_path+row["usace_gage_id"]+".xml", "w") as outfile:
                outfile.write(xml_data)
        elif output_format == "csv":
            csv_data = get_data(downloader, row["office"], row["gage"], begin, end, "csv")
            with open(output_path+row["usace_gage_id"]+".csv", "w") as outfile:
                outfile.write(csv_data)
        else:
            warn("Unexpected output format",RuntimeWarning)




def read_input_csv(name):
    csv_data = []

    with open(name) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        for row in reader:
            csv_data.append(row)
    return csv_data


def get_data(downloader, office, name, begin, end, data_format):

    stop = False
    count = 0
    data = {}

    while not stop and count < 3:
        try:
            data = downloader.get_data_range(office, name, begin, end, data_format)
            stop = True
        except:
            count += 1

    return data


if __name__ == "__main__":
    main()
