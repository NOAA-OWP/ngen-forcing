from datetime import datetime
import csv

import isodate
import os.path

from gov.noaa.nwc.cwmstools import CWMSDownloader


def main():
    csv_data = read_input_csv("/home/dwj/PycharmProjects/CWMS_eval_1/Lakes_upstream_of_A2W_sites_with_NIDID_323_Final.csv")
    downloader = CWMSDownloader()

    output_data = []

    for row in csv_data:

        output_path = "output/" + row["NIDID"]+".csv"

        # dont process gages that allready have stored data
        if os.path.exists(output_path) and os.path.isfile(output_path):
            continue

        # download data about one gage
        (time1, data1) = get_flows_and_times(downloader, row["Office"], row["gage"], "P-70y")

        # calculate daily averages for the current gage's data
        output_data = find_daily_values(time1, data1)

        # write an output csv with data on this gage
        write_output_csv(output_path, output_data)

    return

def find_daily_values(times, values):
    output_data = []

    if ( len(times) < 1):
        output_data.append({"date" : "N/A", "average-flow" : "N/A",
                            "quality" : "N/A"})
    elif ( len(times) == 1):
        output_data.append({"date": times[0], "average-flow": values[0],
                            "quality": 100})
    else:
        current_date = times[0].date()
        current_sum = values[0]
        current_count = 1
        for i in range(0, len(times)):
            if ( times[i].date() == current_date):
                current_sum += values[i]
                current_count += 1
            else:
                output_data.append({"date" : current_date,
                                   "average-flow" : float(current_sum) / current_count,
                                   "quality" : 100})
                current_date = times[i].date()
                current_sum = values[i]
                current_count = 1

    return output_data


def write_output_csv(name, csv_data):
    with open(name, mode='w') as csv_file:
        fieldnames = ["date", "average-flow", "quality"]

        # setup a csv reader with the list of field names
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # write the file
        writer.writerows(csv_data)


def read_input_csv(name):
    csv_data = []

    with open(name) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=",")
        for row in reader:
            csv_data.append(row)
    return csv_data


def get_flows_and_times(downloader, office, name, start):

    stop = False
    count = 0
    data = {}

    while not stop and count < 3:
        try:
            data = downloader.get_data(office, name, start, "json")
            stop = True
        except:
            count += 1

    times = []
    flows = []
    try:
        for part1 in data['time-series']['time-series']:
            if 'irregular-interval-values' in part1.keys():
                for part2 in part1['irregular-interval-values']['values']:
                    times.append(isodate.parse_datetime(part2[0]))
                    flows.append(part2[1])
            elif 'regular-interval-values' in part1.keys():
                for segment in part1['regular-interval-values']['segments']:
                    # print (segment.keys())
                    first_time_str = segment['first-time']
                    last_time_str = segment['last-time']

                    first_time = isodate.parse_datetime(first_time_str)
                    last_time = isodate.parse_datetime(last_time_str)

                    values = segment['values']
                    time_step = (last_time - first_time) / (len(values) - 1)

                    for i in range(0, len(values)):
                        flows.append(values[i][0])
                        times.append(first_time + (i * time_step))

                    # print(first_time, last_time, time_step)

            else:
                print(part1)
    except:
        print("Error no data from office="+office+" name="+name)

    # print(times)
    # print(flows)

    return times, flows


if __name__ == "__main__":
    main()
