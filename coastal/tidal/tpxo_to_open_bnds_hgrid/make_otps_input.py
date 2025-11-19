#! /usr/bin/env python

import os, sys, time, urllib, getopt
import logging
import argparse
from datetime import datetime, timedelta
import numpy as np
import netCDF4

schism_coord_name = "nodeCoords"
schism_open_boundary_name = "openBndNodes"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('schism_grid', type=str, help='.nc file containing schism coordinates')
    parser.add_argument('start_time', type=str, help='start time of the timeseries, YYYYmmddHH')
    parser.add_argument('end_time', type=str, help='end time of the timeseries, YYYYmmddHH')
    parser.add_argument('time_step', type=str, help='time step in seconds')
    parser.add_argument('output', type=str, help='OTPSnc inputfile')

    args = parser.parse_args()

    start_time = datetime.strptime(f'{args.start_time}0000', "%Y%m%d%H%M%S")
    end_time = datetime.strptime(f'{args.end_time}0000', "%Y%m%d%H%M%S")
    time_step = timedelta(seconds=int(args.time_step))

    with netCDF4.Dataset(args.schism_grid) as f_in:
        coords = f_in[schism_coord_name][:]
        valid_indices = f_in[schism_open_boundary_name][:]
        coords = [coords[i].tolist() for i in valid_indices]

    with open(args.output, 'w') as fout:
        for c in coords:
            current = start_time
            while current <= end_time:
              fout.write( f"{c[1]}  {c[0]}  {current.strftime('%Y %m %d %H %M %S')}\n" )
              current += time_step

t0 = time.time()

logging.basicConfig(format=\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info( "System Path: " + str( sys.path ) )

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logger.error("Failed to get program options.", exc_info=True)

