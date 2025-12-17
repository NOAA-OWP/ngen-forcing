#! /usr/bin/env python

import os, sys, time, urllib, getopt
import logging
import argparse
import math
from datetime import datetime
import numpy as np
import netCDF4
from TPXOOut import TPXOOut, isclose

schism_coord_name = "nodeCoords"
schism_open_boundary_name = "openBndNodes"
time_step = 3600
missing = -9999.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('otpsnc_output', type=str, help='Input waterlevel timeseries file')
    parser.add_argument('schism_grid', type=str, help='.nc file containing schism coordinates')
    parser.add_argument('output', type=str, help='Output .nc file')

    args = parser.parse_args()

    tpxo=TPXOOut( args.otpsnc_output )
#   tpxo.print()

    with netCDF4.Dataset(args.schism_grid) as f_in:
        coords = f_in[schism_coord_name][:]
        valid_indices = f_in[schism_open_boundary_name][:]
        coords = [coords[i].tolist() for i in valid_indices]

    #get the start and end time of the time series
    #assume all locations have the same start/end time
    df_selected = tpxo.df[ tpxo.df['Lat'].between( coords[0][1] - 0.001, coords[0][1] + 0.001 ) & \
                  tpxo.df['Lon'].between( coords[0][0] -0.001, coords[0][0] + 0.001) ]
    start = datetime.strptime( f'{tpxo.df["mm.dd.yyyy"].iloc[0]} {tpxo.df["hh:mm:ss"].iloc[0]}', "%m.%d.%Y %H:%M:%S")
    end = datetime.strptime( f'{tpxo.df["mm.dd.yyyy"].iloc[-1]} {tpxo.df["hh:mm:ss"].iloc[-1]}', "%m.%d.%Y %H:%M:%S")
    nsteps = math.floor( (end -start).total_seconds() / 3600 ) + 1

    #create the output file
    with netCDF4.Dataset(args.output, "w", format="NETCDF4") as f_out:
        f_out.createDimension("time", None)
        f_out.createDimension("nOpenBndNodes", len(coords))
        f_out.createDimension("nLevels", 1)
        f_out.createDimension("nComponents", 1)
        f_out.createDimension("one", 1)

        time_step_var = f_out.createVariable("time_step", "f8", ("one",))
        time_var = f_out.createVariable("time", "f8", ("time",))
        time_series_var = f_out.createVariable("time_series", "f8", ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                                                   fill_value=missing, zlib=True)

        time_var.long_name = "model time" ;
        time_var.standard_name = "time" ;
        time_var.units = f'seconds since {start.strftime("%Y-%m-%d %H:%M:%S")}        ! NCDASE - BASE_DAT' ;
        time_var.base_date = f'{start.strftime("%Y-%m-%d %H:%M:%S")}        ! NCDASE - BASE_DATE' ;
        time_var.start_time = 0. ;

        time_step_var[:] = np.array([time_step])
        #time_var[:] = np.arange(0, len(df_selected)*time_step, time_step)
        time_var[:] = np.arange(0, nsteps*time_step, time_step)

        for c in range(0, len(coords)):
           df_selected = tpxo.df[ tpxo.df['Lat'].between( coords[c][1] - 0.0001, coords[c][1] + 0.0001 ) & \
                  tpxo.df['Lon'].between( coords[c][0] -0.0001, coords[c][0] + 0.0001) ]
           if not df_selected.empty:
               data = df_selected["z(m)"]
               data = np.where(data > missing, data, 0)
               time_series_var[:, c, 0, 0] = data[0:nsteps]
           else:
               time_series_var[:, c, 0, 0] = 0

t0 = time.time()

logging.basicConfig(format=\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                level=logging.INFO)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(\
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#logger.setFormatter(formatter)
logger.info( "System Path: " + str( sys.path ) )

if __name__ == "__main__":
   try:
      main()
   except Exception as e:
      logger.error("Failed to get program options.", exc_info=True)

