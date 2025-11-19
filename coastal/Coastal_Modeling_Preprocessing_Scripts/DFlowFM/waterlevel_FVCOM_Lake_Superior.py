"""
Waterlevel extraction script

Extract waterlevel values from ADCIRC and export in DFlow BC format.

Usage:
python waterlevel.py -o waterlevels.bc fort63.nc boundary.csv
    Extract waterlevel values form fort63.nc for the points in boundary.csv
    and export to waterlevels.bc.
    Note that boundary should already represent the region of interest.
"""

import argparse
import pathlib
import time
import os
import datetime

import netCDF4
import numpy as np
import pandas as pd

from common.io import read_pli
from common.geometry import kd_nearest_neighbor

from itertools import islice
import math
import numpy as np
import pandas as pd
import string
import csv
import operator
import datetime
from textwrap import dedent
from io import StringIO
from tlz import take, drop
from tlz.curried import get

class BCFileWriter:
    functions = {'timeseries', 'astronomic'}

    forcing_template = (
        "[forcing]\n"
        "Name = $name\n"
        "Function = $function\n"
        "Time-interpolation = linear"
    )

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self._filehandle = open(self.filename, 'w')
        return self

    def __exit__(self, type, value, traceback):
        self._filehandle.close()

    def add_forcing(self, name, function, units, Lake, data, vectors=None):
        """Add forcing
        Args:
            name (str): Name
            function (str): One of BCFileWriter.functions
            units (list[tuples]): A list of tuples mapping column name to column units.
                The ordering should match the ordering of the data columns.
            data (Iterable of lists): Number of columns in data and len(units) must match.
                Data will be iterated thru row by row
            vectors (list[str]):
        Returns:
            None
        """
        T = [string.Template(self.forcing_template).substitute(name=name, function=function)]
        if vectors is not None:
            for v in vectors:
                T.append(f"Vector = {v}")

        if(Lake != None):
            if(Lake.upper() == 'ERIE'):
                offset = "173.5"
            elif(Lake.upper() == 'ONTARIO'):
                offset = "74.2"
            elif(Lake.upper() == 'SUPERIOR'):
                offset = "183.2"
            elif(Lake.upper() == 'MICHIGAN-HURON'):
                offset = "176.0"
            T.append(f"Offset = {offset}")

        for i, (col, unit) in enumerate(units):
            T.append(f"Quantity = {col}")
            T.append(f"Unit = {unit}")

        arr = StringIO()
        if isinstance(data, np.ndarray):
            np.savetxt(arr, data, fmt='%f', delimiter=' ')
        else:
            for row in data:
                arr.write(" ".join(map(str, row)))
                arr.write("\n")

        T.append(arr.getvalue())
        self._write_str(T)

    def _write_str(self, lines):
        for L in lines:
            self._filehandle.write(L)
            self._filehandle.write("\n")

def invalid_mask(var, chunksize=2**18):
    """Generate a mask of a masked array for the columns that only have completely valid observations.

    An optional chunksize can be set to avoid reading the entire variable mask array into memory.

    Note that the masked array mask is inverted, ie a True value indicates a missing value.
    The mask returned from this function, a True indicates a column with no missing values.

    Args:
        var (netCDF4.Variable): Variable to consider
        chunksize (int, optional): Number of columns to consider at a time. Defaults to 2**18.

    Returns:
        np.ndarray: Mask of valid columns
    """
    rv = np.empty(var.shape[1], dtype=bool)
    buf = np.empty((var.shape[0], chunksize), dtype='bool')
    for i in range(0, var.shape[1], chunksize):
        print(i)
        rv_view = rv[i:i+chunksize]
        buf_view = buf[:, :rv_view.shape[0]]
        #np.equal(var[:, i:i+chunksize], var._FillValue, out=buf_view)
        buf_view[:] = np.ma.getmaskarray(var[:, i:i+chunksize])
        np.any(buf_view, axis=0, out=rv_view)
        np.logical_not(rv_view, out=rv_view)
    return rv


def main(args):
    print("Reading PLI")
    pli_data = read_pli(args.pli)
    FVCOM_daterange = pd.date_range(start=args.start_time.strftime("%Y-%m-%d %H:%M:%S"),end=args.stop_time.strftime("%Y-%m-%d %H:%M:%S"),freq="6H")
    FVCOM_outfile = "glofs.lsofs.fields.nowcast."
    FVCOM_file = FVCOM_outfile + FVCOM_daterange[0].strftime("%Y%m%d") + '.t' + FVCOM_daterange[0].strftime("%H") + 'z.nc'
    FVCOM_data =  netCDF4.Dataset(os.path.join(args.FVCOM,FVCOM_file), mode='r')
    FVCOM_lons =  np.ma.getdata(FVCOM_data.variables['lon'][:].flatten())
    FVCOM_lats =  np.ma.getdata(FVCOM_data.variables['lat'][:].flatten())
    time_col = FVCOM_data.variables['time']
    ref_time = time_col.units
    units = [('time', 'seconds since ' + args.start_time.strftime("%Y-%m-%d %H:%M:%S")),
             ('waterlevelbnd', 'm')]

    zeta = np.empty((len(FVCOM_daterange)*6,len(pli_data['values'][:,0])),dtype=float)
    time = np.empty(len(FVCOM_daterange)*6,dtype=float)
    count = 0

    for i in range(len(FVCOM_daterange)):
        locfile = FVCOM_outfile + FVCOM_daterange[i].strftime("%Y%m%d") + '.t' + FVCOM_daterange[i].strftime("%H") + 'z.nc'
        ds = netCDF4.Dataset(os.path.join(args.FVCOM,locfile), mode='r')
        FVCOM_lons =  np.ma.getdata(FVCOM_data.variables['lon'][:].flatten())
        FVCOM_lats =  np.ma.getdata(FVCOM_data.variables['lat'][:].flatten())
        zeta_flag = np.reshape(ds.variables['zeta'][:].data,(6,len(FVCOM_lats)))
        FVCOM_lons = np.where(np.sum(zeta_flag,axis=0) < -1000.0,-9999,FVCOM_lons)
        FVCOM_lats = np.where(np.sum(zeta_flag,axis=0) < -1000.0,-9999,FVCOM_lats)
        adpts = np.column_stack([FVCOM_lons, FVCOM_lats])
        _, stations = kd_nearest_neighbor(adpts, pli_data['values'])
        zeta[count:count+6,:] = np.reshape(ds.variables['zeta'],(6,len(FVCOM_lats)))[:,stations]
        time[count:count+6] = ds.variables['time'][:].data
        count += 6
    

    time_final = netCDF4.num2date(time,units=ref_time,only_use_cftime_datetimes=False)
    for i in range(len(time_final)):
        if(time_final[i].minute >= 45):
            time_final[i] = datetime.datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour+1)
        else:
            time_final[i] = datetime.datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour)

    # Now we need to subset data based on user specified
    # start time and end time
    time_slice = pd.DataFrame([])
    time_slice['index_slice'] = np.arange(len(time_final))
    time_slice.index = pd.to_datetime(time_final)
    idx = (time_slice.index >= args.start_time.strftime("%Y-%m-%d %H:%M:%S")) & (time_slice.index < args.stop_time.strftime("%Y-%m-%d %H:%M:%S"))
    time_indices = time_slice.loc[idx,'index_slice'].values

    time_netcdf = time_slice.loc[idx,:].index -  pd.to_datetime(args.start_time.strftime("%Y-%m-%d %H:%M:%S"))
    time_netcdf = time_netcdf.total_seconds()

    # Now slice zeta timeseries based on specified time stamp
    # the user has requested
    zeta = zeta[time_indices,:]
    #adpts = np.column_stack([FVCOM_lons, FVCOM_lats])
    #print("Querying nearest points")
    #_, stations = kd_nearest_neighbor(adpts, pli_data['values'])

    print(zeta.shape)
    print(time_netcdf.shape)
    if args.output.is_dir():
        args.output = args.output/"FVCOM_Waterlevel.bc"
        
    out_buf = np.empty((len(time_netcdf), 2), dtype='float64')
    out_buf[:, 0] = time_netcdf
    with BCFileWriter(args.output) as bc_out:
        print("Writing BC output", bc_out.filename)
        for name, station in zip(pli_data['index'], np.arange(len(time_netcdf))):
            print(f"Station {name} ({station})".ljust(50), end="\r")
            print(out_buf[:, 1].shape)
            print(zeta[:, station].shape)
            out_buf[:, 1] = np.ma.getdata(zeta[:, station])
            bc_out.add_forcing(name, 'timeseries', units, args.Lake, out_buf)
    print()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('FVCOM', type=pathlib.Path, help="FVCOM data path")
    parser.add_argument('Lake', type=str, help="Great Lake name so water level datum can be accounted for")
    parser.add_argument('pli', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument('--start', dest='start_time', required=True,
                    help='The date and time to begin making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--stop', dest='stop_time', required=True,
                    help='The date and time to stop making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path('.'), help="Path to bc output directory. Default is current directory")

    args.start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
    args.stop_time = datetime.datetime.strptime(args.stop_time, '%Y-%m-%d_%H:%M:%S')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)
