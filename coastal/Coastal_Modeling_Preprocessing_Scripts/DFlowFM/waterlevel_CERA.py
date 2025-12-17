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
import pandas as pd

import netCDF4 as nc
import numpy as np
from datetime import datetime

from common.io import BCFileWriter, read_pli
from common.geometry import kd_nearest_neighbor
from scipy.interpolate import griddata
from pytides.tide import Tide
import pytides.constituent as con
import os

def PyTide_model(consts_path, schism_x_bnd, schism_y_bnd):
    tide_constants = ['k1', 'k2', 'm2', 'n2', 'o1', 'p1', 'q1', 's2']
    consfiles = list(map(lambda x: os.path.join(consts_path, x)+".nc", tide_constants))
    pha = np.zeros((len(schism_x_bnd), len(consfiles)))
    amp = np.zeros((len(schism_x_bnd), len(consfiles)))
    col = 0
    for file in consfiles:
        data = nc.Dataset(file, 'r')
        if col == 0:
            x = data.variables['lon'][:]
            y = data.variables['lat'][:]
            x, y = np.meshgrid(x, y)
            i = np.where(x > 180.0)
            x[i] = x[i]-360.0
            i = np.where((x < max(schism_x_bnd)+1) & (x > min(schism_x_bnd)-1) & (y < max(schism_y_bnd)+1) & (y > min(schism_y_bnd)-1))
            x = x[i]
            y = y[i]
        a = data.variables['amplitude'][:]
        a = a[i]
        p = data.variables['phase'][:]
        p = p[i]
        xI = x[a.mask == False]
        yI = y[a.mask == False]
        p = p[a.mask == False]
        a = a[a.mask == False]
        amp[:, col] = griddata((xI, yI), a, (schism_x_bnd, schism_y_bnd), method='nearest')
        pha[:, col] = griddata((xI, yI), p, (schism_x_bnd, schism_y_bnd), method='nearest')
        col += 1
    return amp, pha

def tidal_wl(fdate,amplitude,phase):
    pred_times = Tide._times(fdate,np.array([0.0],dtype=float))
    wl = np.zeros((len(pred_times), amplitude.shape[0]))
    cons = [c for c in con.noaa if c != con._Z0]
    n = [3, 34, 0, 2, 5, 29, 25, 1]  # corresponds to position in list in constituent.py
    cons = [cons[c] for c in n]
    tide_model = np.zeros(len(cons), dtype=Tide.dtype)
    tide_model['constituent'] = cons
    for i in range(amplitude.shape[0]):
        tide_model['amplitude'] = amplitude[i, :]
        tide_model['phase'] = phase[i, :]
        tide = Tide(model=tide_model, radians=False)
        wl[:, i] = (tide.at(pred_times))/100.0
    return wl[0,:]

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

    start_time = args.start_time
    end_time = args.stop_time

    amplitude, phase = PyTide_model("/scratch2/NCEPDEV/ohd/Jason.Ducker/NGEN_COASTAL_MODEL_OCEAN_DATA/TidalConst", pli_data['values'][:,0],pli_data['values'][:,1])

    CERA_daterange = pd.date_range(start=start_time.strftime("%Y-%m-%d %H:%M:%S"),end=end_time.strftime("%Y-%m-%d %H:%M:%S"),freq="h")

    ds1 = nc.Dataset(args.CERA, mode='r')
    
    time1_col = ds1.variables['time']
    ref_time1 = time1_col.units.rstrip('UTC').rstrip()
    time1_final = nc.num2date(time1_col[:],units=ref_time1,only_use_cftime_datetimes=False)

    zeta1 = ds1.variables['zeta']
    print("Masking invalid stations")
    mask = invalid_mask(zeta1)
    valid_stations = mask.nonzero()[0]
    print(f"{len(mask) - len(valid_stations)} of {len(mask)} discarded")
    print("Masking coordinates")
    adlons = np.ma.getdata(ds1.variables['x'][mask])
    adlats = np.ma.getdata(ds1.variables['y'][mask])
    adpts = np.column_stack([adlons, adlats])
    print("Querying nearest points")
    _, CN = kd_nearest_neighbor(adpts,pli_data['values'])
    stations1 = valid_stations[CN]
    zeta1 = zeta1[:,stations1]
    ds1.close()

    zeta_final = np.empty((len(CERA_daterange),len(pli_data['values'])),dtype=float)
    for i in range(len(CERA_daterange)):
        if(CERA_daterange[i] in time1_final):
            idx_time = np.where(time1_final==CERA_daterange[i])[0][0]
            zeta_final[i,:] = zeta1[idx_time,:]
        else:
            zeta_final[i,:] = tidal_wl(CERA_daterange[i],amplitude,phase)

    # Now we need to calculate the time variable based on reference
    # date and convert to total seconds
    time_netcdf = CERA_daterange - pd.to_datetime(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    time_netcdf = time_netcdf.total_seconds()

    ref_time = 'seconds since ' + start_time.strftime("%Y-%m-%d %H:%M:%S")
    units = [('time', ref_time),('waterlevelbnd', 'm')]
        
    if args.output.is_dir():
        args.output = args.output/"waterlevel.bc"
        
    out_buf = np.empty((len(time_netcdf), 2), dtype='float64')
    out_buf[:, 0] = time_netcdf
    with BCFileWriter(args.output) as bc_out:
        print("Writing BC output", bc_out.filename)
        for name, station in zip(pli_data['index'], np.arange(len(pli_data['values']))):
            print(f"Station {name} ({station})".ljust(50), end="\r")
            out_buf[:, 1] = np.ma.getdata(zeta_final[:, station])
            bc_out.add_forcing(name, 'timeseries', units, out_buf)
    print()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('CERA', type=pathlib.Path, help="CERA file fort.63.nc path")
    parser.add_argument('--start', dest='start_time', required=True,
                    help='The date and time to begin making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--stop', dest='stop_time', required=True,
                    help='The date and time to stop making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('pli', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path('.'), help="Path to bc output directory. Default is current directory")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_options()
    main(args)
