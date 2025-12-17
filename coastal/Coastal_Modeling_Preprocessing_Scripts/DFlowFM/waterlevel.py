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
from datetime import datetime
import netCDF4
import numpy as np

from common.io import BCFileWriter, read_pli
from common.geometry import kd_nearest_neighbor


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

    with netCDF4.Dataset(args.fort63, mode='r') as ds:
        zeta = ds.variables['zeta']
        print("Masking invalid stations")
        mask = invalid_mask(zeta)
        valid_stations = mask.nonzero()[0]
        print(f"{len(mask) - len(valid_stations)} of {len(mask)} discarded")
        print("Masking coordinates")
        adlons = np.ma.getdata(ds.variables['x'][mask])
        adlats = np.ma.getdata(ds.variables['y'][mask])

        adpts = np.column_stack([adlons, adlats])
        print("Querying nearest points")
        _, CN = kd_nearest_neighbor(adpts, pli_data['values'])
        stations = valid_stations[CN]

        time_col = ds.variables['time']
        ref_time = time_col.units.rstrip('UTC').rstrip()
        units = [('time', ref_time),
                 ('waterlevelbnd', 'm')]
        
        if args.output.is_dir():
            args.output = args.output/"waterlevel.bc"
        
        out_buf = np.empty((len(time_col), 2), dtype='float64')
        out_buf[:, 0] = time_col[:]
        with BCFileWriter(args.output) as bc_out:
            print("Writing BC output", bc_out.filename)
            for name, station in zip(pli_data['index'], stations):
                print(f"Station {name} ({station})".ljust(50), end="\r")
                out_buf[:, 1] = np.ma.getdata(zeta[:, station])
                bc_out.add_forcing(name, 'timeseries', units, out_buf)
        print()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('fort63', type=pathlib.Path, help="Adcirc fort.63.nc path")
    parser.add_argument('pli', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path('.'), help="Path to bc output directory. Default is current directory")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)
