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

import netCDF4
import numpy as np
from os.path import join
from common.io import BCFileWriter, read_pli
from common.geometry import kd_nearest_neighbor
import matplotlib.pyplot as plt
import os

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

def SCHISM_hgrid_coords(gridFile):
    lon = []
    lat = []
    bnodes = []
    with open(gridFile) as f:
        next(f)
        line = f.readline()
        ne = int(line.split()[0])
        nn = int(line.split()[1])
        for i in range(nn):
            line = f.readline()
            lon.append(float(line.split()[1]))
            lat.append(float(line.split()[2]))
        for i in range(ne):
            f.readline()
        line = f.readline()
        nbounds = int(line.split()[0])
        next(f)
        for i in range(nbounds):
            line = f.readline()
            nnodes = int(line.split()[0])
            for j in range(nnodes):
                bnodes.append(int(f.readline()))
    lon = [lon[b-1] for b in bnodes]
    lat = [lat[b-1] for b in bnodes]
    schism_bnds = np.column_stack([lon,lat])
    return schism_bnds, bnodes

def main(args):
    print("Reading PLI")
    pli_data = read_pli(args.pli)
    print("Reading SCHISM hgrid3 file")
    schism_bnds, bnodes = SCHISM_hgrid_coords(args.hgrid3)

    with netCDF4.Dataset(args.elev2d, mode='r') as ds:
        zeta = ds.variables['time_series']
        #print("Masking invalid stations")
        #mask = invalid_mask(zeta)
        #valid_stations = mask.nonzero()[0]
        #print(f"{len(mask) - len(valid_stations)} of {len(mask)} discarded")
        print("Querying nearest points")
        _, stations = kd_nearest_neighbor(schism_bnds,pli_data['values'])
        #stations = valid_stations[CN]

        time_col = ds.variables['time']
        ref_time = 'seconds since ' + args.schism_start_time.strftime("%Y-%m-%d %H:%M:%S")
        units = [('time', ref_time),
                 ('waterlevelbnd', 'm')]
        
        if args.output.is_dir():
            args.output = args.output/"SCHISM_waterlevel.bc"

        plot_output_dir = join(args.output_dir,'zeta_comparison_plots')
        os.mkdir(plot_output_dir)

        #plt.figure(figsize=(10,10))
    
        #ds_dflowfm = netCDF4.Dataset(args.dflowfm_elev2d, mode='r')
        #dflowfm_zeta = ds_dflowfm.variables['time_series']

        bnd_ind = 0

        out_buf = np.empty((len(time_col), 2), dtype='float64')
        out_buf[:, 0] = time_col[:]
        with BCFileWriter(args.output) as bc_out:
            print("Writing BC output", bc_out.filename)
            for name, station in zip(pli_data['index'], stations):
                print(f"Station {name} ({station})".ljust(50), end="\r")
                out_buf[:, 1] = np.ma.getdata(zeta[:, station,0,0])
                bc_out.add_forcing(name, 'timeseries', units, out_buf)
                #output_fig1 = join(plot_output_dir,str(name) + '_zeta_comparison.png')
                #plt.title(f"Station {name} Water level timeseries comparison")
                #plt.xlabel(ref_time)
                #plt.ylabel('Water level (m MSL)')
                #plt.plot(time_col[:],zeta[:,station,0,0])
                #plt.plot(time_col[:],dflowfm_zeta[:,bnd_ind,0,0])
                #plt.legend(['SCHISM','DFLOWFM'])
                #plt.savefig(output_fig1, dpi = 600)
                #plt.clf()
                bnd_ind += 1
        print()

        #ds_dflowfm.close()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('elev2d', type=pathlib.Path, help="SCHISM elev2D.th.nc file")
    #parser.add_argument('dflowfm_elev2d', type=pathlib.Path, help="DFLOWFM elev2D.th.nc file")
    parser.add_argument('hgrid3', type=pathlib.Path, help="SCHISM hgrid.gr3 file")
    parser.add_argument('pli', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument('schism_start_time', help="SCHISM start time for elev2D.th.nc file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path('.'), help="Path to bc output directory. Default is current directory")
    parser.add_argument('output_dir', type=pathlib.Path, help="Output directory pathway")

    args = parser.parse_args()

    args.schism_start_time = datetime.datetime.strptime(args.schism_start_time, '%Y-%m-%d_%H:%M:%S')

    return args

if __name__ == "__main__":
    args = get_options()
    main(args)
