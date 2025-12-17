#!/usr/bin/env python3makeOcean.py
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:47:37 2021

@author: Jason.Ducker
"""
import os, numpy as np, netCDF4 as nc
from datetime import datetime
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import pandas as pd
from common.geometry import kd_nearest_neighbor
from scipy.interpolate import griddata
from pytides.tide import Tide
import pytides.constituent as con


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

##################################### User options to modify ############################
domain = 'pr'

CERA1 = "/scratch2/NCEPDEV/ohd/Jason.Ducker/Puerto_Rico_Data/Irma_fort.63.nc"
CERA2 = "/scratch2/NCEPDEV/ohd/Jason.Ducker/Puerto_Rico_Data/Maria_fort.63.nc"
gridFile = "/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/hgrid.gr3"
outFile = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/PR+USVI/SCHISM/elev2Dth.nc'
#convFile = path+'Data/domainFiles/atl/final/atlBoundNodes_conversion_factor.txt'
start_time = datetime.strptime("2017-09-01 00:00", '%Y-%m-%d %H:%M')
end_time = datetime.strptime("2017-10-15 00:00", '%Y-%m-%d %H:%M')
########################################################################################

if domain == 'pac' or domain == 'atl':
    offset = []
    with open(convFile) as f:
        next(f)
        for line in f:
            offset.append(float(line.split(',')[5]))

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

amplitude, phase = PyTide_model("/scratch2/NCEPDEV/ohd/Jason.Ducker/NGEN_COASTAL_MODEL_OCEAN_DATA/TidalConst", lon, lat)

CERA_daterange = pd.date_range(start=start_time.strftime("%Y-%m-%d %H:%M:%S"),end=end_time.strftime("%Y-%m-%d %H:%M:%S"),freq="h")


schism_bnds = np.column_stack([lon,lat])

ds1 = nc.Dataset(CERA1, mode='r')
ds2 = nc.Dataset(CERA2, mode='r')

time1_col = ds1.variables['time']
ref_time1 = time1_col.units.rstrip('UTC').rstrip()
time1_final = nc.num2date(time1_col[:],units=ref_time1,only_use_cftime_datetimes=False)

time2_col = ds2.variables['time']
ref_time2 = time2_col.units.rstrip('UTC').rstrip()
time2_final = nc.num2date(time2_col[:],units=ref_time2,only_use_cftime_datetimes=False)


zeta1 = ds1.variables['zeta']
print("Masking invalid stations")
mask = invalid_mask(zeta1)
valid_stations = mask.nonzero()[0]
print(f"{len(mask) - len(valid_stations)} of {len(mask)} discarded")
print("Masking coordinates")
#mask = np.where(ds1.variables['x'] < np.nanmin(lon)-1, False, mask)
#mask = np.where(ds1.variables['x'] > np.nanmax(lon)+1, False, mask)
#mask = np.where(ds1.variables['y'] < np.nanmin(lat)-1, False, mask)
#mask = np.where(ds1.variables['y'] > np.nanmax(lat)+1, False, mask)
adlons = np.ma.getdata(ds1.variables['x'][mask])
adlats = np.ma.getdata(ds1.variables['y'][mask])
adpts = np.column_stack([adlons, adlats])
print("Querying nearest points")
_, CN = kd_nearest_neighbor(adpts,schism_bnds)
stations1 = valid_stations[CN]
zeta1 = zeta1[:,stations1]
ds1.close()

zeta2 = ds2.variables['zeta']
print("Masking invalid stations")
mask = invalid_mask(zeta2)
valid_stations = mask.nonzero()[0]
print(f"{len(mask) - len(valid_stations)} of {len(mask)} discarded")
print("Masking coordinates")
#mask = np.where(ds2.variables['x'] < np.nanmin(lon)-1, False, mask)
#mask = np.where(ds2.variables['x'] > np.nanmax(lon)+1, False, mask)
#mask = np.where(ds2.variables['y'] < np.nanmin(lat)-1, False, mask)
#mask = np.where(ds2.variables['y'] > np.nanmax(lat)+1, False, mask)
adlons = np.ma.getdata(ds2.variables['x'][mask])
adlats = np.ma.getdata(ds2.variables['y'][mask])
adpts = np.column_stack([adlons, adlats])
print("Querying nearest points")
_, CN = kd_nearest_neighbor(adpts,schism_bnds)
stations2 = valid_stations[CN]
zeta2 = zeta2[:,stations2]
ds2.close()

zeta_final = np.empty((len(CERA_daterange),len(lon)),dtype=float)
for i in range(len(CERA_daterange)):
    if(CERA_daterange[i] in time2_final):
        idx_time = np.where(time2_final==CERA_daterange[i])[0][0]
        zeta_final[i,:] = zeta2[idx_time,:]
    elif(CERA_daterange[i] in time1_final):
        idx_time = np.where(time1_final==CERA_daterange[i])[0][0]
        zeta_final[i,:] = zeta1[idx_time,:]
    else:
        zeta_final[i,:] = tidal_wl(CERA_daterange[i],amplitude,phase)

# Now we need to calculate the time variable based on reference
# date and convert to total seconds
time_netcdf = CERA_daterange - pd.to_datetime(start_time.strftime("%Y-%m-%d %H:%M:%S"))
time_netcdf = time_netcdf.total_seconds()

if os.path.exists(outFile):
    os.remove(outFile)

# open a netCDF file to write
ncout = nc.Dataset(outFile,'w',format='NETCDF4')

# define axis size
ncout.createDimension('time',None)
ncout.createDimension('nOpenBndNodes',len(lon))
ncout.createDimension('nLevels',1)
ncout.createDimension('nComponents',1)
ncout.createDimension('one',1)

# create time step variable
nctstep = ncout.createVariable('time_step','f8',('one',)) 

# create time axis
nctime = ncout.createVariable('time','f8',('time',))
#nctime.setncattr('units',ref_time)

# create water level time series
ncwl = ncout.createVariable('time_series','f8',('time','nOpenBndNodes','nLevels','nComponents',))

# copy axis from original dataset
nctstep[:] = 3600.0
nctime[:] = time_netcdf
ncwl[:] = zeta_final

ncout.close()
