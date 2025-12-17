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
##################################### User options to modify ############################
DFlowFM = "/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/COASTAL_ACT_WLs/Waterlevel.bc"
DFlowFM_pli = "/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/COASTAL_ACT_WLs/CONT_BND_Mod.pli"
gridFile = "/scratch2/NCEPDEV/ohd/Camaron.George/Data/domainFiles/atl/hgrid.gr3"
outFile = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/COASTAL_ACT_WLs/elev2Dth.nc'
#convFile = path+'Data/domainFiles/atl/final/atlBoundNodes_conversion_factor.txt'
start_time = datetime.strptime("2018-08-31 00:00", '%Y-%m-%d %H:%M')
end_time = datetime.strptime("2018-09-24 00:00", '%Y-%m-%d %H:%M')
########################################################################################

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


DFlowFM_file = open(DFlowFM,'r')
DFlowFM_file = DFlowFM_file.readlines()

time = []
for i in range(len(DFlowFM_file)):
    DFlowFM_line = DFlowFM_file[i].split('\n')
    if(DFlowFM_line[0].split('=')[0].strip() == 'Unit' and i < 8):
        ref_date = DFlowFM_line[0].split('=')[-1].strip()
    if(DFlowFM_line[0] == '' and i > 0):
        break
    try:
        timestep = float(DFlowFM_line[0].split(' ')[0])
        time.append(timestep)
    except:
        pass

time_final = nc.num2date(time,units=ref_date,only_use_cftime_datetimes=False)

bnd_names = []
for i in range(len(DFlowFM_file)):
    if(DFlowFM_file[i].split('\n')[0].split('=')[0].strip() == 'Name'):
        bnd_names.append(DFlowFM_file[i].split('\n')[0].split('=')[-1].strip())

wl_data = np.empty((len(time_final),len(bnd_names)),dtype=float)

index = -1
for i in range(len(DFlowFM_file)):
    DFlowFM_line = DFlowFM_file[i].split('\n')
    if(DFlowFM_line[0] == ''):
        index += 1
        timestep = 0
    try:
        wl = float(DFlowFM_line[0].split(' ')[-1].strip())
        wl_data[timestep,index] = wl
        timestep += 1
    except:
        pass


DFlowFM_file = open(DFlowFM_pli,'r')
DFlowFM_file = DFlowFM_file.readlines()[2:]

DFlowFM_lats = np.empty(len(bnd_names),dtype=float)
DFlowFM_lons = np.empty(len(bnd_names),dtype=float)

for i in range(len(DFlowFM_file)):
    DFlowFM_line = DFlowFM_file[i].split(' ')
    DFlowFM_lons[i] = float(DFlowFM_line[0].strip())
    DFlowFM_lats[i] = float(DFlowFM_line[1].strip())

# Now we need to subset data based on user specified
# start time and end time
time_slice = pd.DataFrame([])
time_slice['index_slice'] = np.arange(len(time_final))
time_slice.index = pd.to_datetime(time_final)
idx = (time_slice.index >= start_time.strftime("%Y-%m-%d %H:%M:%S")) & (time_slice.index < end_time.strftime("%Y-%m-%d %H:%M:%S"))
time_indices = time_slice.loc[idx,'index_slice'].values

time_netcdf = time_slice.loc[idx,:].index -  pd.to_datetime(start_time)
time_netcdf = time_netcdf.total_seconds()


schism_bnds = np.column_stack([lon,lat])
DFlowFM_points = np.column_stack([DFlowFM_lons,DFlowFM_lats])
tree = cKDTree(DFlowFM_points)
_, open_bnds = tree.query(schism_bnds)

wl_data = wl_data[time_indices,:]

zeta = np.empty((len(time_netcdf),len(lon),1,1),dtype=float)

for i in range(len(lon)):
    zeta[:,i,0,0] = wl_data[:,open_bnds[i]]


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
ncwl[:] = zeta 

ncout.close()
