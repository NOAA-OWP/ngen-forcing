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
import geopandas as gpd
##################################### User options to modify ############################
domain = 'alk'

FVCOM_dir = "/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Michigan_Huron/FVCOM/"
gridFile = "/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Michigan-Huron/SCHISM/hgrid.gr3"
outFile = '/scratch2/NCEPDEV/ohd/NG_Coastal_Model_Setups/Lake_Michigan-Huron/SCHISM/elev2Dth.nc'#'/scratch2/NCEPDEV/ohd/Jason.Ducker/Lake_Michigan_Huron/elev2Dth.nc'
#convFile = path+'Data/domainFiles/atl/final/atlBoundNodes_conversion_factor.txt'
start_time = datetime.strptime("2020-06-01 00:00", '%Y-%m-%d %H:%M')
end_time = datetime.strptime("2020-08-01 00:00", '%Y-%m-%d %H:%M')
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


FVCOM_daterange = pd.date_range(start=start_time.strftime("%Y-%m-%d %H:%M:%S"),end=end_time.strftime("%Y-%m-%d %H:%M:%S"),freq="6H")
forecast_range = np.arange(6)+1
FVCOM_outfile = "nos.lmhofs.fields.n00"
FVCOM_file = FVCOM_outfile + str(forecast_range[0]) + '.' + FVCOM_daterange[0].strftime("%Y%m%d") + '.t' + FVCOM_daterange[0].strftime("%H") + 'z.nc'
FVCOM_data =  nc.Dataset(os.path.join(FVCOM_dir,FVCOM_file), mode='r')
FVCOM_lons =  FVCOM_data.variables['lon'][:].data
FVCOM_lats =  FVCOM_data.variables['lat'][:].data
df_var = np.arange(len(FVCOM_lats))+1
df = pd.DataFrame([])
df['ind'] = df_var
df['lat'] = FVCOM_lats
df['lon'] = FVCOM_lons
gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.lon,df.lat),crs='EPSG:4269')
gdf = gdf.to_crs('WGS84')
FVCOM_lats = []
FVCOM_lons = []
for i in range(len(df)):
    FVCOM_lons.append((gdf.geometry[i].coords.xy[0][0]+180)% 360 - 180)
    FVCOM_lats.append(gdf.geometry[i].coords.xy[1][0])

FVCOM_lats = np.ma.getdata(FVCOM_lats)
FVCOM_lons = np.ma.getdata(FVCOM_lons)

time_col = FVCOM_data.variables['time']
ref_time = time_col.units.rstrip('UTC').rstrip()
units = [('time', ref_time),
         ('waterlevelbnd', 'm')]
zeta = np.empty((len(FVCOM_daterange)*6,len(FVCOM_lons)),dtype=float)
time = np.empty(len(FVCOM_daterange)*6,dtype=float)
count = 0
FVCOM_files = []
for i in range(len(FVCOM_daterange)):
    for hour in forecast_range:
        locfile = FVCOM_outfile + str(hour) + '.' + FVCOM_daterange[i].strftime("%Y%m%d") + '.t' + FVCOM_daterange[i].strftime("%H") + 'z.nc'
        ds = nc.Dataset(os.path.join(FVCOM_dir,locfile), mode='r')
        FVCOM_files.append(locfile)
        zeta[count,:] = ds.variables['zeta'][:].data
        time[count] = ds.variables['time'][:].data[0]
        count += 1

time_final = nc.num2date(time,units=ref_time,only_use_cftime_datetimes=False)
for i in range(len(time_final)):
    if(time_final[i].minute >= 45):
        time_final[i] = datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour+1)
    else:
        time_final[i] = datetime(time_final[i].year,time_final[i].month,time_final[i].day,time_final[i].hour)

# Now we need to subset data based on user specified
# start time and end time
time_slice = pd.DataFrame([])
time_slice['index_slice'] = np.arange(len(time_final))
time_slice.index = pd.to_datetime(time_final)
idx = (time_slice.index >= start_time.strftime("%Y-%m-%d %H:%M:%S")) & (time_slice.index <= end_time.strftime("%Y-%m-%d %H:%M:%S"))
time_indices = time_slice.loc[idx,'index_slice'].values

time_netcdf = time_slice.loc[idx,:].index -  pd.to_datetime(start_time)
time_netcdf = time_netcdf.total_seconds()

#time_netcdf = nc.date2num(time_final,ref_time)

schism_bnds = np.column_stack([lon,lat])
FVCOM_points = np.column_stack([FVCOM_lons,FVCOM_lats])
tree = cKDTree(FVCOM_points)
_, open_bnds = tree.query(schism_bnds)

            
zeta = zeta[:,open_bnds]
zeta = zeta[time_indices,:]

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
ncwl[:] = zeta + 176.0

ncout.close()
