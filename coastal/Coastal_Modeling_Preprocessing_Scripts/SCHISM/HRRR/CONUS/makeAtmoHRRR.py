#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 2023

@author: Jason.Ducker
"""

import os, glob, numpy as np, netCDF4 as nc
import shutil
import argparse
import datetime
import pandas as pd
import subprocess
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
import math
from common.io import read_pli, read_polygon

parser = argparse.ArgumentParser()
parser.add_argument('sfluxfile', type=str, help='schism sflux2sourceInput.nc file')
parser.add_argument('polygon_enclosure', type=str, help='polygon_enclosure file')
parser.add_argument('hrrrdir', type=str, help='hrrr file directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('domain', type=str, help='domain name, one of hawaii, prvi, pacfic, or atlgulf')
parser.add_argument('start_time', type=str, help='Start time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 00:00"')
parser.add_argument('end_time', type=str, help='End time of the SCHISM simulation period in a string that is supported by the Python dateutil.parser package, for example, "2024-01-02 23:00"')
args = parser.parse_args()

c1 = 3600.0 #min/hr, sec/hr, or hr depending on the time variable in the wind files
c2 = 86400.0 #min/day, sec/day, or hr/day depending on the time variable in the wind files

# Specify 
#path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Pacific_Data/SCHISM'
#dataPath = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Pacific_Data/HRRR'
#path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/SCHISM'
path = args.output_dir
#dataPath = '/scratch2/NCEPDEV/ohd/Jason.Ducker/Atlantic_Data/HRRR'
dataPath = args.hrrrdir
#path to where the forcing data is stored
sfluxFile = args.sfluxfile

#start_time = "2014-07-31 00:00:00"
#end_time = "2014-08-09 01:00:00"
#start_time = "2018-09-12 00:00:00"
#end_time = "2018-09-22 00:00:00"
start_time = args.start_time
end_time = args.end_time


# parse start and end dates into expected 
# 1hr HRRR data intervals
start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
HRRR_datetimes = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=end_time.strftime('%Y-%m-%d %H:%M:%S'),freq='h')

#HRRR forecast issue time should be 1 hour earlier
forecast_issue_time = start_time - datetime.timedelta( hours=1 )

#file for rest of atmo variables, which will be used in the simulation so this should be saved in the folder where it will be used in the run
outFile = os.path.join(path,'sflux_air_1.0001.nc')
vFile = os.path.join(path,'source2.nc')

# open sflux file to read source/sink points
data = nc.Dataset(sfluxFile,'r')
precip2flux = data.variables['precip2flux'][:]
simplex = data.variables['simplex'][:]
area_cor = data.variables['area_cor'][:]
x = data.variables['x'][:]
y = data.variables['y'][:]


t = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
q = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
u = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
v = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
p = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
rainrate = np.zeros((len(HRRR_datetimes)+1,x.shape[0],x.shape[1]))
r = np.zeros((len(HRRR_datetimes),len(precip2flux)))

subhourly = False
# Specify where a Alaska HRRR raw grib2 file is on your system
#hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[0].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[0].strftime('%H') +'z.wrfsfcf01.grib2')
hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfsfcf01.grib2')
if(os.path.exists(hrrr_file) == False):
    hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfprsf01.grib2')
    if(os.path.exists(hrrr_file) == False):
        hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfsubhf01.grib2')
        subhourly = True

hrrr_tmp_nc = os.path.join(dataPath,'HRRR_tmp.nc')
# Execute the wgrib2 command to convert the HRRR grib2 file into a netcdf file
############ Remember to set the Linux $WGRIB2 environmental variable to the wgrib2 exectuable pathway ###############
if(subhourly):
    wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground:15 min fcst:|VGRD:10 m above ground:15 min fcst:|TMP:2 m above ground:15 min fcst:|DTMP:2 m above ground:15 min fcst:|PRATE:surface:15 min fcst:|PRES:surface:15 min fcst:)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
else:
    wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'

# Use subprocess module to execute the linux wgrib2 command and create the HRRR netcdf file
exitcode = subprocess.call(wgrib2_cmd, shell=True)


# Open temporary HRRR netcdf file to extract
# necessary time information
data = nc.Dataset(hrrr_tmp_nc)
# get reference time of the HRRR data
ref_date = pd.Timestamp(data.variables['time'].units[14:-7])
ref_time = data.variables['time'].units[:-7]
# calculate start time of the HRRR dataset extraction
# based on user specification
#start = (HRRR_datetimes[0] - ref_date).total_seconds()
#SCHISM assuses the units always start at hour 0 of the day
start = ( pd.Timestamp( HRRR_datetimes[0].year, HRRR_datetimes[0].month,HRRR_datetimes[0].day) - ref_date).total_seconds()
# create SCHISM string of reference data
#ref = 'days since ' + HRRR_datetimes[0].strftime('%Y-%m-%d ') + str(HRRR_datetimes[0].hour)
ref = 'days since ' + HRRR_datetimes[0].strftime('%Y-%m-%d')
# Get base date of start date for SCHISM file
#baseDate = [HRRR_datetimes[0].year,HRRR_datetimes[0].month,HRRR_datetimes[0].day,HRRR_datetimes[0].hour]
baseDate = [HRRR_datetimes[0].year,HRRR_datetimes[0].month,HRRR_datetimes[0].day, 0]
# close and remove the temporary HRRR file
data.close()


HRRR_proj = 'PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

met_data = xr.open_dataset(hrrr_tmp_nc)

geospatial_data = nc.Dataset(hrrr_tmp_nc)

met_data = met_data.rio.write_crs(HRRR_proj)

P_Enclosure = Polygon(read_polygon(args.polygon_enclosure))
#P_Enclosure = Polygon(read_polygon('/scratch2/NCEPDEV/ohd/Jason.Ducker/extract4dflow-master/Extract_DFlowFM_Input_Files/Polygon_enclosure_files/SCHISM_Pac_Enclosure.pol'))

dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

HRRR_x = met_data.x.values
HRRR_y = met_data.y.values

met_data.close()

mesh_x, mesh_y = np.meshgrid(HRRR_x,HRRR_y)

HRRR_lats = geospatial_data.variables['latitude'][:].data
HRRR_lons = (geospatial_data.variables['longitude'][:].data + 180) % 360 -180

geospatial_data.close()

HRRR_df = pd.DataFrame([])
HRRR_df['ids'] = np.arange(len(HRRR_lats.flatten()))
HRRR_df['x'] = HRRR_lons.flatten()
HRRR_df['y'] = HRRR_lats.flatten()
HRRR_df['HRRR_x'] = mesh_x.flatten()
HRRR_df['HRRR_y'] = mesh_y.flatten()


HRRR_gdf = gpd.GeoDataFrame(HRRR_df,geometry=gpd.points_from_xy(HRRR_df['x'],HRRR_df['y']),crs='WGS84')

intersection = HRRR_gdf.intersects(P_Enclosure)
ind_x = np.where(np.in1d(HRRR_x,HRRR_gdf.HRRR_x[intersection].values)==True)[0]
ind_y = np.where(np.in1d(HRRR_y,HRRR_gdf.HRRR_y[intersection].values)==True)[0]

os.remove(hrrr_tmp_nc)


# Make a temporary directory to store all netcdf converted files if it doesn't already exist
hrrr_nc_dir = os.path.join(path,'HRRR_nc_files')
if (os.path.exists(hrrr_nc_dir) == False):
    os.mkdir(hrrr_nc_dir)
else:
    shutil.rmtree( hrrr_nc_dir )
    os.mkdir(hrrr_nc_dir)

for i in range(len(HRRR_datetimes)):
    file_avail = True
    subhourly = False
    hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfsfcf' + f"{i+1:02}.grib2")
    if(os.path.exists(hrrr_file) == False):
        hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfprsf' + f"{i+1:02}.grib2")
        if(os.path.exists(hrrr_file) == False):
            hrrr_file = os.path.join(dataPath,'hrrr.'+forecast_issue_time.strftime('%Y%m%d') +'/' + 'hrrr.t'+ forecast_issue_time.strftime('%H') +'z.wrfsubhf' + f"{i+1:02}.grib2")
            if(os.path.exists(hrrr_file) == False):
                file_avail = False
            else:
                subhourly = True
                print(hrrr_file)
    if(file_avail):
        hrrr_tmp_nc = os.path.join(hrrr_nc_dir,'hrrr.'+ HRRR_datetimes[i].strftime('%Y_%m_%d_%H') +'.nc')
        if(subhourly):
            hrrr_precip = os.path.join(hrrr_nc_dir,'hrrr_precip.nc')
            wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground:15 min fcst:|VGRD:10 m above ground:15 min fcst:|TMP:2 m above ground:15 min fcst:|DPT:2 m above ground:15 min fcst:|PRES:surface:15 min fcst:)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
            exitcode = subprocess.call(wgrib2_cmd, shell=True)
            wgrib2_cmd = f'$WGRIB2 -match "(PRATE:surface:15 min fcst:|PRATE:surface:30 min fcst:|PRATE:surface:45 min fcst:|PRATE:surface:60 min fcst:)" {hrrr_file} -netcdf {hrrr_precip}'
            exitcode = subprocess.call(wgrib2_cmd, shell=True)
            hrrr_data = xr.open_dataset(hrrr_tmp_nc)
            precip_data = xr.open_dataset(hrrr_precip)
            precip = np.sum(precip_data['PRATE_surface'].data,axis=0)/3600.0
            hrrr_data['APCP_surface'] = hrrr_data['PRES_surface']
            hrrr_data.APCP_surface.attrs['short_name'] = "APCP_surface"
            hrrr_data.APCP_surface.attrs['long_name'] = "Total Precipitation"
            hrrr_data.APCP_surface.attrs['units'] = "kg/m^2"
            hrrr_data.variables['APCP_surface'].values[0,:,:] = precip
            hrrr_data['SPFH_2maboveground'] = hrrr_data['DPT_2maboveground']
            hrrr_data.APCP_surface.attrs['short_name'] = "SPFH_2maboveground"
            hrrr_data.APCP_surface.attrs['long_name'] = "Specific Humidity"
            hrrr_data.APCP_surface.attrs['units'] = "kg/kg"
            hrrr_data.variables['SPFH_2maboveground'].values[0,:,:] = (622*6.113/hrrr_data.PRES_surface.values*100)*np.exp(5423*(hrrr_data.DPT_2maboveground.values - 273.15)/(hrrr_data.DPT_2maboveground.values*273.15))/1000
            hrrr_data = hrrr_data.drop_vars(['DPT_2maboveground'])
            #hrrr_data.x.attrs['_FillValue'] = None
            #hrrr_data.y.attrs['_FillValue'] = None
            encoding = {'x': {'_FillValue': None, 'trivial_index': True}, 'y': {'_FillValue': None, 'trivial_index': True}, 'latitude': {'_FillValue': None, 'trivial_index': True}, 'longitude': {'_FillValue': None, 'trivial_index': True}, 'time': {'_FillValue': None, 'trivial_index': True}} 
            #hrrr_data.time.values = data.time.values - pd.Timedelta(15,'m')
            hrrr_data.to_netcdf(hrrr_tmp_nc)#,encoding=encoding
            hrrr_data.close()
            precip_data.close()
            os.remove(hrrr_precip)
        else:
            wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
            exitcode = subprocess.call(wgrib2_cmd, shell=True)

# Sort the HRRR netcdf files for xarray processing
hrrr_nc_files = glob.glob(os.path.join(hrrr_nc_dir,'*.nc'))
hrrr_nc_files.sort()

# Open all of the HRRR netcdf files at once using xarray
data = xr.open_mfdataset(hrrr_nc_files)
# Resample the 3hourly data and data gaps into hourly data and
# Interpolate the resampled data to fill in the NAN data gaps
# introduced by the resample method, which highlight the hourly
# time intervals where HRRR 3-hourly data is not available for Alaska
data = data.resample(time='1H').interpolate(kind='slinear')


#print(t.shape)
#print(data.shape)
t[0:-1,:,:] = data.variables['TMP_2maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
q[0:-1,:,:] = data.variables['SPFH_2maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
u[0:-1,:,:] = data.variables['UGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
v[0:-1,:,:] = data.variables['VGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
p[0:-1,:,:] = data.variables['PRES_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
rainrate = data.variables['APCP_surface'][:,:,:].values
for i in range(len(HRRR_datetimes)):
    if(i == len(HRRR_datetimes) -1):
        r[i,:] = np.sum((rainrate[-1,:,:][ind_y,:][:,ind_x].flatten()[simplex]/3600.0)*area_cor,axis=1)*precip2flux
    else:
        r[i,:] = np.sum((rainrate[i+1,:,:][ind_y,:][:,ind_x].flatten()[simplex]/3600.0)*area_cor,axis=1)*precip2flux

# add extra time step to the rest of the variables
t[-1,:,:] = data.variables['TMP_2maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
q[-1,:,:] = data.variables['SPFH_2maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
u[-1,:,:] = data.variables['UGRD_10maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
v[-1,:,:] = data.variables['VGRD_10maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
p[-1,:,:] = data.variables['PRES_surface'][-1,:,:].values[ind_y,:][:,ind_x]
time = np.zeros((len(HRRR_datetimes)+1))
#
#By default, the time values are rounded up by numpy. This causes probelms in SCHISM
# at reading the times when the time is a number with repeating digits.
#time[0:-1] = ((HRRR_datetimes - ref_date).total_seconds()-start)/c2
#time[0:-1] = ((HRRR_datetimes - ref_date).total_seconds()-start)/3600/24
#Here, we must round down (truncate) the numbers such that SCHISM can read the time correctly
for i, tm in enumerate( HRRR_datetimes ):
    time[i] = math.floor(((tm - ref_date).total_seconds()-start)/3600/24 * 10000000 ) / 10000000

#time[-1] = ((HRRR_datetimes[-1] - ref_date).total_seconds()-start)/c2
#expect 1 hour intervals. Add one more time step to the end
#
#time[-1] = ((HRRR_datetimes[-1] - ref_date).total_seconds()-start + 3600)/c2
#time[-1] = ((HRRR_datetimes[-1] - ref_date).total_seconds()-start + 3600)/3600/24
time[-1] = math.floor(((HRRR_datetimes[-1] - ref_date).total_seconds()-start + 3600)/3600/24 * 10000000)/10000000

data.close()

# Remove the HRRR netcdf file diectory
# after we're finished with the files
#if os.path.exists(hrrr_nc_dir):
#    os.rmdir(hrrr_nc_dir)


# If source2.nc file exists, then
# wipe out the file so we can create
# a new one for the HRRR data here
if os.path.exists(vFile):
    os.remove(vFile)
 
#write source2.nc file      
ncout = nc.Dataset(vFile,'w',format='NETCDF4')

ncout.createDimension('time_vsource',len(time)-1)
ncout.createDimension('nsources',len(precip2flux))

ncvso = ncout.createVariable('vsource','f8',('time_vsource','nsources',))

ncvso[:] = r

ncout.close()

if os.path.exists(outFile):
    os.remove(outFile)
            
#write atmo input file
ncoutt = nc.Dataset(outFile,'w',format='NETCDF4')

ncoutt.createDimension('time',len(time))
ncoutt.createDimension('ny_grid',x.shape[0])
ncoutt.createDimension('nx_grid',x.shape[1])

nctime = ncoutt.createVariable('time','f4',('time',))
nclonn = ncoutt.createVariable('lon','f4',('ny_grid','nx_grid',))
nclat = ncoutt.createVariable('lat','f4',('ny_grid','nx_grid',))
ncu = ncoutt.createVariable('uwind','f4',('time','ny_grid','nx_grid',))
ncv = ncoutt.createVariable('vwind','f4',('time','ny_grid','nx_grid',))
ncp = ncoutt.createVariable('prmsl','f4',('time','ny_grid','nx_grid',))
nct = ncoutt.createVariable('stmp','f4',('time','ny_grid','nx_grid',))
ncq = ncoutt.createVariable('spfh','f4',('time','ny_grid','nx_grid',))

nctime[:] = time
nctime.long_name = "Time"
nctime.standard_name = "time"
nctime.units = ref
nctime.base_date = baseDate

nclonn[:,:] = x
nclonn.long_name = "Longitude"
nclonn.standard_name = "longitude"
nclonn.units = "degrees_east"
nclat[:,:] = y
nclat.long_name = "Latitude"
nclat.standard_name = "latitude"
nclat.units = "degrees_north"
ncu[:,:,:] = u
ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
ncu.standard_name = "eastward_wind"
ncu.units = "m/s"
ncv[:,:,:] = v
ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
ncv.standard_name = "northward_wind"
ncv.units = "m/s"
ncp[:,:,:] = p
ncp.long_name = "Pressure reduced to MSL"
ncp.standard_name = "air_pressure_at_sea_level"
ncp.units = "Pa"
nct[:,:,:] = t
nct.long_name = "Surface Air Temperature (2m AGL)"
nct.standard_name = "air_temperature"
nct.units = "K"
ncq[:,:,:] = q
ncq.long_name = "Surface Specific Humidity (2m AGL)"
ncq.standard_name = "specific_humidity"
ncq.units = "kg/kg"

ncoutt.close()
