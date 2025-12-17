#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 2023

@author: Jason.Ducker
"""

import os, glob, numpy as np, netCDF4 as nc
import datetime
import pandas as pd
import subprocess
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon
from common.io import read_pli, read_polygon

c1 = 3600.0 #min/hr, sec/hr, or hr depending on the time variable in the wind files
c2 = 86400.0 #min/day, sec/day, or hr/day depending on the time variable in the wind files

# Specify 
path = '/scratch2/NCEPDEV/ohd/Jason.Ducker/HRRR_SCHISM/SCHISM_2023'
dataPath = '/scratch2/NCEPDEV/ohd/Jason.Ducker/HRRR_SCHISM/Alaska_HRRR/HRRR_2023/part1'
#path to where the forcing data is stored
sfluxFile = os.path.join(path,'sflux2sourceInput.nc')

start_time = "2023-04-07 00:00:00"
end_time = "2023-06-07 00:00:00"

# create datetimes for user specified start and end time
start_time_user_dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time_user_dt = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')


# Initialize start and end time of user
# so search algorithim below can find nearest
# 3HR HRRR cycle needed to fill the data gaps
start_time_user = start_time
end_time_user = end_time

# We need to account for grabbing the previous
# HRRR cycling if user requests a certain hour
# of the day that does not start OR
# end on a 3Hr cycle based on time series requested
start_hour = int(start_time[11:13])
if(start_hour == 0):
    start_time_tmp = (datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    start_time =  start_time_tmp[0:11] + '21' + start_time_tmp[13:]
elif(start_hour >= 1 and start_hour < 4):
    start_time = start_time[0:11] + '00' + start_time[13:]
elif(start_hour >= 4 and start_hour < 7):
    start_time = start_time[0:11] + '03' + start_time[13:]
elif(start_hour >= 7 and start_hour < 10):
    start_time = start_time[0:11] + '06' + start_time[13:]
elif(start_hour >= 10 and start_hour < 13):
    start_time = start_time[0:11] + '09' + start_time[13:]
elif(start_hour >= 13 and start_hour < 16):
    start_time = start_time[0:11] + '12' + start_time[13:]
elif(start_hour >= 16 and start_hour < 19):
    start_time = start_time[0:11] + '15' + start_time[13:]
elif(start_hour >= 19 and start_hour < 22):
    start_time = start_time[0:11] + '18' + start_time[13:]
elif(start_hour > 22):
    start_time = start_time[0:11] + '21' + start_time[13:]

end_hour = int(end_time[11:13])
if(end_hour <= 1):
    end_time =  end_time[0:11] + '00' + end_time[13:]
elif(end_hour > 1 and end_hour <= 4):
    end_time = end_time[0:11] + '03' + end_time[13:]
elif(end_hour > 4 and end_hour <= 7):
    end_time = end_time[0:11] + '06' + end_time[13:]
elif(end_hour > 7 and end_hour <= 10):
    end_time = end_time[0:11] + '09' + end_time[13:]
elif(end_hour > 10 and end_hour <= 13):
    end_time = end_time[0:11] + '12' + end_time[13:]
elif(end_hour > 13 and end_hour <= 16):
    end_time = end_time[0:11] + '15' + end_time[13:]
elif(end_hour > 16 and end_hour <= 19):
    end_time = end_time[0:11] + '18' + end_time[13:]
elif(end_hour > 19 and end_hour <= 22):
    end_time = end_time[0:11] + '21' + end_time[13:]
elif(end_hour > 22):
    end_time_tmp = (datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')    
    end_time = end_time_tmp[0:11] + '00' + end_time_tmp[13:]

end_time = (datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
   

# parse start and end dates into expected 
# 3hr Alaska HRRR data intervals
start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
HRRR_datetimes = pd.date_range(start=start_time,end=end_time,freq='3H')
HRRR_datetimes_interp = pd.date_range(start=start_time_user,end=end_time_user,freq='H')[0:-1]


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
        
t = np.zeros((len(HRRR_datetimes_interp)+1,x.shape[0],x.shape[1]))
q = np.zeros((len(HRRR_datetimes_interp)+1,x.shape[0],x.shape[1]))
u = np.zeros((len(HRRR_datetimes_interp)+1,x.shape[0],x.shape[1]))
v = np.zeros((len(HRRR_datetimes_interp)+1,x.shape[0],x.shape[1]))
p = np.zeros((len(HRRR_datetimes_interp)+1,x.shape[0],x.shape[1]))
r = np.zeros((len(HRRR_datetimes_interp),len(precip2flux)))
time = np.zeros((len(HRRR_datetimes_interp)+1))

# Quick algorithim to adjust data availability based
# on user start and end times specified for Alaska
# HRRR data since there are missing data gaps on the
# HPSS tape storage system
data_avail = False
offset = 0
while(data_avail == False):
    start_datetime = HRRR_datetimes[0] - datetime.timedelta(hours=offset)
    hrrr_file = os.path.join(dataPath,'hrrr.'+start_datetime.strftime('%Y%m%d') +'/' + 'hrrr.t'+ start_datetime.strftime('%H') +'z.wrfprsf01.ak.grib2')
    if os.path.exists(hrrr_file):
        data_avail = True
    offset += 3
    
start_time = start_datetime.strftime('%Y-%m-%d %H:%M:%S')

data_avail = False
offset = 0
while(data_avail == False):
    end_datetime = HRRR_datetimes[-1] + datetime.timedelta(hours=offset)
    hrrr_file = os.path.join(dataPath,'hrrr.'+end_datetime.strftime('%Y%m%d') +'/' + 'hrrr.t'+ end_datetime.strftime('%H') +'z.wrfprsf01.ak.grib2')
    if os.path.exists(hrrr_file):
        data_avail = True
    offset += 3
    
end_time = end_datetime.strftime('%Y-%m-%d %H:%M:%S')



HRRR_datetimes = pd.date_range(start=start_time,end=end_time,freq='3H')


# Specify where a Alaska HRRR raw grib2 file is on your system
hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[0].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[0].strftime('%H') +'z.wrfprsf01.ak.grib2')
hrrr_tmp_nc = os.path.join(path,'HRRR_tmp.nc')
# Execute the wgrib2 command to convert the HRRR grib2 file into a netcdf file
############ Remember to set the Linux $WGRIB2 environmental variable to the wgrib2 exectuable pathway ###############
wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|DSWRF:surface|DLWRF:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
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
start = (HRRR_datetimes_interp[0] - ref_date).total_seconds()
# create SCHISM string of reference data
ref = 'days since ' + HRRR_datetimes_interp[0].strftime('%Y-%m-%d ') + str(HRRR_datetimes_interp[0].hour)
# Get base date of start date for SCHISM file
baseDate = [HRRR_datetimes_interp[0].year,HRRR_datetimes_interp[0].month,HRRR_datetimes_interp[0].day,HRRR_datetimes_interp[0].hour]
# close and remove the temporary HRRR file
data.close()


HRRR_AK_proj = 'PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",60],PARAMETER["central_meridian",225],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]'

met_data = xr.open_dataset(hrrr_tmp_nc)

geospatial_data = nc.Dataset(hrrr_tmp_nc)

met_data = met_data.rio.write_crs(HRRR_AK_proj)

P_Enclosure = Polygon(read_polygon('/scratch2/NCEPDEV/ohd/Jason.Ducker/extract4dflow-master/Extract_DFlowFM_Input_Files/Polygon_enclosure_files/AK_Enclosure.pol'))

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
hrrr_nc_dir = os.path.join(dataPath,'HRRR_nc_files')
if (os.path.exists(hrrr_nc_dir) == False):
    os.mkdir(hrrr_nc_dir)

# Loop over all HRRR 3-hourly timestamps
for i in range(len(HRRR_datetimes)):

    # Get the latest HRRR file
    hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[i].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[i].strftime('%H') +'z.wrfprsf01.ak.grib2')

    hrrr_tmp_nc = os.path.join(hrrr_nc_dir,'hrrr.t'+ HRRR_datetimes[i].strftime('%Y_%m_%d_%H') +'_ak.nc')

    print(hrrr_tmp_nc)
    # Flag to catch missing HRRR files that were not available on the HPSS tape storage system
    if os.path.exists(hrrr_file):
        
        # If file exists, then create the wgrib2 command to convert
        # HRRR grib2 file into subset of data needed for SCHISM
        wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|DSWRF:surface|DLWRF:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
        # Call Linux process to execute wgrib2 command 
        exitcode = subprocess.call(wgrib2_cmd, shell=True)
    else:
        print(hrrr_file + 'is missing, must be a data gap issue for HRRR. Will perform linear interpolation on missing data gap later')
      
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

# Now finally, slice the xarray dataframe based on the actual
# user specified start and end times of their time series
# that they are requesting for HRRR Alaska data
data = data.sel(time=slice(start_time_user_dt.strftime('%Y-%m-%d %H:%M:%S'),(end_time_user_dt- datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')))

t[0:-1,:,:] = data.variables['TMP_2maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
q[0:-1,:,:] = data.variables['SPFH_2maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
u[0:-1,:,:] = data.variables['UGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
v[0:-1,:,:] = data.variables['VGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
p[0:-1,:,:] = data.variables['PRES_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
time[0:-1] = ((pd.DatetimeIndex(data.variables['time'][:].values) - ref_date).total_seconds()-start)/c2
rainrate = data.variables['APCP_surface'][:,:,:].values
for i in range(len(HRRR_datetimes_interp)):
    if(i == len(HRRR_datetimes_interp) -1):
        r[i,:] = np.sum((rainrate[-1,:,:][ind_y,:][:,ind_x].flatten()[simplex]/3600.0)*area_cor,axis=1)*precip2flux
    else:
        r[i,:] = np.sum((rainrate[i+1,:,:][ind_y,:][:,ind_x].flatten()[simplex]/3600.0)*area_cor,axis=1)*precip2flux

# add extra time step to the rest of the variables   
t[-1,:,:] = data.variables['TMP_2maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
q[-1,:,:] = data.variables['SPFH_2maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
u[-1,:,:] = data.variables['UGRD_10maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
v[-1,:,:] = data.variables['VGRD_10maboveground'][-1,:,:].values[ind_y,:][:,ind_x]
p[-1,:,:] = data.variables['PRES_surface'][-1,:,:].values[ind_y,:][:,ind_x]
time[-1] = ((pd.Timestamp(data.variables['time'][:].values[-1]) - ref_date).total_seconds()-start)/c2

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
ncout = nc.Dataset(outFile,'w',format='NETCDF4')

ncout.createDimension('time',len(time))
ncout.createDimension('ny_grid',x.shape[0])
ncout.createDimension('nx_grid',x.shape[1])

nctime = ncout.createVariable('time','f4',('time',))
nclon = ncout.createVariable('lon','f4',('ny_grid','nx_grid',))
nclat = ncout.createVariable('lat','f4',('ny_grid','nx_grid',))
ncu = ncout.createVariable('uwind','f4',('time','ny_grid','nx_grid',))
ncv = ncout.createVariable('vwind','f4',('time','ny_grid','nx_grid',))
ncp = ncout.createVariable('prmsl','f4',('time','ny_grid','nx_grid',))
nct = ncout.createVariable('stmp','f4',('time','ny_grid','nx_grid',))
ncq = ncout.createVariable('spfh','f4',('time','ny_grid','nx_grid',))

nctime[:] = time
nctime.long_name = "Time"
nctime.standard_name = "time"
nctime.units = ref
nctime.base_date = baseDate
nclon[:] = x
nclon.long_name = "Longitude"
nclon.standard_name = "longitude"
nclon.units = "degrees_east"
nclat[:] = y
nclat.long_name = "Latitude"
nclat.standard_name = "latitude"
nclat.units = "degrees_north"
ncu[:] = u
ncu.long_name = "Surface Eastward Air Velocity (10m AGL)"
ncu.standard_name = "eastward_wind"
ncu.units = "m/s"
ncv[:] = v
ncv.long_name = "Surface Northward Air Velocity (10m AGL)"
ncv.standard_name = "northward_wind"
ncv.units = "m/s"
ncp[:] = p
ncp.long_name = "Pressure reduced to MSL"
ncp.standard_name = "air_pressure_at_sea_level"
ncp.units = "Pa"
nct[:] = t
nct.long_name = "Surface Air Temperature (2m AGL)"
nct.standard_name = "air_temperature"
nct.units = "K"
ncq[:] = q
ncq.long_name = "Surface Specific Humidity (2m AGL)"
ncq.standard_name = "specific_humidity"
ncq.units = "kg/kg"

ncout.close()
