import argparse
import pathlib
import time
import json
import datetime
import xarray as xr
import numpy as np
import cftime
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import pandas as pd
from common.io import read_pli, read_polygon
import netCDF4 as nc
import subprocess
import os
import glob
import shutil

def DFlowFM_HRRR_clip_to_netcdf(start,stop,dataPath,polygon,output_filename):

    # create datetimes for user specified start and end time
    start_time_user_dt = start
    end_time_user_dt = stop

    # Initialize start and end time of user
    # so search algorithim below can find nearest
    # 3HR HRRR cycle needed to fill the data gaps
    start_time_user = start.strftime('%Y-%m-%d %H:%M:%S')
    end_time_user = stop.strftime('%Y-%m-%d %H:%M:%S')

    start_time = start.strftime('%Y-%m-%d %H:%M:%S')
    end_time = stop.strftime('%Y-%m-%d %H:%M:%S')

    start_hour = int(start_time[11:13])

    if(start_hour == 0):
        start_time_tmp = (start - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
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
    hrrr_tmp_nc = os.path.join(dataPath,'HRRR_tmp.nc')
    # Execute the wgrib2 command to convert the HRRR grib2 file into a netcdf file
    ############ Remember to set the Linux $WGRIB2 environmental variable to the wgrib2 exectuable pathway ###############
    wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|DSWRF:surface|DLWRF:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
    # Use subprocess module to execute the linux wgrib2 command and create the HRRR netcdf file
    exitcode = subprocess.call(wgrib2_cmd, shell=True)



    geospatial_data = nc.Dataset(hrrr_tmp_nc)
   

    timeseries_length = len(HRRR_datetimes_interp)

    met_data = xr.open_dataset(hrrr_tmp_nc)

    HRRR_proj = 'PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",60],PARAMETER["central_meridian",225],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]'

    met_data = met_data.rio.write_crs(HRRR_proj)

    P_Enclosure = Polygon(read_polygon(args.polygon))

    dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

    #dflowfm_df = dflowfm_df.to_crs(HRRR_proj)

    #met_data = met_data.rio.clip(dflowfm_df.geometry.apply(mapping), dflowfm_df.crs, all_touched = True)

    HRRR_x = met_data.x.values
    HRRR_y = met_data.y.values

    met_data.close()

    x, y = np.meshgrid(HRRR_x,HRRR_y)

    HRRR_lats = geospatial_data.variables['latitude'][:].data
    HRRR_lons = (geospatial_data.variables['longitude'][:].data + 180) % 360 -180

    geospatial_data.close()

    os.remove(hrrr_tmp_nc)

    HRRR_df = pd.DataFrame([])
    HRRR_df['ids'] = np.arange(len(HRRR_lats.flatten()))
    HRRR_df['x'] = HRRR_lons.flatten()
    HRRR_df['y'] = HRRR_lats.flatten()
    HRRR_df['HRRR_x'] = x.flatten()
    HRRR_df['HRRR_y'] = y.flatten()


    HRRR_gdf = gpd.GeoDataFrame(HRRR_df,geometry=gpd.points_from_xy(HRRR_df['x'],HRRR_df['y']),crs='WGS84')

    #HRRR_gdf = HRRR_gdf.to_crs('WGS84')

    intersection = HRRR_gdf.intersects(P_Enclosure)
    ind_x = np.where(np.in1d(HRRR_x,HRRR_gdf.HRRR_x[intersection].values)==True)[0]
    ind_y = np.where(np.in1d(HRRR_y,HRRR_gdf.HRRR_y[intersection].values)==True)[0]

    clip_lats = HRRR_gdf[intersection].groupby(HRRR_gdf.HRRR_y).y.mean().values
    clip_lons = HRRR_gdf[intersection].groupby(HRRR_gdf.HRRR_x).x.mean().values


    pres_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    rainfall_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    u10_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    v10_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)

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


    pres_data[:,:,:] = data.variables['PRES_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    rainfall_data[:,:,:] = data.variables['APCP_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    u10_data[:,:,:] = data.variables['UGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    v10_data[:,:,:] = data.variables['VGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]

    data.close()


    # Remove the HRRR netcdf file diectory
    # after we're finished with the files
    if os.path.exists(hrrr_nc_dir):
        shutil.rmtree(hrrr_nc_dir,ignore_errors=True)

    print('Creating D-FlowFM netcdf configured meterological forcing file')
    # open a netCDF file to write
    ncout = nc.Dataset(output_filename,'w',format='NETCDF4')
    # Generate timestamp
    time_netcdf = HRRR_datetimes_interp - HRRR_datetimes_interp[0]
    time_netcdf = np.array(time_netcdf.total_seconds(),dtype='i8')
    time_units = 'seconds since ' + start.strftime('%Y-%m-%d %H:%M:%S')
    # define axis size
    ncout.createDimension('time',timeseries_length)
    ncout.createDimension('latitude',len(clip_lats))
    ncout.createDimension('longitude',len(clip_lons))
    # create time axis
    nctime = ncout.createVariable('time','i8',('time',))
    nctime.setncattr('units',time_units)
    # create latitude
    nclat = ncout.createVariable('latitude','f8',('latitude',))
    nclat.setncattr('standard_name','latitude')
    nclat.setncattr('short_name','lat')
    nclat.setncattr('units','degree_north')
    # create longitude
    nclon = ncout.createVariable('longitude','f8',('longitude',))
    nclon.setncattr('standard_name','longitude')
    nclon.setncattr('short_name','long')
    nclon.setncattr('units','degree_east')

    # create pressure fields
    nc_pres = ncout.createVariable('msl','f8',('time','latitude','longitude',))
    nc_pres.setncattr('standard_name','air_pressure')
    nc_pres.setncattr('short_name','msl')
    nc_pres.setncattr('level','Mean sea level')
    nc_pres.setncattr('units','Pa')

    # create rainfall fields
    nc_rainfall = ncout.createVariable('rainfall','f8',('time','latitude','longitude',))
    nc_rainfall.setncattr('standard_name','rainfall_rate')
    nc_rainfall.setncattr('level','Surface')
    nc_rainfall.setncattr('units','mm/day')

    # create u10 fields
    nc_u10 = ncout.createVariable('u10','f8',('time','latitude','longitude',))
    nc_u10.setncattr('standard_name','eastward_wind')
    nc_u10.setncattr('short_name','u10')
    nc_u10.setncattr('level','10 m above ground')
    nc_u10.setncattr('units','m s-1')

    # create v10 fields
    nc_v10 = ncout.createVariable('v10','f8',('time','latitude','longitude',))
    nc_v10.setncattr('standard_name','northward_wind')
    nc_v10.setncattr('short_name','v10')
    nc_v10.setncattr('level','10 m above ground')
    nc_v10.setncattr('units','m s-1')

    # copy axis from original dataset
    nctime[:] = time_netcdf
    nclat[:] = clip_lats
    nclon[:] = clip_lons
    nc_pres[:] = pres_data
    nc_rainfall[:] = rainfall_data*24
    nc_u10[:] = u10_data
    nc_v10[:] = v10_data

    ncout.close()




def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forcing',dest='forcings', type=pathlib.Path, required=True, help='Path of forcings inputs to regrid.')
    parser.add_argument("--clip", dest="polygon", required=True, type=pathlib.Path, help="Use a polygon to select sites for output")
    parser.add_argument('--start', dest='start_time', required=True,
                    help='The date and time to begin making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--stop', dest='stop_time', required=True,
                    help='The date and time to stop making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--output_file', dest='output',type=pathlib.Path, required=True, help="Directory to write output")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
    args.stop_time = datetime.datetime.strptime(args.stop_time, '%Y-%m-%d_%H:%M:%S')
    return args

def main(args):

    DFlowFM_HRRR_clip_to_netcdf(args.start_time,args.stop_time,args.forcings,args.polygon,args.output)

    print("Finished clipping all HRRR forcing inputs into DFlowFM netcdf file")

if __name__ == "__main__":
    args = get_options()
    main(args)

