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

def DFlowFM_HRRR_clip_to_netcdf(start,stop,dataPath,polygon,output_filename,output):

    HRRR_datetimes = pd.date_range(start=start,end=stop,freq='H')[0:-1]

    subhourly = False
    # Specify where a Alaska HRRR raw grib2 file is on your system
    hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[0].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[0].strftime('%H') +'z.wrfsfcf01.grib2')
    if(os.path.exists(hrrr_file) == False):
        hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[0].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[0].strftime('%H') +'z.wrfprsf01.grib2')
        if(os.path.exists(hrrr_file) == False):
            hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[0].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[0].strftime('%H') +'z.wrfsubhf01.grib2')
            subhourly = True

    hrrr_tmp_nc = os.path.join(dataPath,'HRRR_tmp.nc')
    # Execute the wgrib2 command to convert the HRRR grib2 file into a netcdf file
    ############ Remember to set the Linux $WGRIB2 environmental variable to the wgrib2 exectuable pathway ###############
    if(subhourly):
        wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground:15 min fcst:|VGRD:10 m above ground:15 min fcst:|TMP:2 m above ground:15 min fcst:|DTMP:2 m above ground:15 min fcst:|PRATE:surface:15 min fcst:|PRES:surface:15 min fcst:)" {hrrr_file} -netcdf {hrrr_tmp_nc}'
    else:
        wgrib2_cmd = f'$WGRIB2 -match "(UGRD:10 m above ground|VGRD:10 m above ground|TMP:2 m above ground|SPFH:2 m above ground|APCP:surface|PRES:surface)" {hrrr_file} -netcdf {hrrr_tmp_nc}'

    exitcode = subprocess.call(wgrib2_cmd, shell=True)
    
    geospatial_data = nc.Dataset(hrrr_tmp_nc)
   

    timeseries_length = len(HRRR_datetimes)

    met_data = xr.open_dataset(hrrr_tmp_nc)

    HRRR_proj = 'PROJCS["unnamed",GEOGCS["Coordinate System imported from GRIB file",DATUM["unnamed",SPHEROID["Sphere",6371229,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["latitude_of_origin",38.5],PARAMETER["central_meridian",-97.5],PARAMETER["standard_parallel_1",38.5],PARAMETER["standard_parallel_2",38.5],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Metre",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'

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
    hrrr_nc_dir = os.path.join(output,'HRRR_nc_files')
    if (os.path.exists(hrrr_nc_dir) == False):
        os.mkdir(hrrr_nc_dir)

    for i in range(len(HRRR_datetimes)):
        file_avail = True
        subhourly = False
        hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[i].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[i].strftime('%H') +'z.wrfsfcf01.grib2')
        if(os.path.exists(hrrr_file) == False):
            hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[i].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[i].strftime('%H') +'z.wrfprsf01.grib2')
            if(os.path.exists(hrrr_file) == False):
                hrrr_file = os.path.join(dataPath,'hrrr.'+HRRR_datetimes[i].strftime('%Y%m%d') +'/' + 'hrrr.t'+ HRRR_datetimes[i].strftime('%H') +'z.wrfsubhf01.grib2')
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

    
    pres_data[:,:,:] = data.variables['PRES_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    rainfall_data[:,:,:] = data.variables['APCP_surface'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    u10_data[:,:,:] = data.variables['UGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]
    v10_data[:,:,:] = data.variables['VGRD_10maboveground'][:,:,:].values[:,ind_y,:][:,:,ind_x]

    data.close()

    # Remove the HRRR netcdf file diectory
    # after we're finished with the files
    if os.path.exists(hrrr_nc_dir):
        shutil.rmtree(hrrr_nc_dir)

    print('Creating D-FlowFM netcdf configured meterological forcing file')
    # open a netCDF file to write
    ncout = nc.Dataset(output_filename,'w',format='NETCDF4')
    # Generate timestamp
    time_netcdf = HRRR_datetimes - HRRR_datetimes[0]
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
    parser.add_argument('--output_file', dest='output_file',type=pathlib.Path, required=True, help="Directory to write output")
    parser.add_argument('--output_HRRR',dest='output', type=pathlib.Path, required=True, help='Path to output directory for HRRR files')

    args = parser.parse_args()
    args.output = args.output.resolve()
    args.start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
    args.stop_time = datetime.datetime.strptime(args.stop_time, '%Y-%m-%d_%H:%M:%S')
    return args

def main(args):

    DFlowFM_HRRR_clip_to_netcdf(args.start_time,args.stop_time,args.forcings,args.polygon,args.output_file,args.output)

    print("Finished clipping all HRRR forcing inputs into DFlowFM netcdf file")

if __name__ == "__main__":
    args = get_options()
    main(args)

