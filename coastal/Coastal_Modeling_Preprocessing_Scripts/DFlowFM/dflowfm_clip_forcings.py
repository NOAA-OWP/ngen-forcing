import argparse
import pathlib
import time
import json

import xarray as xr
import numpy as np
import cftime
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import pandas as pd
from common.io import read_pli, read_polygon
import netCDF4 as nc
import datetime
import glob
import os

def DFlowFM_AORC_netcdf_configuration(ds,start_time,stop_time,output_filename):
    ds = ds.drop_vars(["DLWRF_surface","DSWRF_surface","SPFH_2maboveground","TMP_2maboveground"])
    ds['msl'] = ds.PRES_surface
    ds['msl'].attrs['standard_name'] = "air_pressure"
    ds['msl'].attrs['level'] = "Mean sea level"
    ds['msl'].attrs['short_name'] = "msl"
    ds['msl'].attrs['units'] = "Pa"
    ds['u10'] = ds.UGRD_10maboveground
    ds['u10'].attrs['long_name'] = "eastward_wind"
    ds['u10'].attrs['standard_name'] = "eastward_wind"
    ds['u10'].attrs['short_name'] = "u10"
    ds['u10'].attrs['units'] = "m s-1"
    ds['u10'].attrs['level'] = "10 m above ground"
    ds['v10'] = ds.VGRD_10maboveground
    ds['v10'].attrs['long_name'] = "northward_wind"
    ds['v10'].attrs['standard_name'] = "northward_wind"
    ds['v10'].attrs['level'] = "10 m above ground"
    ds['v10'].attrs['short_name'] = "v10"
    ds['v10'].attrs['units'] = "m s-1"
    ds['rainfall'] = ds['APCP_surface']/3600.0
    ds['rainfall'].attrs['standard_name'] = 'rainfall_rate'
    ds['rainfall'].attrs['long_name'] = 'Precipitation Rate at surface'
    ds['rainfall'].attrs['level'] = 'Surface'
    time_units = 'hours since ' + start_time
    ds = ds.drop_vars(["APCP_surface","PRES_surface","UGRD_10maboveground","VGRD_10maboveground","time"])
    timestamps = pd.date_range(start=start_time, end=stop_time, freq='H')
    ds.assign_coords(time=timestamps)
    ds.time.encoding['units'] = time_units
    ds['time'] = ds.time
    ds['time'].attrs['units'] = time_units
    ds.to_netcdf(output_filename)

def DFlowFM_AORC_clip_to_netcdf(forcings,start_time,stop_time,polygon,output_filename):

    if forcings.is_dir():
        files = glob.glob(os.path.join(forcings,"*.nc*"))
        files.sort()
        AORC_datetimes = pd.date_range(start=start_time.strftime('%Y-%m-%d %H:%M:%S'),end=stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='h')
        AORC_indices = []
        for i in range(len(AORC_datetimes)):
            AORC_indices.append([idx for idx, s in enumerate(files) if AORC_datetimes[i].strftime('%Y%m%d%H') in s][0])

        timesteps = np.array(files)[AORC_indices]
        met_data = xr.open_dataset(timesteps[0])
        timeseries_length = len(timesteps)
    else:
        met_data = xr.open_dataset(forcings)
        timesteps = args.forcings
        timeseries_length = 1

    met_data = met_data.rio.write_crs('WGS84')

    P_Enclosure = Polygon(read_polygon(args.polygon))

    dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

    met_data = met_data.rio.clip(dflowfm_df.geometry.apply(mapping), dflowfm_df.crs, all_touched = True)

    AORC_data = xr.open_dataset(timesteps[0])
    AORC_lats = AORC_data.latitude.values
    AORC_lons = AORC_data.longitude.values
    AORC_data.close()

    clip_lats = met_data.latitude.values
    clip_lons = met_data.longitude.values

    idx_lats = np.in1d(AORC_lats,clip_lats)
    idx_lons = np.in1d(AORC_lons,clip_lons)

    pres_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    rainfall_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    u10_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)
    v10_data = np.empty((timeseries_length,len(clip_lats),len(clip_lons)),dtype=float)

    print('Loading netcdf data to arrays for manipulation')
    for i in range(timeseries_length):
        nc_data = nc.Dataset(timesteps[i])
        pres_data[i,:,:] = nc_data.variables['PRES_surface'][:][0,idx_lats,:][:,idx_lons]
        rainfall_data[i,:,:] = nc_data.variables['APCP_surface'][:][0,idx_lats,:][:,idx_lons]
        u10_data[i,:,:] = nc_data.variables['UGRD_10maboveground'][:][0,idx_lats,:][:,idx_lons]
        v10_data[i,:,:] = nc_data.variables['VGRD_10maboveground'][:][0,idx_lats,:][:,idx_lons]

    nc_data.close()

    print('Creating D-FlowFM netcdf configured meterological forcing file')
    # open a netCDF file to write
    ncout = nc.Dataset(output_filename,'w',format='NETCDF4')
    # Generate timestamp
    time_netcdf = AORC_datetimes -  pd.to_datetime(start_time)
    time_netcdf = np.array(time_netcdf.total_seconds(),dtype='i8')
    time_units = 'seconds since ' + start_time.strftime('%Y-%m-%d %H:%M:%S')
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
  

    ################################## OLDER METHOD WITH XARRAY, DOESNT SAVE DATA CORRECTLY WITH OPEN_MFDATASET #############################
    #if args.forcings.is_dir():
    #    timesteps = sorted(args.forcings.glob("*.nc4"))
    #    met_data = xr.open_mfdataset(timesteps)
    #else:
    #    met_data = xr.open_dataset(args.forcings)
    #    timesteps = args.forcings

    #start_time = str(timesteps[0]).split('/')[-1].split('_')[-1].split('z')[0]
    #start_time = start_time[0:4] + '-' + start_time[4:6] + '-' + start_time[6:8] + ' ' + start_time[8:10] + ':00:00'
    #stop_time = str(timesteps[-1]).split('/')[-1].split('_')[-1].split('z')[0]
    #stop_time = stop_time[0:4] + '-' + stop_time[4:6] + '-' + stop_time[6:8] + ' ' + stop_time[8:10] + ':00:00'

    #met_data = met_data.rio.write_crs('WGS84')

    #P_Enclosure = Polygon(read_polygon(args.polygon))

    #dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

    #met_data = met_data.rio.clip(dflowfm_df.geometry.apply(mapping), dflowfm_df.crs, all_touched = True)

    #met_data = met_data.drop_vars(["DLWRF_surface","DSWRF_surface","SPFH_2maboveground","TMP_2maboveground"])
    #comp = dict(zlib=True, complevel=1, shuffle=True, _FillValue=-99999.)
    #encoding = {var: comp for var in ['APCP_surface','PRES_surface','UGRD_10maboveground','VGRD_10maboveground']}
    #met_data.to_netcdf(encoding=encoding)
    #DFlowFM_AORC_netcdf_configuration(met_data,start_time,stop_time,args.output)
    ############################################################################################################################################

    #if(str(args.model).upper() == 'AORC'):
    DFlowFM_AORC_clip_to_netcdf(args.forcings,args.start_time,args.stop_time,args.polygon,args.output)
    
    print("Finished clipping all forcing inputs into DFlowFM netcdf file")

if __name__ == "__main__":
    args = get_options()
    main(args)
