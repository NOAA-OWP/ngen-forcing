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

def DFlowFM_NWM_clip_to_netcdf(forcings,polygon,domain,output_filename):

    if forcings.is_dir():
        timesteps = sorted(forcings.glob("*DOMAIN1"))
        met_data = xr.open_dataset(domain,engine='netcdf4')
        timeseries_length = len(timesteps)
    else:
        met_data = xr.open_dataset(forcings,engine='netcdf4')
        timesteps = args.forcings
        timeseries_length = 1

    start_time = str(timesteps[0]).split('/')[-1].split('.')[0]
    start_time = start_time[0:4] + '-' + start_time[4:6] + '-' + start_time[6:8] + ' ' + start_time[8:10] + ':00:00'
    stop_time = str(timesteps[-1]).split('/')[-1].split('.')[0]
    stop_time = stop_time[0:4] + '-' + stop_time[4:6] + '-' + stop_time[6:8] + ' ' + stop_time[8:10] + ':00:00'

    NWM_Hawaii_proj = "PROJCS[\"Lambert_Conformal_Conic\",GEOGCS[\"GCS_Sphere\",DATUM[\"D_Sphere\",SPHEROID[\"Sphere\",6370000.0,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"false_easting\",0.0],PARAMETER[\"false_northing\",0.0],PARAMETER[\"central_meridian\",-157.42],PARAMETER[\"standard_parallel_1\",10.0],PARAMETER[\"standard_parallel_2\",30.0],PARAMETER[\"latitude_of_origin\",20.6],UNIT[\"Meter\",1.0]];-38489800 -26559400 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

    NWM_Alaska_proj = "PROJCS[\'Sphere_Stereographic\',GEOGCS[\'GCS_Sphere\',DATUM[\'D_Sphere\',SPHEROID[\'Sphere\',6370000.0,0.0]],PRIMEM[\'Greenwich\',0.0],UNIT[\'Degree\',0.0174532925199433]],PROJECTION[\'Stereographic_North_Pole\'],PARAMETER[\'False_Easting\',0.0],PARAMETER[\'False_Northing\',0.0],PARAMETER[\'Central_Meridian\',-135.0],PARAMETER[\'standard_parallel_1\',60.0],UNIT[\'Meter\',1.0]];-30977400 -30977400 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

    NWM_PR_proj = "PROJCS[\"Sphere_Lambert_Conformal_Conic\",GEOGCS[\"GCS_Sphere\",DATUM[\"D_Sphere\",SPHEROID[\"Sphere\",6370000.0,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"false_easting\",0.0],PARAMETER[\"false_northing\",0.0],PARAMETER[\"central_meridian\",-65.91],PARAMETER[\"standard_parallel_1\",18.1],PARAMETER[\"standard_parallel_2\",18.1],PARAMETER[\"latitude_of_origin\",18.1],UNIT[\"Meter\",1.0]];-38202500 -26634000 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

    NWM_CONUS_proj = "PROJCS[\"Lambert_Conformal_Conic\",GEOGCS[\"GCS_Sphere\",DATUM[\"D_Sphere\",SPHEROID[\"Sphere\",6370000.0,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Lambert_Conformal_Conic_2SP\"],PARAMETER[\"false_easting\",0.0],PARAMETER[\"false_northing\",0.0],PARAMETER[\"central_meridian\",-97.0],PARAMETER[\"standard_parallel_1\",30.0],PARAMETER[\"standard_parallel_2\",60.0],PARAMETER[\"latitude_of_origin\",40.0],UNIT[\"Meter\",1.0]];-35691800 -29075200 10000;-100000 10000;-100000 10000;0.001;0.001;0.001;IsHighPrecision"

    #met_data = met_data.rio.write_crs(NWM_Hawaii_proj)    

    P_Enclosure = Polygon(read_polygon(args.polygon))
    dflowfm_df = gpd.GeoDataFrame({'DFLOWFM': ['POLYGON_ENCLOSURE']}, geometry=[P_Enclosure],crs="WGS84")

    NWM_x = met_data.XLONG_M.values[0,:,:]
    NWM_y = met_data.XLAT_M.values[0,:,:]

    met_data.close()

    HRRR_df = pd.DataFrame([])
    HRRR_df['ids'] = np.arange(len(NWM_x.flatten()))
    HRRR_df['x'] = NWM_x.flatten()
    HRRR_df['y'] = NWM_y.flatten()


    NWM_gdf = gpd.GeoDataFrame(HRRR_df,geometry=gpd.points_from_xy(HRRR_df['x'],HRRR_df['y']),crs='WGS84')

    intersection = NWM_gdf.intersects(P_Enclosure)

    indices = intersection.values.reshape(NWM_x.shape)
    indices = indices[:,np.all(indices==True,axis=0)]

    pres_data = np.empty((timeseries_length,indices.shape[0],indices.shape[-1]),dtype=float)
    rainfall_data = np.empty((timeseries_length,indices.shape[0],indices.shape[-1]),dtype=float)
    u10_data = np.empty((timeseries_length,indices.shape[0],indices.shape[-1]),dtype=float)
    v10_data = np.empty((timeseries_length,indices.shape[0],indices.shape[-1]),dtype=float)

    print('Loading netcdf data to arrays for manipulation')
    for i in range(timeseries_length):
        nc_data = nc.Dataset(timesteps[i])
        pres_data[i,:,:] = nc_data.variables['PSFC'][:][0,:,:].flatten().reshape(indices.shape)
        rainfall_data[i,:,:] = nc_data.variables['RAINRATE'][:][0,:,:].flatten().reshape(indices.shape)*3600
        u10_data[i,:,:] = nc_data.variables['U2D'][:][0,:,:].flatten().reshape(indices.shape)
        v10_data[i,:,:] = nc_data.variables['V2D'][:][0,:,:].flatten().reshape(indices.shape)
        ws = np.sqrt(u10_data[i,:,:]**2+v10_data[i,:,:]**2)
    nc_data.close()

    print('Creating D-FlowFM netcdf configured meterological forcing file')
    # open a netCDF file to write
    ncout = nc.Dataset(output_filename,'w',format='NETCDF4')
    # Generate timestamp
    timestamps = pd.date_range(start=start_time, end=stop_time, freq='H')
    time_netcdf = timestamps -  pd.to_datetime(start_time)
    time_netcdf = np.array(time_netcdf.total_seconds(),dtype='i8')
    time_units = 'seconds since ' + start_time
    # define axis size
    ncout.createDimension('time',timeseries_length)
    ncout.createDimension('latitude',indices.shape[0])
    ncout.createDimension('longitude',indices.shape[1])
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
    nclat[:] = np.mean(NWM_y,axis=1)
    nclon[:] = np.mean(NWM_x,axis=0)
    nc_pres[:] = pres_data
    nc_rainfall[:] = rainfall_data*24
    nc_u10[:] = u10_data
    nc_v10[:] = v10_data

    ncout.close()



def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forcing',dest='forcings', type=pathlib.Path, required=True, help='Path of forcings inputs to regrid.')
    parser.add_argument("--clip", dest="polygon", required=True, type=pathlib.Path, help="Use a polygon to select sites for output")
    parser.add_argument("--domain", dest="domain", required=True, type=pathlib.Path, help="NWM geo_em.nc file")
    parser.add_argument('--output_file', dest='output_file',type=pathlib.Path, required=True, help="Output file pathway")
    args = parser.parse_args()
    args.output = args.output.resolve()
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
    DFlowFM_NWM_clip_to_netcdf(args.forcings,args.polygon,args.domain,args.output)
    
    print("Finished clipping all forcing inputs into DFlowFM netcdf file")

if __name__ == "__main__":
    args = get_options()
    main(args)
