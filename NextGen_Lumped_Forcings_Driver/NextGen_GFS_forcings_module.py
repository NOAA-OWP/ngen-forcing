import numpy as np
import geopandas as gpd
import netCDF4 as nc4
import os
import glob
from pathlib import Path
from os.path import join
import argparse
import math
from math import tau as TWO_PI
import pandas as pd
from  multiprocessing import Process, Lock, Queue
import multiprocessing
import time
import datetime
import re
import gc
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, CoverageProcessor, CoverageWriter, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
from osgeo import gdal


def create_ngen_netcdf(final_df,netcdf_dir,num_catchments,num_files):
    """
    Create NextGen netcdf file with specified format
    """

    # Get start time from first instance of final dataframe
    start_time = (pd.Timestamp("1970-01-01 00:00:00") + pd.Timedelta(seconds=final_df.time.values[0])).strftime("%Y%m%d%H")

    #create output netcdf file name
    output_netcdf = join(netcdf_dir, "NextGen_GFS_forcing_"+start_time+".nc")

     
    # write data to netcdf files
    ncfile_out = nc4.Dataset(output_netcdf, 'w', format='NETCDF4')


    #add the dimensions
    time_dim = ncfile_out.createDimension('time', None)
    catchment_id_dim = ncfile_out.createDimension('catchment-id', num_catchments)
    string_dim =ncfile_out.createDimension('str_dim', 1)

    # create variables
    cat_id_out = ncfile_out.createVariable('ids', 'str', ('catchment-id'), fill_value="None")
    time_out = ncfile_out.createVariable('Time', 'double', ('catchment-id','time',), fill_value=-99999)
    PRATE_surface_out = ncfile_out.createVariable('RAINRATE', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    TMP_2maboveground_out = ncfile_out.createVariable('T2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    SPFH_2maboveground_out = ncfile_out.createVariable('Q2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    UGRD_10maboveground_out = ncfile_out.createVariable('U2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    VGRD_10maboveground_out = ncfile_out.createVariable('V2D', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    PRES_surface_out = ncfile_out.createVariable('PSFC', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DSWRF_surface_out = ncfile_out.createVariable('SWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    DLWRF_surface_out = ncfile_out.createVariable('LWDOWN', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)

    #set attributes for variables
    setattr(cat_id_out, 'description', 'catchment_id')
    setattr(time_out, "units", "seconds")
    setattr(time_out, "epoch_start", "01/01/1970 00:00:00")
    setattr(PRATE_surface_out, "long_name", "Precipitation Rate")
    setattr(PRATE_surface_out, "short_name", "RAINRATE")
    setattr(PRATE_surface_out, "units", "mm s-1")
    setattr(PRATE_surface_out, "level", "surface")
    setattr(TMP_2maboveground_out, "long_name", "Temperature")
    setattr(TMP_2maboveground_out, "short_name", "TMP_2maboveground")
    setattr(TMP_2maboveground_out, "units", "K")
    setattr(TMP_2maboveground_out, "level", "2 m above ground")
    setattr(SPFH_2maboveground_out, "long_name", "Specific Humidity")
    setattr(SPFH_2maboveground_out, "short_name", "SPFH_2maboveground")
    setattr(SPFH_2maboveground_out, "units", "kg/kg")
    setattr(SPFH_2maboveground_out, "level", "2 m above ground")
    setattr(UGRD_10maboveground_out, "long_name", "U-Component of Wind")
    setattr(UGRD_10maboveground_out, "short_name", "UGRD_10maboveground")
    setattr(UGRD_10maboveground_out, "units", "m/s")
    setattr(UGRD_10maboveground_out, "level", "10 m above ground")
    setattr(VGRD_10maboveground_out, "long_name", "V-Component of Wind")
    setattr(VGRD_10maboveground_out, "short_name", "VGRD_10maboveground")
    setattr(VGRD_10maboveground_out, "units", "m/s")
    setattr(VGRD_10maboveground_out, "level", "10 m above ground")
    setattr(PRES_surface_out, "long_name", "Pressure")
    setattr(PRES_surface_out, "short_name", "PRES_surface")
    setattr(PRES_surface_out, "units", "Pa")
    setattr(PRES_surface_out, "level", "surface")
    setattr(DSWRF_surface_out, "long_name", "Downward Short-Wave Rad. Flux")
    setattr(DSWRF_surface_out, "short_name", "DSWRF_surface")
    setattr(DSWRF_surface_out, "units", "W/m^2")
    setattr(DSWRF_surface_out, "level", "surface")
    setattr(DLWRF_surface_out, "long_name", "Downward Long-Wave Rad. Flux")
    setattr(DLWRF_surface_out, "short_name", "DLWRF_surface")
    setattr(DLWRF_surface_out, "units", "W/m^2")
    setattr(DLWRF_surface_out, "level", "surface")



    cat_id_out[:] = final_df['cat-id'].unique()
    time_out[:,:] = np.reshape(final_df['time'].values,(num_catchments,num_files))
    PRATE_surface_out[:,:] = np.reshape(final_df['PRATE_surface'].values,(num_catchments,num_files))
    DLWRF_surface_out[:,:] = np.reshape(final_df['DLWRF_surface'].values,(num_catchments,num_files))
    DSWRF_surface_out[:,:] = np.reshape(final_df['DSWRF_surface'].values,(num_catchments,num_files))
    PRES_surface_out[:,:] = np.reshape(final_df['PRES_surface'].values,(num_catchments,num_files))
    SPFH_2maboveground_out[:,:] = np.reshape(final_df['SPFH_2maboveground'].values,(num_catchments,num_files))
    TMP_2maboveground_out[:,:] = np.reshape(final_df['TMP_2maboveground'].values,(num_catchments,num_files))
    UGRD_10maboveground_out[:,:] = np.reshape(final_df['UGRD_10maboveground'].values,(num_catchments,num_files))
    VGRD_10maboveground_out[:,:] = np.reshape(final_df['VGRD_10maboveground'].values,(num_catchments,num_files))

    # Now close NextGen netcdf file
    ncfile_out.close()

def create_ngen_csv_catchments(final_df, num_processes, csv_dir, num_catchments,num_forcing_files):
    """
    Create NextGen csv files with specified format
    """

    final_df['Time'] = pd.Timestamp("1970-01-01 00:00:00") + pd.TimedeltaIndex(final_df['time'].values,'s')

    #generate the data objects for child processes and partition out the data based on catchment id and timeseries
    id_groups = np.array_split(np.reshape(final_df['cat-id'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    time_groups = np.array_split(np.reshape(final_df['Time'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    precip_groups = np.array_split(np.reshape(final_df['PRATE_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    q_groups = np.array_split(np.reshape(final_df['SPFH_2maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    tmp_groups = np.array_split(np.reshape(final_df['TMP_2maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    ugrd_groups = np.array_split(np.reshape(final_df['UGRD_10maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    vgrd_groups = np.array_split(np.reshape(final_df['VGRD_10maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    lw_groups = np.array_split(np.reshape(final_df['DLWRF_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    sw_groups = np.array_split(np.reshape(final_df['DSWRF_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
    pres_groups = np.array_split(np.reshape(final_df['PRES_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)


    # Delete main data at this point to save RAM
    del(final_df)

    process_data = []
    process_list = []
    lock = Lock()

    # Collect garbage from main thread after partitioning data
    gc.collect()

    for i in range(num_processes):
        # fill the dictionary with needed at
        data = {}
        data["cat_ids"] = id_groups[i]
        data["Time"] = time_groups[i]
        data["RAINRATE"] = precip_groups[i]
        data["Q2D"] = q_groups[i]
        data["T2D"] = tmp_groups[i]
        data["U2D"] = ugrd_groups[i]
        data["V2D"] = vgrd_groups[i]
        data["LWDOWN"] = lw_groups[i]
        data["SWDOWN"] = sw_groups[i]
        data["PSFC"] = pres_groups[i]

        #append to the list
        process_data.append(data)

        p = Process(target=process_csv_ids, args=(data, lock, i, csv_dir))

        process_list.append(p)

    # Delete variables to save RAM
    del(id_groups)
    del(time_groups)
    del(precip_groups)
    del(q_groups)
    del(tmp_groups)
    del(ugrd_groups)
    del(vgrd_groups)
    del(lw_groups)
    del(sw_groups)
    del(pres_groups)

    # collect garbage from threads to save RAM
    gc.collect()

    #start all processes
    for p in process_list:
        p.start()

    #wait for termination
    for p in process_list:
        p.join()

def process_csv_ids(data : dict, lock: Lock, num: int, csv_dir):
    # Get the number of unique catchment ids in each thread
    cat_ids = np.unique(data['cat_ids'][:,0])
    num_cats = len(cat_ids)

    print("Thread " + str(num) + " array shape is " + str(len(data['cat_ids'])) + " and number of unique ids is " + str(len(cat_ids)))
    # Loop through each catchment and create/save csv
    for i in range(num_cats):
        csv_df = pd.DataFrame([])
        csv_df['Time'] = data['Time'][i,:]
        csv_df['RAINRATE'] = data["RAINRATE"][i,:]
        csv_df['Q2D'] = data['Q2D'][i,:]
        csv_df['T2D'] = data['T2D'][i,:]
        csv_df['U2D'] = data['U2D'][i,:]
        csv_df['V2D'] = data['V2D'][i,:]
        csv_df['LWDOWN'] = data['LWDOWN'][i,:]
        csv_df['SWDOWN'] = data['SWDOWN'][i,:]
        csv_df['PSFC'] = data['PSFC'][i,:]
        csv_df = csv_df.sort_values(by=['Time'])
        NextGen_csv = join(csv_dir,str(data['cat_ids'][i,0])+'.csv')
        csv_df.to_csv(NextGen_csv,index=False)

        if(i == 0):
            csv_length = len(csv_df)
        else:
            if(len(csv_df) != csv_length):
                print(cat_ids[i] + ' likely missing some of time series')

        print(str((i+1)/num_cats*100) + '% complete updating catchment csvs')
    # collect the garbage within the thread before sending results back to main
    # thread to maximize our RAM as much as possible
    gc.collect()

def GFS_bias_correction(dataset, hour, current_output_step):

    # Perform bias correction for incoming short wave radiation
    dataset['DLWRF_surface'] = gfs_lwdown_bias_correction(dataset['DLWRF_surface'].values, hour, current_output_step)

   
    # Perform bias correction for 2m Air Temperature
    dataset['TMP_2maboveground'] = gfs_tmp_bias_correction(dataset['TMP_2maboveground'].values, hour, current_output_step)

    # Perform bias correction for u and v wind components collectively
    dataset['UGRD_10maboveground'], dataset['VGRD_10maboveground'] = gfs_wspd_bias_correction(dataset['UGRD_10maboveground'].values, dataset['VGRD_10maboveground'].values, hour, current_output_step)
  
    return dataset
    
# GFS temperature bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_tmp_bias_correction(tmp_data, hour, current_output_step):
    hh = hour
    net_bias_mr = -0.18
    fhr_mult_mr = 0.002
    diurnal_ampl_mr = -1.4
    diurnal_offs_mr = -2.1

    fhr = current_output_step

    bias_corr = net_bias_mr + fhr_mult_mr * fhr + diurnal_ampl_mr * math.sin(diurnal_offs_mr + hh / 24 * TWO_PI)
    tmp_data = tmp_data + bias_corr

    return tmp_data

# GFS Incoming longwave radiation bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_lwdown_bias_correction(lwdown_data, hour, current_output_step):

    hh = hour
    fhr = current_output_step

    lwdown_net_bias_mr = 9.9
    lwdown_fhr_mult_mr = 0.00
    lwdown_diurnal_ampl_mr = -1.5
    lwdown_diurnal_offs_mr = 2.8

    bias_corr = lwdown_net_bias_mr + lwdown_fhr_mult_mr * fhr + lwdown_diurnal_ampl_mr * \
                math.sin(lwdown_diurnal_offs_mr + hh / 24 * TWO_PI)

    lwdown_data = lwdown_data + bias_corr

    return lwdown_data

# GFS wind speed bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def gfs_wspd_bias_correction(ugrd, vgrd, hour, current_output_step):

    hh = hour
    fhr = current_output_step
    wspd_net_bias_mr = -0.20
    wspd_fhr_mult_mr = 0.00
    wspd_diurnal_ampl_mr = -0.32
    wspd_diurnal_offs_mr = -1.1

    # need to get wind speed from U, V components
    wdir = np.arctan2(vgrd, ugrd)
    wspd = np.sqrt(np.square(ugrd) + np.square(vgrd))

    bias_corr = wspd_net_bias_mr + wspd_fhr_mult_mr * fhr + \
                wspd_diurnal_ampl_mr * math.sin(wspd_diurnal_offs_mr + hour / 24 * TWO_PI)

    wspd = wspd + bias_corr
    wspd = np.where(wspd < 0, 0, wspd)
    # Now seperate wind speed components and return back 
    ugrid_out = wspd * np.cos(wdir)
    vgrid_out = wspd * np.sin(wdir)

    return ugrid_out, vgrid_out


    
def GFS_downscaling(csv_results,timestamp,NextGen_catchment_features):
    
    # Extract regridded GFS model elevation to use as
    # input in the GFS downscaling functions
    elev = csv_results['HGT_surface'].values

    # Perform downscaling on GFS temperature and reassign 
    # data to pandas dataframe
    tmp_2m = csv_results['TMP_2maboveground'].values
    tmp_downscaled = gfs_temp_downscaling_simple_lapse(tmp_2m, elev,NextGen_catchment_features)
    csv_results['TMP_2maboveground'] = tmp_downscaled

    # Perform downscaling on GFS surface pressure and reassign
    # data to pandas dataframe
    pres_old = csv_results['PRES_surface'].values
    pres_downscaled = gfs_pres_downscaling_classic(pres_old,tmp_downscaled,elev,NextGen_catchment_features)
    csv_results['PRES_surface'] = pres_downscaled

    # Perform downscaling on GFS 2m specific humidity and reassign
    # data to pandas dataframe
    tmpHumidity = csv_results['SPFH_2maboveground'].values
    humidity_downscaled = gfs_hum_downscaling_classic(tmpHumidity,tmp_downscaled,tmp_2m,pres_old)
    csv_results['SPFH_2maboveground'] = humidity_downscaled

    
    # Perform downscaling on GFS incoming short wave radiation and reassign
    # data to pandas dataframe
    dswr = csv_results['DSWRF_surface'].values
    dswr_downscaled = gfs_dswr_downscaling_topo_adj(dswr, timestamp,NextGen_catchment_features)
    csv_results['DSWRF_surface'] = dswr_downscaled

    return csv_results

# GFS Temperature downscaling function based on simple lapse rate derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_temp_downscaling_simple_lapse(tmp_in, elev,NextGen_catchment_features):
    """
    Function that applies a single lapse rate adjustment to modeled
    2-meter temperature by taking the difference of the native
    input elevation and the WRF-hydro elevation.
    :param tmp_in:
    :param elev:
    :param NextGen_catchment_features: globally available
    :return: air_pressure: downscaled surface air pressure
    """

    tmp2m = tmp_in

    elevDiff = elev - NextGen_catchment_features.elevation.values

    # Apply single lapse rate value to the input 2-meter
    # temperature values.
    tmp2m = tmp2m + (6.49/1000.0)*elevDiff

    return tmp2m

# GFS pressure downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_pres_downscaling_classic(pres_in, tmp2m, elev,NextGen_catchment_features):
    """
    Generic function to downscale surface pressure to the WRF-Hydro domain.
    :param : pres_in
    :param : tmp2m -- temperature
    :param : elev -- GFS elevation grid
    :param NextGen_catchment_features:  globally available
    :return: pres_downscaled - downscaled surface air pressure
    """
    elevDiff = elev - NextGen_catchment_features.elevation.values
    pres_downscaled = pres_in + (pres_in*elevDiff*9.8)/(tmp2m*287.05)

    return  pres_downscaled

# GFS specific humidity downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_hum_downscaling_classic(tmpHumidity, tmp2m, t2dTmp, psfcTmp):
    """
    Function for downscaling 2-meter specific humidity using already downscaled
    2-meter temperature, unadjusted surface pressure, and downscaled surface
    pressure.
    :param humidity, --- 2-meter specific humidity
           tmp2m -- downscaled temperatures ,
           t2dTmp -- orignial undownscaled temperatures
           psfcTmp--- orignial undownscaled air pressure
    :return: q2Tmp -- downscaled 2-meter specific humidity
    """

    # First calculate relative humidity given original surface pressure and 2-meter
    # temperature

    tmpHumidity = tmpHumidity/(1-tmpHumidity)

    T0 = 273.15
    EP = 0.622
    ONEMEP = 0.378
    ES0 = 6.11
    A = 17.269
    B = 35.86

    EST = ES0 * np.exp((A * (t2dTmp - T0)) / (t2dTmp - B))
    QST = (EP * EST) / ((psfcTmp * 0.01) - ONEMEP * EST)
    relHum = 100 * (tmpHumidity / QST)

    term1 = A * (tmp2m - T0)
    term2 = tmp2m - B
    EST = np.exp(term1 / term2) * ES0

    QST = (EP * EST) / ((tmpHumidity/100.0) - ONEMEP * EST)
    q2Tmp = QST * (relHum * 0.01)
    q2Tmp = q2Tmp / (1.0 + q2Tmp)

    return q2Tmp

# GFS incoming short wave radiation downscaling function based on topographic adjustment derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def gfs_dswr_downscaling_topo_adj(dswr_in, timestamp,NextGen_catchment_features):
    """
    Topographic adjustment of incoming shortwave radiation fluxes,
    given input parameters.
    :param dswr_in:
    :param current_output_date:
    :param NextGen_catchment_features: global variable
    :return: SWDOWN_OUT:  downscaled incoming shortwave radiation
    """

    # By the time this function has been called, necessary input static grids (height, slope, etc),
    # should have been calculated for each local slab of data.
    DEGRAD = math.pi/180.0
    DPD = 360.0/365.0

    # For short wave radiation
    DECLIN = 0.0
    SOLCON = 0.0

    coszen_loc, hrang_loc = calc_coszen(DECLIN,timestamp,NextGen_catchment_features)
    dswr_downscaled = TOPO_RAD_ADJ_DRVR(dswr_in,coszen_loc,DECLIN,SOLCON,hrang_loc,NextGen_catchment_features)
    return dswr_downscaled

# Calculate cosine of solar zenith angle based on time of day derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def calc_coszen(declin,timestamp,NextGen_catchment_features):
    """
    Downscaling function to compute radiation terms based on current datetime
    information and lat/lon grids.
    :param input_forcings:
    :param declin:
    :return:
    """
    degrad = math.pi / 180.0
    gmt = 0

    # Calculate the current julian day.
    julian = timestamp.to_julian_date()

    da = 6.2831853071795862 * ((julian - 1) / 365.0)
    eot = ((0.000075 + 0.001868 * math.cos(da)) - (0.032077 * math.sin(da)) - \
           (0.014615 * math.cos(2 * da)) - (0.04089 * math.sin(2 * da))) * 229.18
    xtime = timestamp.hour * 60.0  # Minutes of day
    xt24 = int(xtime) % 1440 + eot
    tloctm = NextGen_catchment_features.centroid_lon.values/15.0 + gmt + xt24/60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    xxlat = NextGen_catchment_features.centroid_lat.values * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(declin) * np.cos(hrang)

    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None

    return coszen, hrang

# GFS topography shortwave radiation adjustment based on time of day derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py    
def TOPO_RAD_ADJ_DRVR(dswr_in,COSZEN,declin,solcon,hrang2d,NextGen_catchment_features):
    """
    Downscaling driver for correcting incoming shortwave radiation fluxes from a low
    resolution to a a higher resolution.
    :param NextGen_catchment_features:
    :param dswr_in
    :param COSZEN:
    :param declin:
    :param solcon:
    :param hrang2d:
    :return:
    """
    degrad = math.pi / 180.0

    # Does this need to be just the local reference for a catchment or
    # entire grid???? Either nx, ny = 1 or len(num_catchments)???
    ny = len(NextGen_catchment_features)
    nx = len(NextGen_catchment_features)

    xxlat = NextGen_catchment_features.centroid_lat.values*degrad

    # Sanity checking on incoming shortwave grid.
    SWDOWN = dswr_in
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0
    COSZEN[np.where(COSZEN < 1E-4)] = 1E-4

    corr_frac = np.empty([ny], np.int64)
    diffuse_frac = np.empty([ny], np.int64)
    corr_frac[:] = 0
    diffuse_frac[:] = 0

    indTmp = np.where((NextGen_catchment_features.slope_m_km.values == 0.0) &
                      (SWDOWN <= 10.0))
    corr_frac[indTmp] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term2 = ((0 - np.cos(NextGen_catchment_features.aspect.values)) *
             np.sin(NextGen_catchment_features.slope_m_km.values))
    term3 = np.sin(hrang2d) * (np.sin(NextGen_catchment_features.aspect.values) *
                               np.sin(NextGen_catchment_features.slope_m_km.values))
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(NextGen_catchment_features.slope_m_km.values)
    term5 = np.cos(xxlat) * (np.cos(NextGen_catchment_features.aspect.values) *
                             np.sin(NextGen_catchment_features.slope_m_km.values))
    term6 = np.sin(xxlat) * np.cos(NextGen_catchment_features.slope_m_km.values)

    csza_slp = (term1 * term2 - term3 + term4) * math.cos(declin) + \
               (term5 + term6) * math.sin(declin)

    # Topographic shading
    csza_slp[np.where(csza_slp <= 1E-4)] = 1E-4

    # Correction factor for sloping topographic: the diffuse fraction of solar
    # radiation is assumed to be unaffected by the slope.
    corr_fac = diffuse_frac + ((1 - diffuse_frac) * csza_slp) / COSZEN
    corr_fac[np.where(corr_fac > 1.3)] = 1.3

    # Peform downscaling
    SWDOWN_OUT = SWDOWN * corr_fac

    # Reset variables to free up memory
    # corr_frac = None
    diffuse_frac = None
    term1 = None
    term2 = None
    term3 = None
    term4 = None
    term5 = None
    term6 = None

    return SWDOWN_OUT

def python_ExactExtract_Interp_weights(file_index_id,output_root,gfs_grib2_files,forecast_hour,hyfabfile,NextGen_catchment_features,GFS_weights_copy,bias_calibration, downscaling,transform, projection, i1_conus, j1_conus, cols_conus, rows_conus,max_forecast_hour):

    # get current and next 3hr forecast cycle within
    # block in order to interpolate between cycles
    gfs_grib2_file_cycle = [x for x in gfs_grib2_files if re.search(str(forecast_hour) + '.grib2',x) or re.search(str(forecast_hour + 3) + '.grib2',x)]

    # Initalize pandas dataframe to save results to csv file
    csv_results = pd.DataFrame([])

    # Assign GFS met variables to use for regridding techniques
    GFS_met_vars = ["PRES_surface","HGT_surface","TMP_2maboveground","SPFH_2maboveground","UGRD_10maboveground","VGRD_10maboveground","PRATE_surface","DSWRF_surface","DLWRF_surface"]

    # file_loop variable to indicate which forecast cycle
    # hour were looping between
    file_loop = 0

    # Obtain the timestamps needed for bias calibration
    # and downscaling methods to be implemented on
    # interpolated timesteps
    hours = []
    forecast_hours = []
    timestamps = []

    for grib2_file in gfs_grib2_file_cycle:

        # Initalize which raster bands we will will extract
        # in the grib2 file. For the initalized GFS forecast
        # (hour 0)however the band indices will differ, so
        # we are accounting for that. Were also accounting
        # for the fact that at hour 0, the initalized GFS
        # model has no data available for PRATE
        gfs_band_rasters = [6,7,33,34,39,40,43,95,96]

        # Initalize gdal gtiff driver before GFS variable loop
        driver = gdal.GetDriverByName("GTiff")

        # Open GFS grib2 dataset using GDAL library
        # before the GFS variable loop
        dataset = gdal.Open(grib2_file)

        # Grab the first raster band from GFS grib2 file
        # to obtain GFS metadata
        band = dataset.GetRasterBand(1)

        # Grab the first GFS raster band in the dataset and
        # extract metadata into array to get datetime info
        metadata = np.array(list(band.GetMetadata().items()))

        # Extract the metadata reference and forecast time
        # to create current GFS forecast timestamp, this will be
        # needed to disscet within the GFS bias corrections and
        # downscaling functions
        timestamp = pd.to_datetime(metadata[4][1].split(' ')[5].split('=')[1][:-1]) + pd.TimedeltaIndex([float(metadata[3][1].split(' ')[0])],'s')

        # get current hour of GFS forecast cycle in metadata,
        # this is needed within GFS bias correction and downscaling functions
        current_output_step = float(metadata[3][1].split(' ')[0])/3600.0

        # get hour of the day based on current GFS timestamp, this is needed
        # within the GFS bias correction and downscaling functions
        hour = timestamp.hour[0]

        # Initalize variables needed for manual aerial weight calculation
        GFS_data = np.zeros((len(GFS_met_vars),rows_conus,cols_conus))
        GFS_weights = GFS_weights_copy.copy()
        EE_data_sum = np.zeros((len(GFS_met_vars),len(GFS_weights)))
        EE_coverage_fraction_sum = EE_data_sum.copy()[0,:]
        # loop over each meteorological variable and generate a single
        # CONUS .tif file for the GFS data that can be properly regridded
        # to the NextGen catchments feastures within the CONUS domain
        for i in np.arange(len(GFS_met_vars)):
            # Define which variable we are looking
            # to regrid for GFS forcings
            variable = GFS_met_vars[i]
            # Since GFS forecast files are sorted, index 0 implies the initalized
            # forecast time for the model run. NWP models won't generate time=0
            # flux fields, the model physics need to integrate forward in order
            # to calculate fluxes. Base on National Water Model logic, we will assume
            # that precipitation remains contstant from hour zero to hour 1 of the
            # forecast cycle and therefore use precipitation from forecast hour 1

            # Assign GFS band to extract grib2 data from
            # based on the variable assigned in the loop
            gfs_band = gfs_band_rasters[i]
            # Get GFS data pertaining to variable in the loop
            band = dataset.GetRasterBand(gfs_band)
            # Get CONUS susbet of the GFS raster, where grid spacing is even
            # and NextGen data of interest. Flag for unit conversion for temperature.
            if(variable == "TMP_2maboveground"):
                # Convert GFS grib2 temp data from Celsius to Kelvin
                GFS_data[i,:,:] = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus) + 273.15,dtype=np.float32)
            else:
                GFS_data[i,:,:] = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus),dtype=np.float32)

        # Now loop through EE weights and raster indices and
        # calculate coverage fraction summation
        for row in zip(GFS_weights.index, GFS_weights['row'], GFS_weights['col'], GFS_weights['coverage_fraction']):
            # Flag to discard missing gfs grid cell
            # data from aerial weight average
            if(GFS_data[0,int(row[1]),int(row[2])] > -1000.0):
                # Loop over each gfs met variable and calculate
                # coverage fraction summation (value * coverage fraction)
                for var in np.arange(len(GFS_met_vars)):
                    EE_data_sum[var,row[0]] += (GFS_data[var,int(row[1]),int(row[2])]*row[3])
                # Account for coverage fraction with available data
                EE_coverage_fraction_sum[row[0]] += row[3]
        # Once summation is finished for all met variables
        # then we groupby the catchment ids and  calculate
        # coverage fraction weighted mean (summation/coverage fraction total)
        # over each met variable
        var_loop = 0
        for var in GFS_met_vars:
            GFS_weights[var] = np.array(EE_data_sum[var_loop,:],dtype=float)
            var_loop += 1
        # Add coverage fraction that accounted for only available data
        # to the dataframe before grouping by catchment ids
        GFS_weights['EE_coverage_fraction'] = EE_coverage_fraction_sum[:]
        GFS_weights = GFS_weights.groupby('divide_id').sum()
        GFS_weights['cat-id'] = np.array(GFS_weights.index.values)

        # get datetime of forcing file for global time variable from netcdf file
        # which is in seconds since gfs reference date and the date-time string
        time = np.zeros(len(GFS_weights))
        time[:] = (timestamp - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()

        GFS_weights['time'] = time

    # Finalize aerial weighted calculation for GFS data
        for var in GFS_met_vars:
            GFS_weights[var] = (GFS_weights[var]/GFS_weights['EE_coverage_fraction'])

        # Now sort the dataframe columnds and drop the groupby
        # index before we return the dataframe to thread
        gfs_results = GFS_weights[['cat-id','time','PRATE_surface','HGT_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]
        gfs_results = gfs_results.reset_index(drop=True)

        # Now, go ahead and copy the data for two missing hours
        # between the 3hr forecast cycles and append them to dataset
        # before appending the next gfs forecast data
        if(file_loop == 1):
            missing_data = gfs_results.copy()
            missing_data_hr2 = gfs_results.copy()
            for column in GFS_met_vars:
                if(column != "PRATE_surface"):
                    missing_data[column] = pd.NA
                    missing_data_hr2[column] = pd.NA
            missing_data['time'] = time_1hr
            missing_data_hr2['time'] = time_2hr

            csv_results = pd.concat([csv_results,missing_data])
            csv_results = pd.concat([csv_results,missing_data_hr2])
            csv_results = pd.concat([csv_results,gfs_results])
            hours.append(hour)
            timestamps.append(timestamp[0])
            forecast_hours.append(current_output_step)
        else:
            csv_results = pd.concat([csv_results,gfs_results])
            time_1hr = time + 3600.0
            time_2hr = time + 7200.0
            hours.append(hour)
            hours.append(int(hour+1))
            hours.append(int(hour+2))
            timestamps.append(timestamp[0])
            timestamps.append((timestamp[0] + pd.TimedeltaIndex([3600],'s'))[0])
            timestamps.append((timestamp[0] + pd.TimedeltaIndex([7200],'s'))[0])
            forecast_hours.append(current_output_step)
            forecast_hours.append(current_output_step+1.0)
            forecast_hours.append(current_output_step+2.0)

        file_loop += 1

    csv_results.index = pd.Timestamp("1970-01-01 00:00:00") + pd.TimedeltaIndex(csv_results['time'].values,'s')

    csv_results_final = csv_results.interpolate(method='linear')

    csv_results = pd.DataFrame([])

    time_steps = csv_results_final['time'].unique()[0:-1]

    for i in range(len(time_steps)):

        gfs_results = csv_results_final.loc[csv_results_final.time == time_steps[i],:]
        ############### This is where we call GFS bias correction function  #########
        if(bias_calibration):
            gfs_results = GFS_bias_correction(gfs_results,hours[i],forecast_hours[i])

        ############### This is where we call GFS downscaling function  #########
        if(bias_calibration):
            gfs_results = GFS_downscaling(gfs_results,timestamps[i],NextGen_catchment_features)

        ####################################################################

        csv_results = pd.concat([csv_results,gfs_results])

    return csv_results


def python_ExactExtract_weights(gfs_grib2_file,file_index_id,output_root,gfs_grib2_files,hyfabfile,NextGen_catchment_features,GFS_weights,bias_calibration, downscaling,transform, projection, i1_conus, j1_conus, cols_conus, rows_conus):

    # Initalize pandas dataframe to save results to csv file
    csv_results = pd.DataFrame([])

    # Assign GFS met variables to use for regridding techniques
    GFS_met_vars = ["PRES_surface","HGT_surface","TMP_2maboveground","SPFH_2maboveground","UGRD_10maboveground","VGRD_10maboveground","PRATE_surface","DSWRF_surface","DLWRF_surface"]

    # Initalize which raster bands we will will extract
    # in the grib2 file. For the initalized GFS forecast
    # (hour 0)however the band indices will differ, so
    # we are accounting for that. Were also accounting
    # for the fact that at hour 0, the initalized GFS
    # model has no data available for PRATE
    if(file_index_id == 0):
        gfs_band_rasters = [6,7,28,29,30,31,0,48,49]
        gfs_precip_hour1_raster = [43]
    else:
        gfs_band_rasters = [6,7,33,34,39,40,43,95,96]

    # Initalize gdal gtiff driver before GFS variable loop
    driver = gdal.GetDriverByName("GTiff")

    # Open GFS grib2 dataset using GDAL library
    # before the GFS variable loop
    dataset = gdal.Open(gfs_grib2_file)

    # Grab the first raster band from GFS grib2 file
    # to obtain GFS metadata
    band = dataset.GetRasterBand(1)

    # Grab the first GFS raster band in the dataset and
    # extract metadata into array to get datetime info
    metadata = np.array(list(band.GetMetadata().items()))

    # Extract the metadata reference and forecast time
    # to create current GFS forecast timestamp, this will be
    # needed to disscet within the GFS bias corrections and
    # downscaling functions
    timestamp = pd.to_datetime(metadata[4][1].split(' ')[5].split('=')[1][:-1]) + pd.TimedeltaIndex([float(metadata[3][1].split(' ')[0])],'s')

    # get current hour of GFS forecast cycle in metadata,
    # this is needed within GFS bias correction and downscaling functions
    current_output_step = float(metadata[3][1].split(' ')[0])/3600.0

    # get hour of the day based on current GFS timestamp, this is needed
    # within the GFS bias correction and downscaling functions
    hour = timestamp.hour[0]

    # Initalize variables needed for manual aerial weight calculation
    GFS_data = np.zeros((len(GFS_met_vars),rows_conus,cols_conus))
    EE_data_sum = np.zeros((len(GFS_met_vars),len(GFS_weights)))
    EE_coverage_fraction_sum = EE_data_sum.copy()[0,:]

    # loop over each meteorological variable and get data
    # from each gdal raster to manually regrid
    for i in np.arange(len(GFS_met_vars)):
        # Define which variable we are looking
        # to regrid for GFS forcings
        variable = GFS_met_vars[i]
        # Since GFS forecast files are sorted, index 0 implies the initalized
        # forecast time for the model run. NWP models won't generate time=0
        # flux fields, the model physics need to integrate forward in order
        # to calculate fluxes. Base on National Water Model logic, we will assume
        # that precipitation remains contstant from hour zero to hour 1 of the
        # forecast cycle and therefore use precipitation from forecast hour 1
        if(file_index_id == 0 and variable == 'PRATE_surface'):
            dataset_hour1 = gdal.Open(gfs_grib2_files[1])
            gfs_band = gfs_precip_hour1_raster[0]
            band = dataset.GetRasterBand(gfs_band)
            GFS_data[i,:,:] = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus),dtype=np.float32)
        else:
            # Assign GFS band to extract grib2 data from
            # based on the variable assigned in the loop
            gfs_band = gfs_band_rasters[i]
            # Get GFS data pertaining to variable in the loop
            band = dataset.GetRasterBand(gfs_band)
            # Get CONUS susbet of the GFS raster, where grid spacing is even
            # and NextGen data of interest. Flag for unit conversion for temperature.
            if(variable == "TMP_2maboveground"):
                # Convert GFS grib2 temp data from Celsius to Kelvin
                GFS_data[i,:,:] = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus) + 273.15,dtype=np.float32)
            else:
                GFS_data[i,:,:] = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus),dtype=np.float32)

    # Now loop through EE weights and raster indices and
    # calculate coverage fraction summation
    for row in zip(GFS_weights.index, GFS_weights['row'], GFS_weights['col'], GFS_weights['coverage_fraction']):
        # Flag to discard missing GFS grid cell
        # data from aerial weight average
        if(GFS_data[0,int(row[1]),int(row[2])] > 0.0):
            # Loop over each GFS met variable and calculate
            # coverage fraction summation (value * coverage fraction)
            for var in np.arange(len(GFS_met_vars)):
                EE_data_sum[var,row[0]] += (GFS_data[var,int(row[1]),int(row[2])]*row[3])
            # Account for coverage fraction with available data
            EE_coverage_fraction_sum[row[0]] += row[3]
    # Once summation is finished for all met variables
    # then we groupby the catchment ids and  calculate
    # coverage fraction weighted mean (summation/coverage fraction total)
    # over each met variable
    var_loop = 0
    for var in GFS_met_vars:
        GFS_weights[var] = np.array(EE_data_sum[var_loop,:],dtype=float)
        var_loop += 1
    # Add coverage fraction that accounted for only available data
    # to the dataframe before grouping by catchment ids
    GFS_weights['EE_coverage_fraction'] = EE_coverage_fraction_sum[:]
    GFS_weights = GFS_weights.groupby('divide_id').sum()
    GFS_weights['cat-id'] = np.array(GFS_weights.index.values)

    # get datetime of forcing file for global time variable from netcdf file
    # which is in seconds since GFS reference date and the date-time string
    time = np.zeros(len(GFS_weights))
    time[:] = (timestamp - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()

    GFS_weights['time'] = time

    # Finalize aerial weighted calculation
    for var in GFS_met_vars:
        GFS_weights[var] = (GFS_weights[var]/GFS_weights['EE_coverage_fraction'])

    # Now sort the dataframe columnds and drop the groupby
    # index before we return the dataframe to thread
    csv_results = GFS_weights[['cat-id','time','PRATE_surface','HGT_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]

    ################ Once GFS regridding is complete, we now call ##############
    ################ the bias correcation and downscaling functions ###########
    ################ sequentially according to NCAR documentation ############

    ############### This is where we call GFS bias correction function  #########

    if(bias_calibration):
        csv_results = GFS_bias_correction(csv_results,hour,current_output_step)

    ############### This is where we call GFS downscaling function  #########
    if(downscaling):
        csv_results = GFS_downscaling(csv_results,timestamp[0],NextGen_catchment_features)

    ####################################################################

    csv_results = csv_results.reset_index(drop=True)


    return csv_results

def Python_ExactExtract_Coverage_Fraction_Weights(gfs_grib2_file, hyfab_file, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus, output_dir):

    # Initalize gdal gtiff driver for GFS data
    driver = gdal.GetDriverByName("GTiff")

    # Open GFS grib2 dataset using GDAL library
    dataset = gdal.Open(gfs_grib2_file)

    # Get GFS surface pressure raster band
    band = dataset.GetRasterBand(6)

    ## Initalize GFS tif file were creating for the transformed
    ## CONUS subset of a CFS raster
    gfs_tif = join(output_dir, "GFS_weights.tif")

    ## Now create a geotiff dataset of the CONUS GFS raster subset
    dst_ds = driver.Create(gfs_tif,cols_conus,rows_conus,1,gdal.GDT_Float32)

    ## Set CONUS geometry and GFS grid spacing
    dst_ds.SetGeoTransform(transform)
    ## Set GFS projection to geotiff
    dst_ds.SetProjection(projection.ExportToWkt())

    # Get GFS surface pressure data and write to tif file
    data = np.array(band.ReadAsArray(i1_conus,j1_conus,cols_conus,rows_conus),dtype=np.float32)
    dst_ds.GetRasterBand(1).WriteArray(data)
    # Set raster band GFS variable name
    dst_ds.GetRasterBand(1).SetDescription('PRES_surface')

    # Saves data to disk
    dst_ds.FlushCache()
    # Close geotiff file for use in ExactExtract
    dst_ds = None

    # Define raster wrapper for CFS meteorological variable
    # and specify nc_file attribute to be True. Otherwise,
    # this function will expect a .tif file. Assign data for dict variable
    rsw = GDALRasterWrapper(gfs_tif,band_idx=1)

    # hydrofabric raster dataset to regrid forcings
    # based on user subset
    dsw = GDALDatasetWrapper(hyfab_file)

    # Define output writer and coverage fraction weights
    # output file
    EE_coverage_fraction_csv = os.path.join(output_dir + "GFS_ExactExtract_Weights.csv")
    writer = CoverageWriter(EE_coverage_fraction_csv, dsw)

    # Process the data and produce the coverage fraction
    # weights file between the hydrofabric and GFS data
    processor = CoverageProcessor(dsw, writer, rsw)
    processor.process()

    # Flush changes to disk
    writer = None

    # Since we have finished using the GFS tif file
    # we can now just remove file from system and save
    # disk storage on server
    os.remove(gfs_tif)


    # Return the pathway of the coverage fraction weight file
    return EE_coverage_fraction_csv


def gdal_grib2_transformation(grib2_file):
    # Open grib2 dataset using gdal module
    dataset = gdal.Open(grib2_file)

    # Get just the first raster band of dataset to
    # transform and subset the grid
    band = dataset.GetRasterBand(1)

    # Get rows and columns of raster dataset
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    # Get current GFS grib2 transformation of dataset
    transform = dataset.GetGeoTransform()

    # Get min and max values of latitude and longitude coordinates
    minx = transform[0]
    maxx = transform[0] + cols * transform[1] + rows * transform[2]

    miny = transform[3] + cols * transform[4] + rows * transform[5]
    maxy = transform[3]

    # Get width and height dimsenions of grid spacing (degrees)
    width = maxx - minx
    height = maxy - miny

    # Get origin coordinates of GFS grid and its grid spacing
    # in the x and y directions (not consistent as you near the poles)
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]

    # Pre-define CONUS subset, with oceanic boundaries equivalent to HRRR
    newminx = 225.0
    newmaxx = 300.0
    newminy = 21.0
    newmaxy = 53.0

    # Define coordinate bounds for CONUS
    p1 = (newminx, newmaxy)
    p2 = (newmaxx, newminy)

    # find new array indices where CONUS
    # domain is bounded for grib data
    i1 = int((p1[0] - xOrigin) / pixelWidth)
    j1 = int((yOrigin - p1[1])  / pixelHeight)
    i2 = int((p2[0] - xOrigin) / pixelWidth)
    j2 = int((yOrigin - p2[1]) / pixelHeight)

    # now redefine the new number of columns and rows
    # for the subset CONUS domain
    new_cols = i2 - i1
    new_rows = j2 - j1

    # Now lets define new coordinate transformation bounds
    # for the raster arrays in the CONUS domain.
    # We also convert longitude coordinates from 0-360
    # to -180 180 in order to properly construct the raster
    # tif file for gdal to read in ExactExtract
    new_x = ((xOrigin + i1*pixelWidth) + 180) % 360 - 180
    new_y = yOrigin - j1*pixelHeight

    # Define new transformation for CONUS tif files
    new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

    # Get the original Projection of the grib2 file
    # to save the projection data to tif files
    wkt = dataset.GetProjection()
    srs = gdal.osr.SpatialReference()
    srs.ImportFromWkt(wkt)

    return new_transform, srs, i1, j1, new_cols, new_rows


def process_sublist(data : dict, lock: Lock, num: int, EE_results, output_root, datafiles, hyfabfile, weights, bias_calibration, downscaling, max_forecast_hour, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus):
    num_files = len(data["forcing_files"])    

    # Read in hydrofabric file for forcing metadata
    # to utilize bias calibration and dowscaling function
    # and sort by divide values since that is the way the
    # coverage fraction weights sorts the data by
    NextGen_catchment_features = gpd.read_file(hyfabfile)
    NextGen_catchment_features = NextGen_catchment_features.sort_values(by='divide_id')

    # Read in ExactExtract coverage fraction weights file
    weights = pd.read_csv(weights)

    # Initalize pandas dataframe to save the
    # regridded GFS ExactExtract results from
    # each GFS file we loop through
    EE_df_final = pd.DataFrame()

    for i in range(num_files):
        # extract forcing file and file index
        gfs_grib_file = data["forcing_files"][i]
        file_index_id = data["file_index"][i]
        # Get hour of GFS forecast cycle time
        forecast_hour = int(gfs_grib_file.split(".")[-2].split("f")[2])

        if(forecast_hour >= 120 and forecast_hour != max_forecast_hour):
            # Call python ExactExtract interpolation routine
            # and interpolate meteorological data between 
            # 3 hourly output timestamps, while accounting whether
            # or not user requests weight calculation that are already
            # provided within NextGen hydrofabric file
            EE_df = python_ExactExtract_Interp_weights(file_index_id,output_root,datafiles,forecast_hour,hyfabfile,NextGen_catchment_features,weights.copy(), bias_calibration, downscaling, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus,max_forecast_hour)
        else:

            # Call python ExactExtract routine to directly extract 
            # GFS regridded results to global GFS variables, while
            # accouting whether or not user requests weight calculation
            # that are alreadt provided within NextGen hydrofabric file
            EE_df = python_ExactExtract_weights(gfs_grib_file,file_index_id,output_root,datafiles,hyfabfile,NextGen_catchment_features,weights.copy(), bias_calibration, downscaling, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus)

        # concatenate the regridded data to threads final dataframe
        EE_df_final = pd.concat([EE_df_final,EE_df])

        # Print how far thread is along with regridding all GFS data
        print("Thread (" + str(num) + ") is " + str((i+1)/num_files*100) + '% complete')
     
    # collect the garbage within the thread before sending results back to main
    # thread to maximize our RAM as much as possible
    gc.collect()

    # Put regridded results into thread queue to return to main thread
    EE_results.put(EE_df_final)

def NextGen_Forcings_GFS(output_root, gfs_grib2, netcdf, csv, hyfabfile, weights_file, bias_calibration, downscaling, num_processes):

    if(netcdf):
        netcdf_dir = join(output_root,"netcdf")
        if not os.path.isdir(netcdf_dir):
            os.makedirs(netcdf_dir)
    else:
        netcdf_dir = ''

    if(csv):
        csv_dir = join(output_root,"csv")
        if not os.path.isdir(csv_dir):
            os.makedirs(csv_dir)
    else:
        csv_dir = ''

    #generate catchment geometry from hydrofabric
    cat_df_full = gpd.read_file(hyfabfile)
    g = [i for i in cat_df_full.geometry]
    h = [i for i in cat_df_full.divide_id]
    n_cats = len(g)
    num_catchments = n_cats
    print("number of catchments = {}".format(n_cats))

    # GFS 12km surface flux data pathway to grib2 files
    datafile_path = join(gfs_grib2, "*.grib2")
    #get list of files
    datafiles = glob.glob(datafile_path)
    print("number of forcing files = {}".format(len(datafiles)))
    #process data with time ordered
    datafiles.sort()

    if(bias_calibration):
        print('Will perform GFS bias calibration on forcing data')
    if(downscaling):
        print('Will perform GFS downscaling on forcing data')

    # NWM operational confiugration runs for 10 days, so cap forecast
    # cycle by that much within the data directory
    datafiles = datafiles[0:161]

    # Get final hour of GFS forecast cycle time
    max_forecast_hour = int(datafiles[-1].split(".")[-2].split("f")[2])

    #prepare for processing, number of files is
    # equivalent to the number of GFS forecast
    # hours to regrid for in NextGen
    num_files = max_forecast_hour + 1

    # Now we want to create a variable to index the datafile 
    # that are already sorted. This index will assign data
    # to global meterological variables below
    file_index = np.arange(num_files)

    # GFS reference time to use with
    # respect to GFS forecast time
    ref_date = pd.Timestamp("1970-01-01 00:00:00")

    # Get raster transformation (longitude scale to -180 to 180) and subset
    # indices for CONUS data where grid spacing is even and not distorted
    transform, projection, i1_conus, j1_conus, cols_conus, rows_conus = gdal_grib2_transformation(datafiles[2])

    # Need to reproject the hydrofabric crs to the meteorological forcing
    # dataset crs for ExactExtract to properly regrid the data
    hyfabfile_final = join(output_root,"hyfabfile_final.json")
    hyfab_data = gpd.read_file(hyfabfile,layer='divides')
    hyfab_data = hyfab_data.to_crs(projection.ExportToWkt())
    hyfab_data.to_file(hyfabfile_final,driver="GeoJSON")

    # Flag to see if user has already provided an ExactExtract
    # coverage weights file, otherwise go ahead and produce the file
    if(weights_file != None):
        weights = weights_file
    else:
        # Generate the ExactExtract Coverage Fraction Weights File
        weights = Python_ExactExtract_Coverage_Fraction_Weights(datafiles[1], hyfabfile_final, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus, output_root)

    #generate the data objects for child processes
    file_groups = np.array_split(np.array(datafiles), num_processes)
    file_index_groups = np.array_split(file_index, num_processes)

    process_data = []
    process_list = []
    lock = Lock()

    # Initalize thread storage to return to main program
    EE_results = Queue()

    for i in range(num_processes):
        # fill the dictionary with gfs grib2 file and its indices
        data = {}
        data["forcing_files"] = file_groups[i]
        data["file_index"] = file_index_groups[i]
      
        #append to the list
        process_data.append(data)

        p = Process(target=process_sublist, args=(data, lock, i, EE_results, output_root, datafiles, hyfabfile_final, weights, bias_calibration, downscaling, max_forecast_hour, transform, projection, i1_conus, j1_conus, cols_conus, rows_conus))

        process_list.append(p)

    #start all processes
    for p in process_list:
        p.start()

    # Before we terminate threads, aggregate thread
    # regridded results together and save to main thread
    final_df = pd.DataFrame()
    for i in range(num_processes):
        result = EE_results.get()
        final_df = pd.concat([final_df,result])

    #wait for termination
    for p in process_list:
        p.join()

    # delete variables to free up RAM
    del(EE_results)

    # Sort aggregated data based on cat-id and timestamp
    final_df = final_df.sort_values(by=['cat-id','time'])

    if(netcdf):
        # generate single NextGen netcdf file from global variables
        # of regridded GFS data
        create_ngen_netcdf(final_df,netcdf_dir,num_catchments,num_files)

    if(csv):
        # generate NextGen csv formatted files for each catchment id
        # within the hydrofabric file
        create_ngen_csv_catchments(final_df, num_processes, csv_dir, num_catchments,num_files)

    # Now clean up I/O files from the script to free up memory for the user
    # Remove the temporary hydrofabric file
    os.remove(hyfabfile_final)
