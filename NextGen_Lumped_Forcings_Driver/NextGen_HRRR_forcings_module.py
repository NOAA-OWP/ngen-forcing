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
# load python C++ binds from ExactExtract module library
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, CoverageProcessor, CoverageWriter, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
# must import gdal to properly read and partiton rasters
# from HRRR netcdf files
from osgeo import gdal

def create_ngen_netcdf(final_df,netcdf_dir,num_catchments,num_files):
    """
    Create NextGen netcdf file with specified format
    """

    # Get start time from first instance of final dataframe
    start_time = (pd.Timestamp("1970-01-01 00:00:00") + pd.Timedelta(seconds=final_df.time.values[0])).strftime("%Y%m%d%H")

    #create output netcdf file name
    output_netcdf = join(netcdf_dir, "NextGen_HRRR_forcing_"+start_time+".nc")

     
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
    setattr(PRATE_surface_out, "long_name", "Total Precipitation")
    setattr(PRATE_surface_out, "short_name", "APCP_surface")
    setattr(PRATE_surface_out, "units", "kg/m^2")
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
        csv_df['RAINRATE'] = data["RAINRATE"][i,:]/3600.0
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

def HRRR_bias_correction(dataset, timestamp, current_output_step, NextGen_catchment_features, n_fcst_hr, AnA):

    # Perform bias correction for incoming short wave radiation
    dataset['DSWRF_surface'] = HRRR_swdown_bias_correction(dataset['DSWRF_surface'].values, timestamp, current_output_step, NextGen_catchment_features, n_fcst_hr, AnA)


    # Perform bias correction for 2m Air Temperature
    dataset['TMP_2maboveground'] = HRRR_tmp_bias_correction(dataset['TMP_2maboveground'].values, timestamp, current_output_step, AnA)

    # Perform bias correction for u and v wind components collectively
    dataset['UGRD_10maboveground'], dataset['VGRD_10maboveground'] = HRRR_wspd_bias_correction(dataset['UGRD_10maboveground'].values, dataset['VGRD_10maboveground'].values, timestamp, current_output_step, AnA)

    return dataset

# HRRR Incoming longwave radiation bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def HRRR_swdown_bias_correction(swdown_data, timestamp, current_output_step, NextGen_catchment_features, n_fcst_hr, AnA):
    """
    Function to implement a bias correction to the forecast incoming shortwave radiation fluxes.
    NOTE!!!! - This bias correction is based on in-situ analysis performed against HRRRv3
    fields. It's high discouraged to use this for any other NWP products, or even with the HRRR
    changes versions in the future.
    :param input_forcings:
    :param geo_meta_wrf_hydro:
    :param config_options:
    :param mpi_config:
    :param force_num:
    :return:
    """

    # Establish constant parameters. NOTE!!! - THESE WILL CHANGE WITH FUTURE HRRR UPGRADES.
    # determine if we're in AnA or SR configuration
    if AnA:   #Analysis and Assimilation run
        c1 = 0.079
        c2 = 0
    else:
        c1 = -0.032
        c2 = -0.040

    # Establish current datetime information, along wth solar constants.
    f_hr = current_output_step
    # For now, hard-coding the total number of forecast hours to be 18, since we
    # are assuming this is HRRR
    #n_fcst_hr = 18

    # Trig params
    d2r = math.pi / 180.0
    r2d = 180.0 / math.pi
    date_current = timestamp
    hh = timestamp.hour
    mm = 0.0
    ss = 0.0
    doy = float(time.strptime(date_current.strftime('%Y.%m.%d'), '%Y.%m.%d').tm_yday)
    frac_year = 2.0 * math.pi / 365.0 * (doy - 1.0 + (hh / 24.0) + (mm / 1440.0) + (ss / 86400.0))
    # eqtime is the difference in minutes between true solar time and that if solar noon was at actual noon.
    # This difference is due to Earth's elliptical orbit around the sun.
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(frac_year) - 0.032077 * math.sin(frac_year) -
                       0.014615 * math.cos(2.0 * frac_year) - 0.040849 * math.sin(2.0 * frac_year))

    # decl is the solar declination angle in radians: how much the Earth is tilted toward or away from the sun
    decl = 0.006918 - 0.399912 * math.cos(frac_year) + 0.070257 * math.sin(frac_year) - \
        0.006758 * math.cos(2.0 * frac_year) + 0.000907 * math.sin(2.0 * frac_year) - \
        0.002697 * math.cos(3.0 * frac_year) + 0.001480 * math.sin(3.0 * frac_year)

    # Create temporary grids for calculating the solar zenith angle, which will be used in the bias correction.
    # time offset in minutes from the prime meridian
    time_offset = eqtime + 4.0 * NextGen_catchment_features.X.values

    # tst is the true solar time: the number of minutes since solar midnight
    tst = hh * 60.0 + mm + ss / 60.0 + time_offset

    # solar hour angle in radians: the amount the sun is off from due south
    ha = d2r * ((tst / 4.0) - 180.0)

    # solar zenith angle is the angle between straight up and the center of the sun's disc
    # the cosine of the sol_zen_ang is proportional to the solar intensity
    # (not accounting for humidity or cloud cover)
    sol_cos = np.sin(NextGen_catchment_features.Y.values * d2r) * math.sin(decl) + np.cos(NextGen_catchment_features.Y.values * d2r) * math.cos(decl) * np.cos(ha)

    # Check for any values outside of [-1,1] (this can happen due to floating point rounding)
    sol_cos[np.where(sol_cos < -1.0)] = -1.0
    sol_cos[np.where(sol_cos > 1.0)] = 1.0

    sol_zen_ang = r2d * np.arccos(sol_cos)

    # Check for any values greater than 90 degrees.
    sol_zen_ang[np.where(sol_zen_ang > 90.0)] = 90.0

    # Calculate where we have valid values.
    #ind_valid = None
    #ind_valid = np.where(sw_tmp != config_options.globalNdv)

    # Perform the bias correction.
    swdown_data = swdown_data + \
                       (c1 + (c2 * ((f_hr - 1) / (n_fcst_hr - 1)))) * np.cos(sol_zen_ang * d2r) * \
                       swdown_data


    # Reset variables to keep memory footprints low.
    del time_offset
    del tst
    del ha
    del sol_zen_ang
    #del ind_valid

    return swdown_data

# HRRR wind speed bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def HRRR_wspd_bias_correction(ugrd, vgrd, timestamp, current_output_step, AnA):

    hh = timestamp.hour
    MM = timestamp.month

    fhr = current_output_step

    wdir = np.arctan2(vgrd, ugrd)
    wspd = np.sqrt(np.square(ugrd) + np.square(vgrd))

    # determine if we're in AnA or SR configuration
    if AnA:
        hh -= 60.0 / 60
        if hh < 0:
            hh += 24
        net_bias_AA = 0.23
        diurnal_ampl_AA = -0.13
        diurnal_offs_AA = -0.6
        monthly_ampl_AA = 0.0
        monthly_offs_AA = 0.0

        wspd_bias_corr = net_bias_AA + diurnal_ampl_AA * math.sin(diurnal_offs_AA + hh / 24 * 2 * math.pi) + \
                         monthly_ampl_AA * math.sin(monthly_offs_AA + MM / 12 * 2*math.pi)
    else:
        net_bias_SR = -0.03
        diurnal_ampl_SR = -0.15
        diurnal_offs_SR = -0.3
        monthly_ampl_SR = 0.0
        monthly_offs_SR = 0.0
        fhr_mult_SR = -0.007

        wspd_bias_corr = net_bias_SR + fhr * fhr_mult_SR + \
                         diurnal_ampl_SR * math.sin(diurnal_offs_SR + hh / 24 * 2*math.pi) + \
                         monthly_ampl_SR * math.sin(monthly_offs_SR + MM / 12 * 2*math.pi)

    wspd = wspd + wspd_bias_corr
    wspd = np.where(wspd < 0, 0, wspd)

    ugrd_out = wspd * np.cos(wdir)
    vgrd_out = wspd * np.sin(wdir)

    return ugrd_out, vgrd_out

# HRRR temperature bias correction formula found from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/bias_correction.py
def HRRR_tmp_bias_correction(tmp_data, timestamp, current_output_step, AnA):

    #date_current = config_options.current_output_date
    hh = timestamp.hour
    MM= timestamp.month

    # determine if we're in AnA or SR configuration
    if AnA:
        hh -= 60. / 60
        if hh < 0:
            hh += 24
        net_bias_AA = 0.019
        diurnal_ampl_AA = -0.06
        diurnal_offs_AA = 1.5
        monthly_ampl_AA = 0.0
        monthly_offs_AA = 0.0

        bias_corr = net_bias_AA + diurnal_ampl_AA * math.sin(diurnal_offs_AA + hh / 24 * 2 * math.pi) + \
                    monthly_ampl_AA * math.sin(monthly_offs_AA + MM / 12 * 2*math.pi)

    else:
        net_bias_SR = 0.018
        diurnal_ampl_SR = -0.06
        diurnal_offs_SR = -0.6
        monthly_ampl_SR = 0.0
        monthly_offs_SR = 0.0
        fhr_mult_SR = -0.01

        fhr = current_output_step

        bias_corr = net_bias_SR + fhr * fhr_mult_SR + \
                    diurnal_ampl_SR * math.sin(diurnal_offs_SR + hh / 24 * 2*math.pi) + \
                    monthly_ampl_SR * math.sin(monthly_offs_SR + MM / 12 * 2*math.pi)


    tmp_data = tmp_data + bias_corr


    return tmp_data


    
def HRRR_downscaling(csv_results,timestamp,NextGen_catchment_features):
    
    # Extract regridded HRRR model elevation to use as
    # input in the HRRR downscaling functions
    elev = csv_results['HGT_surface'].values

    # Perform downscaling on HRRR temperature and reassign 
    # data to pandas dataframe
    tmp_2m = csv_results['TMP_2maboveground'].values
    tmp_downscaled = HRRR_temp_downscaling_simple_lapse(tmp_2m, elev,NextGen_catchment_features)
    csv_results['TMP_2maboveground'] = tmp_downscaled

    # Perform downscaling on HRRR surface pressure and reassign
    # data to pandas dataframe
    pres_old = csv_results['PRES_surface'].values
    pres_downscaled = HRRR_pres_downscaling_classic(pres_old,tmp_downscaled,elev,NextGen_catchment_features)
    csv_results['PRES_surface'] = pres_downscaled

    # Perform downscaling on HRRR 2m specific humidity and reassign
    # data to pandas dataframe
    tmpHumidity = csv_results['SPFH_2maboveground'].values
    humidity_downscaled = HRRR_hum_downscaling_classic(tmpHumidity,tmp_downscaled,tmp_2m,pres_old)
    csv_results['SPFH_2maboveground'] = humidity_downscaled

    
    # Perform downscaling on HRRR incoming short wave radiation and reassign
    # data to pandas dataframe
    dswr = csv_results['DSWRF_surface'].values
    dswr_downscaled = HRRR_dswr_downscaling_topo_adj(dswr, timestamp,NextGen_catchment_features)
    csv_results['DSWRF_surface'] = dswr_downscaled

    return csv_results

# HRRR Temperature downscaling function based on simple lapse rate derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def HRRR_temp_downscaling_simple_lapse(tmp_in, elev,NextGen_catchment_features):
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

    elevDiff = elev - NextGen_catchment_features.elevation_mean.values

    # Apply single lapse rate value to the input 2-meter
    # temperature values.
    tmp2m = tmp2m + (6.49/1000.0)*elevDiff

    return tmp2m

# HRRR pressure downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def HRRR_pres_downscaling_classic(pres_in, tmp2m, elev,NextGen_catchment_features):
    """
    Generic function to downscale surface pressure to the WRF-Hydro domain.
    :param : pres_in
    :param : tmp2m -- temperature
    :param : elev -- HRRR elevation grid
    :param NextGen_catchment_features:  globally available
    :return: pres_downscaled - downscaled surface air pressure
    """
    elevDiff = elev - NextGen_catchment_features.elevation_mean.values

    pres_downscaled = pres_in + (pres_in*elevDiff*9.8)/(tmp2m*287.05)

    return  pres_downscaled

# HRRR specific humidity downscaling function based on hypsometirc equation derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def HRRR_hum_downscaling_classic(tmpHumidity, tmp2m, t2dTmp, psfcTmp):
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

# HRRR incoming short wave radiation downscaling function based on topographic adjustment derived from
# https://github.com/NCAR/WrfHydroForcing/blob/main/core/downscale.py
def HRRR_dswr_downscaling_topo_adj(dswr_in, timestamp,NextGen_catchment_features):
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
    tloctm = NextGen_catchment_features.X.values/15.0 + gmt + xt24/60.0
    hrang = ((tloctm - 12.0) * degrad) * 15.0
    xxlat = NextGen_catchment_features.Y.values * degrad
    coszen = np.sin(xxlat) * math.sin(declin) + np.cos(xxlat) * math.cos(declin) * np.cos(hrang)

    # Reset temporary variables to free up memory.
    tloctm = None
    xxlat = None

    return coszen, hrang

# HRRR topography shortwave radiation adjustment based on time of day derived from
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

    xxlat = NextGen_catchment_features.Y.values*degrad

    # Sanity checking on incoming shortwave grid.
    SWDOWN = dswr_in
    SWDOWN[np.where(SWDOWN < 0.0)] = 0.0
    SWDOWN[np.where(SWDOWN >= 1400.0)] = 1400.0
    COSZEN[np.where(COSZEN < 1E-4)] = 1E-4

    corr_frac = np.empty([ny], np.int64)
    diffuse_frac = np.empty([ny], np.int64)
    corr_frac[:] = 0
    diffuse_frac[:] = 0

    indTmp = np.where((NextGen_catchment_features.slope_mean.values == 0.0) &
                      (SWDOWN <= 10.0))
    corr_frac[indTmp] = 1

    term1 = np.sin(xxlat) * np.cos(hrang2d)
    term2 = ((0 - np.cos(NextGen_catchment_features.aspect_c_mean.values)) *
             np.sin(NextGen_catchment_features.slope_mean.values))
    term3 = np.sin(hrang2d) * (np.sin(NextGen_catchment_features.aspect_c_mean.values) *
                               np.sin(NextGen_catchment_features.slope_mean.values))
    term4 = (np.cos(xxlat) * np.cos(hrang2d)) * np.cos(NextGen_catchment_features.slope_mean.values)
    term5 = np.cos(xxlat) * (np.cos(NextGen_catchment_features.aspect_c_mean.values) *
                             np.sin(NextGen_catchment_features.slope_mean.values))
    term6 = np.sin(xxlat) * np.cos(NextGen_catchment_features.slope_mean.values)

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

def Python_ExactExtract_Coverage_Fraction_Weights(HRRR_grib2_file, hyfab_file, output_dir):

    # Initalize gdal gtiff driver for HRRR data
    driver = gdal.GetDriverByName("GTiff")

    # Open HRRR grib2 dataset using GDAL library
    dataset = gdal.Open(HRRR_grib2_file)

    # Get HRRR surface pressure raster band
    band = dataset.GetRasterBand(62)

    ## Initalize HRRR tif file were creating for the transformed
    ## CONUS subset of a HRRR raster
    hrrr_tif = join(output_dir, "HRRR_weights.tif")

    # Assign HRRR met variables to use for regridding techniques
    HRRR_met_vars = ["PRES_surface","HGT_surface","TMP_2maboveground","SPFH_2maboveground","UGRD_10maboveground","VGRD_10maboveground","PRATE_surface","DSWRF_surface","DLWRF_surface"]

    ## Now create a geotiff dataset of the CONUS HRRR raster subset
    dst_ds = driver.Create(hrrr_tif,band.ReadAsArray().shape[1],band.ReadAsArray().shape[0],len(HRRR_met_vars),gdal.GDT_Float32)

    ## Set HRRR geometry, grid (pixel) spacing for TIF file, and also
    ## account for in coordinate reference system projection
    dst_ds.SetGeoTransform(dataset.GetGeoTransform())
    dst_ds.SetProjection(dataset.GetProjection())

    # Get HRRR surface pressure data and write to tif file
    data = np.array(band.ReadAsArray(),dtype=np.float32)
    dst_ds.GetRasterBand(1).WriteArray(data)
    # Set raster band HRRR variable name
    dst_ds.GetRasterBand(1).SetDescription('PRES_surface')

    # Saves data to disk
    dst_ds.FlushCache()
    # Close geotiff file for use in ExactExtract
    dst_ds = None

    # Define raster wrapper for HRRR meteorological variable
    # and specify nc_file attribute to be True. Otherwise,
    # this function will expect a .tif file. Assign data for dict variable
    rsw = GDALRasterWrapper(hrrr_tif,band_idx=1)

    # hydrofabric raster dataset to regrid forcings
    # based on user subset
    dsw = GDALDatasetWrapper(hyfab_file)

    # Define output writer and coverage fraction weights
    # output file
    EE_coverage_fraction_csv = os.path.join(output_dir + "HRRR_ExactExtract_Weights.csv")
    writer = CoverageWriter(EE_coverage_fraction_csv, dsw)

    # Process the data and produce the coverage fraction
    # weights file between the hydrofabric and HRRR data
    processor = CoverageProcessor(dsw, writer, rsw)
    processor.process()

    # Flush changes to disk
    writer = None

    # Since we have finished using the HRRR tif file
    # we can now just remove file from system and save
    # disk storage on server
    os.remove(hrrr_tif)

    # Return the pathway of the coverage fraction weight file
    return EE_coverage_fraction_csv

def python_ExactExtract_weights(HRRR_grib2_file,file_index_id,output_root,hyfabfile,forcing_metadata,bias_calibration,downscaling,AnA,HRRR_weights):

    # Initalize pandas dataframe to save results to csv file
    csv_results = pd.DataFrame([])

    # Assign HRRR met variables to use for regridding techniques
    HRRR_met_vars = ["PRES_surface","HGT_surface","TMP_2maboveground","SPFH_2maboveground","UGRD_10maboveground","VGRD_10maboveground","PRATE_surface","DSWRF_surface","DLWRF_surface"]

    # Initalize which raster bands we will will extract
    # in the HRRR grib2 file based on forecast hour
    if(HRRR_grib2_file.split(".")[-2].split('f')[-1] == '00'):
        HRRR_band_rasters = [62,63,71,73,77,78,84,123,124]
    else:
        HRRR_band_rasters = [62,63,71,73,77,78,84,126,127]

    # Initalize gdal gtiff driver before HRRR variable loop
    driver = gdal.GetDriverByName("GTiff")

    # Open HRRR grib2 dataset using GDAL library
    # before the HRRR variable loop
    dataset = gdal.Open(HRRR_grib2_file)

    # Grab the first raster band from HRRR grib2 file
    # to obtain HRRR metadata
    band = dataset.GetRasterBand(1)

    # Grab the first HRRR raster band in the dataset and
    # extract metadata into array to get datetime info
    metadata = np.array(list(band.GetMetadata().items()))

    # Extract the metadata reference and forecast time
    # to create current HRRR forecast timestamp, this will be
    # needed to disscet within the HRRR bias corrections and
    # downscaling functions
    timestamp = pd.to_datetime(metadata[4][1].split(' ')[5].split('=')[1][:-1]) + pd.TimedeltaIndex([float(metadata[3][1].split(' ')[0])],'s')

    # get current hour of HRRR forecast cycle in metadata,
    # this is needed within HRRR bias correction and downscaling functions
    current_output_step = float(metadata[3][1].split(' ')[0])/3600.0

    # get hour of the day based on current HRRR timestamp, this is needed
    # within the HRRR bias correction and downscaling functions
    hour = timestamp.hour[0]

    # Initalize variables needed for manual aerial weight calculation
    HRRR_data = np.zeros((len(HRRR_met_vars),band.ReadAsArray().shape[0],band.ReadAsArray().shape[1]))
    EE_data_sum = np.zeros((len(HRRR_met_vars),len(HRRR_weights)))
    EE_coverage_fraction_sum = EE_data_sum.copy()[0,:]

    # Get hourly timestamp and forecast hour for respective daily forecast cycle
    hr_cycle = HRRR_grib2_file.split('/')[-1].split('.')[1].split('t')[1]
    forecast_hr = HRRR_grib2_file.split('/')[-1].split('.')[2].split('sfc')[1]
    date_time_cycle = HRRR_grib2_file.split('/')[-3].split('.')[-1]

    # Since HRRR hourly forecast cycles extend between 18-48 hours depending on
    # the hour of the day (6z cycles - 48 hr forecast; any other hour - 18 hr forecast)
    # then we will use a remainder function to see whether or not the number of hours for
    # a given forecast cycle is either 18 or 48 hours for HRRR bias calibration funtions
    remainder = float(hr_cycle.split('z')[0]) % 6.0
    if(remainder == 0.0):
        n_fcst_hr = 48.0
    else:
        n_fcst_hr = 18.0

    # loop over each meteorological variable and get data
    # from each gdal raster to manually regrid
    for i in np.arange(len(HRRR_met_vars)):
        # Define which variable we are looking
        # to regrid for HRRR forcings
        variable = HRRR_met_vars[i]
        # Assign HRRR band to extract grib2 data from
        # based on the variable assigned in the loop
        HRRR_band = HRRR_band_rasters[i]
        # Get HRRR data pertaining to variable in the loop
        band = dataset.GetRasterBand(HRRR_band)
        # Get CONUS susbet of the HRRR raster, where grid spacing is even
        # and NextGen data of interest. Flag for unit conversion for temperature.
        if(variable == "TMP_2maboveground"):
            # Convert HRRR grib2 temp data from Celsius to Kelvin
            HRRR_data[i,:,:] = np.array(band.ReadAsArray() + 273.15,dtype=np.float32)
        # Grib2 banding for radiative fluxes is random, so we need to
        # account for variability in the grib2 band assignment
        elif(variable == "DSWRF_surface" or variable == "DLWRF_surface"):
            # Get band and find variable name in metadata. If the name
            # doesnt match expected radiative fluxes then the offset for
            # the file is by 3 bands
            band = dataset.GetRasterBand(HRRR_band)
            variable_name = np.array(list(band.GetMetadata().items()))[0][1].split(' [')[0]
            if(variable_name != 'Downward short-wave radiation flux' or variable_name != 'Downward long-wave radiation flux'):
                band = dataset.GetRasterBand(HRRR_band+3)
                HRRR_data[i,:,:] = np.array(band.ReadAsArray(),dtype=np.float32)
            else:
                HRRR_data[i,:,:] = np.array(band.ReadAsArray(),dtype=np.float32)
        else:
           HRRR_data[i,:,:] = np.array(band.ReadAsArray(),dtype=np.float32)

    # Now loop through EE weights and raster indices and
    # calculate coverage fraction summation
    for row in zip(HRRR_weights.index, HRRR_weights['row'], HRRR_weights['col'], HRRR_weights['coverage_fraction']):
        # Flag to discard missing HRRR grid cell
        # data from aerial weight average
        if(HRRR_data[0,int(row[1]),int(row[2])] > 0.0):
            # Loop over each HRRR met variable and calculate
            # coverage fraction summation (value * coverage fraction)
            for var in np.arange(len(HRRR_met_vars)):
                EE_data_sum[var,row[0]] += (HRRR_data[var,int(row[1]),int(row[2])]*row[3])
            # Account for coverage fraction with available data
            EE_coverage_fraction_sum[row[0]] += row[3]
    # Once summation is finished for all met variables
    # then we groupby the catchment ids and  calculate
    # coverage fraction weighted mean (summation/coverage fraction total)
    # over each met variable
    var_loop = 0
    for var in HRRR_met_vars:
        HRRR_weights[var] = np.array(EE_data_sum[var_loop,:],dtype=float)
        var_loop += 1

    # Add coverage fraction that accounted for only available data
    # to the dataframe before grouping by catchment ids
    HRRR_weights['EE_coverage_fraction'] = EE_coverage_fraction_sum[:]
    HRRR_weights = HRRR_weights.groupby('divide_id').sum()
    HRRR_weights['cat-id'] = np.array(HRRR_weights.index.values)

    # get datetime of forcing file for global time variable from netcdf file
    # which is in seconds since HRRR reference date and the date-time string
    time = np.zeros(len(HRRR_weights))
    time[:] = (timestamp - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()

    HRRR_weights['time'] = time

    # Finish aerial weighted calculation
    for var in HRRR_met_vars:
        HRRR_weights[var] = (HRRR_weights[var]/HRRR_weights['EE_coverage_fraction'])

    # Now sort the dataframe columnds and drop the groupby
    # index before we return the dataframe to thread
    csv_results = HRRR_weights[['cat-id','time','PRATE_surface','HGT_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]

    ################ Once HRRR regridding is complete, we now call ##############
    ################ the bias correcation and downscaling functions ###########
    ################ sequentially according to NCAR documentation ############

    ############### This is where we call HRRR bias correction function  #########
    if(bias_calibration):
        csv_results = HRRR_bias_correction(csv_results,timestamp[0],current_output_step,forcing_metadata,n_fcst_hr,AnA)

    ############### This is where we call HRRR downscaling function  #########
    if(downscaling):
        csv_results = HRRR_downscaling(csv_results,timestamp[0],forcing_metadata)

    ####################################################################

    csv_results = csv_results.reset_index(drop=True)


    return csv_results


def process_sublist(data : dict, lock: Lock, num: int, EE_results, output_root, hyfabfile, forcing_metadata, weights, bias_calibration, downscaling, AnA):
    num_files = len(data["forcing_files"])    
    
    # Read in ExactExtract coverage fraction weights file
    HRRR_weights = pd.read_csv(weights)

    # Initalize pandas dataframe to save the
    # regridded HRRR ExactExtract results from
    # each HRRR file we loop through
    EE_df_final = pd.DataFrame()

    for i in range(num_files):
        # extract forcing file and file index
        HRRR_grib_file = data["forcing_files"][i]
        file_index_id = data["file_index"][i]

        # Call python ExactExtract routine to directly extract 
        # HRRR regridded results to global HRRR variables from the
        # pregenerated weights file calculated from NextGen hydrofabric
        EE_df = python_ExactExtract_weights(HRRR_grib_file,file_index_id,output_root,hyfabfile,forcing_metadata,bias_calibration,downscaling,AnA,HRRR_weights.copy())
      
        # concatenate the regridded data to threads final dataframe
        EE_df_final = pd.concat([EE_df_final,EE_df])

        # Print how far thread is along with regridding all HRRR data
        print("Thread (" + str(num) + ") is " + str((i+1)/num_files*100) + '% complete')
     
    # collect the garbage within the thread before sending results back to main
    # thread to maximize our RAM as much as possible
    gc.collect()

    # Put regridded results into thread queue to return to main thread
    EE_results.put(EE_df_final)

def NextGen_Forcings_HRRR(output_root, HRRR_directory, forecast_start_time, AnA, netcdf, csv, hyfabfile, hyfabfile_parquet, weights_file, bias_calibration, downscaling, num_processes):

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
    cat_df_full = gpd.read_file(hyfabfile,layer='divides')
    g = [i for i in cat_df_full.geometry]
    h = [i for i in cat_df_full.divide_id]
    n_cats = len(g)
    num_catchments = n_cats
    print("number of catchments = {}".format(n_cats))

    start_time = pd.period_range(start=forecast_start_time,end=forecast_start_time,freq='H')
    if(AnA):
        datafiles = []
        # User selected Analysis and Assimilation configuration
        # for HRRR data, so we must retrieve a 28-hour look back
        # period from the forecast start time user selected
        for i in range(28):
            hr_look_back = start_time - pd.Timedelta(hours=i+1)
            hr_look_back_dir = join(HRRR_directory, 'hrrr.'+hr_look_back.strftime('%Y%m%d')[0] + '/')
            hr_look_back_file = glob.glob(hr_look_back_dir + '/**/hrrr.t' + hr_look_back.strftime('%H')[0] + 'z.wrfsfcf01.grib2')
            datafiles.append(hr_look_back_file[0])
    else:
        # Find the HRRR short range (18 hours) forecast cycle
        # based on user start time
        print(join(HRRR_directory, 'hrrr.'+start_time.strftime('%Y%m%d')[0] + '/conus/'))
        print('hrrr.t' + start_time.strftime('%H')[0] + '*wrfsfc*.grib2')
        datafiles = glob.glob(join(HRRR_directory, 'hrrr.'+start_time.strftime('%Y%m%d')[0] + '/conus/') + 'hrrr.t' + start_time.strftime('%H')[0] + '*wrfsfc*.grib2')

    print("number of forcing files = {}".format(len(datafiles)))
    #process data with time ordered
    datafiles.sort()

    if(bias_calibration):
        print('Will perform HRRR bias calibration on forcing data')
    if(downscaling):
        print('Will perform HRRR downscaling on forcing data')

    # Get number of HRRR files to loop through
    num_files = len(datafiles)

    # Now we want to create a variable to index the datafile 
    # that are already sorted. This index will assign data
    # to global meterological variables below
    file_index = np.arange(num_files)

    # HRRR reference time to use with
    # respect to HRRR forecast time
    ref_date = pd.Timestamp("1970-01-01 00:00:00")

    # Need to obtain HRRR projection to transform the NextGen
    # hydrofabric coordinate reference system
    dataset = gdal.Open(datafiles[1])
    projection = dataset.GetProjection()
    # Need to reproject the hydrofabric crs to the meteorological forcing
    # dataset crs for ExactExtract to properly regrid the data
    hyfabfile_final = join(output_root,"hyfabfile_final.json")
    hyfab_data = gpd.read_file(hyfabfile,layer='divides')
    hyfab_data = hyfab_data.to_crs(projection)

    # Now sort the catchment id values and save the geopackage file
    # into a geojson file
    hyfab_data = hyfab_data.sort_values('divide_id')
    hyfab_data = hyfab_data.reset_index()
    hyfab_data = hyfab_data.drop('index',axis=1)    
    hyfab_data.to_file(hyfabfile_final,driver="GeoJSON")

    # Flag to see if user has already provided the hydrofabric parquet
    # file that is required to process forcing metadata for
    # NCAR bias/calibration functionality if selected
    if(hyfabfile_parquet != None):
        forcing_metadata = pd.read_parquet(hyfabfile_parquet)
        forcing_metadata = forcing_metadata[['divide_id', 'elevation_mean', 'slope_mean','aspect_c_mean','X', 'Y']]
        forcing_metadata = forcing_metadata.sort_values('divide_id')
        forcing_metadata = forcing_metadata.reset_index()        

    # Flag to see if user has already provided an ExactExtract
    # coverage weights file, otherwise go ahead and produce the file
    if(weights_file != None):
        weights = weights_file
    else:    
        # Generate the ExactExtract Coverage Fraction Weights File
        weights = Python_ExactExtract_Coverage_Fraction_Weights(datafiles[1], hyfabfile_final, output_root)

    #generate the data objects for child processes
    file_groups = np.array_split(np.array(datafiles), num_processes)
    file_index_groups = np.array_split(file_index, num_processes)

    process_data = []
    process_list = []
    lock = Lock()

    # Initalize thread storage to return to main program
    EE_results = Queue()

    for i in range(num_processes):
        # fill the dictionary with HRRR grib2 file and its indices
        data = {}
        data["forcing_files"] = file_groups[i]
        data["file_index"] = file_index_groups[i]
      
        #append to the list
        process_data.append(data)

        p = Process(target=process_sublist, args=(data, lock, i, EE_results, output_root, hyfabfile_final, forcing_metadata, weights, bias_calibration, downscaling, AnA))

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
        # of regridded HRRR data
        create_ngen_netcdf(final_df,netcdf_dir,num_catchments,num_files)

    if(csv):
        # generate NextGen csv formatted files for each catchment id
        # within the hydrofabric file
        create_ngen_csv_catchments(final_df, num_processes, csv_dir, num_catchments,num_files)

    # Now clean up I/O files from the script to free up memory for the user
    # Remove the temporary hydrofabric file
    os.remove(hyfabfile_final)
