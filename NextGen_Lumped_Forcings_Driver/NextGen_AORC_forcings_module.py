import multiprocessing
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Process, Lock, Queue, Manager, current_process
from multiprocessing.pool import ThreadPool
import numpy as np
import geopandas as gpd
import netCDF4 as nc4
from scipy.spatial import cKDTree
from shapely.geometry import Point
import os
import glob
from pathlib import Path
from os.path import join
import argparse
import pandas as pd
import time
import datetime
import re
from exactextract import GDALDatasetWrapper, GDALRasterWrapper, CoverageProcessor, CoverageWriter, Operation, MapWriter, FeatureSequentialProcessor, GDALWriter
from osgeo import gdal
import ssl
import xarray as xr
import gc
import wget
import s3fs
import dask
import dask.array
import sys
import zarr
import resource
from osgeo import osr
import json
import shutil
import threading
from functools import lru_cache


#dask.config.set(pool=ThreadPool(16))

# Thread-safe cache with size limit
class YearCache:
    def __init__(self, max_size=10):
        #self.mangager = Manager()
        self.max_size = max_size
        self._cache = {}
        self._lock = threading.Lock()
        
    def get(self, year):
        with self._lock:
            return self._cache.get(year)
            
    def set(self, year, data):
        with self._lock:
            if len(self._cache) >= self.max_size and year not in self._cache:
                # Remove least recently used year
                lru_year = min(self._cache.keys())
                del self._cache[lru_year]
            self._cache[year] = data
            
    def contains(self, year):
        with self._lock:
            return year in self._cache

_year_cache = YearCache(max_size=10)

def get_spatial_indices(ds, bounds):
    """Helper function to get indices for spatial subset"""
    lat_mask = (ds.latitude >= bounds[1]) & (ds.latitude <= bounds[3])
    lon_mask = (ds.longitude >= bounds[0]) & (ds.longitude <= bounds[2])
    return {
        'latitude': lat_mask,
        'longitude': lon_mask
    }

def get_cached_subset(year, hyfabfile, AORC_met_vars):
    cached_data = _year_cache.get(year)
    if cached_data is not None:
        return cached_data
        
    _s3 = s3fs.S3FileSystem(anon=True)
    zarr_path = f"s3://noaa-nws-aorc-v1-1-1km/{year}.zarr"
    store = s3fs.S3Map(root=zarr_path, s3=_s3, check=False)
    
    ds = xr.open_zarr(store)
    ds_subset = subset_zarr_by_bounds(ds, hyfabfile)
    ds_subset = ds_subset[AORC_met_vars]
    
    print(f"Computing cached subset for {year}...")
    t0 = time.time()
    ds_subset = ds_subset.compute(
        scheduler='threads',
        num_workers=24,
        optimize_graph=True
    )
    t1 = time.time()
    print(f"finished computing cached subset for {year}: {t1-t0:.3f}s")
    
    _year_cache.set(year, ds_subset)
    return ds_subset

def process_years_chunk(years_chunk, data, lock, shared_results, thread_num, 
                       met_dataset_pathway, output_root, hyfabfile, weights,
                       add_offset, scale_factor, AORC_met_vars, AORC_missing_value,
                       aorc_ncfile, NN_table, gapfill, zarr_data):
    """Process a chunk of years for a given thread"""
    EE_df_final = pd.DataFrame()
    time_subset = data["time_periods"]
    
    for year in years_chunk:
        year_periods = time_subset[time_subset.year == year]
        
        for timestamp in year_periods:
            EE_df = python_ExactExtract_zarr(
                met_dataset_pathway,
                hyfabfile,
                add_offset,
                scale_factor,
                zarr_data['variables'],
                NN_table,
                gapfill,
                timestamp.to_timestamp(),
                len(time_subset)
            )
            EE_df_final = pd.concat([EE_df_final, EE_df])
            
    shared_results.append(EE_df_final)
        
def generate_nearest_neighbor_correction_table(NextGen_hyfabfile, AORC_file, add_offset, scale_factor, AORC_met_vars):

    # Quickly generate ExactExtract results to see if any catchments had missing data from AORC domain
    if AORC_file.startswith('s3://'):
        _s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=AORC_file, s3=_s3, check=False)
        ds = xr.open_dataset(store, engine='zarr', chunks={'time': 'auto'})
        first_timestamp = pd.Timestamp(ds.time.values[0])
        EE_df = python_ExactExtract_zarr(AORC_file, NextGen_hyfabfile, add_offset, scale_factor, AORC_met_vars, None, False, first_timestamp)
        ds.close()
    else:
       EE_df = python_ExactExtract(AORC_file, NextGen_hyfabfile, add_offset, scale_factor, AORC_met_vars, None, False)
    # See if there is any missing catchment data for given hydrofabric file user specified 
    try:

        # Read the hydrofabric file for analysis
        NextGen_hyfabfile = gpd.read_file(NextGen_hyfabfile)
        
        # Find missing values for catchments in VPU
        idx = EE_df.APCP_surface.isnull()
        missing_data_ids = EE_df.loc[idx,'cat-id'].values
        missing_NextGen_hyfabfile = NextGen_hyfabfile.loc[NextGen_hyfabfile['divide_id'].isin(missing_data_ids)]

        # Find catchments with available AORC data
        available_data_ids = EE_df.loc[idx==False,'cat-id'].values
        available_NextGen_hyfabfile = NextGen_hyfabfile.loc[NextGen_hyfabfile['divide_id'].isin(available_data_ids)]

        missing_NextGen_hyfabfile['nearest_neighbor_id'] = missing_NextGen_hyfabfile.divide_id

        # Using the geometries of the missing catchment ids, find the closest geometries for each missing AORC catchment
        missing_points_x, missing_points_y = missing_NextGen_hyfabfile.geometry.centroid.x.values, missing_NextGen_hyfabfile.geometry.centroid.y.values
        point_loop = 0
        for index, row in missing_NextGen_hyfabfile.iterrows():
            missing_point = Point(missing_points_x[point_loop],missing_points_y[point_loop])
            polygon_index = available_NextGen_hyfabfile.distance(missing_point).sort_values().index[0]
            missing_NextGen_hyfabfile.at[index,'nearest_neighbor_id'] = available_NextGen_hyfabfile.loc[polygon_index,'divide_id']
            point_loop += 1

        # Generate nearest neighbor table for missing catchment ids to gapfill AORC data
        NN_df = pd.DataFrame([])
        NN_df['cat-id'] = missing_NextGen_hyfabfile['divide_id'].values
        NN_df['nearest_neighbor_id'] = missing_NextGen_hyfabfile['nearest_neighbor_id'].values

        return NN_df

    # If no missing data, then return None flag back
    except:

        return None

def nearest_neighbor_correction(csv_results, EE_NN_df, AORC_met_vars):
    # loop over EE results and gapfill missing catchment ids from nearest neighbor table
    EE_cat_ids = csv_results['cat-id'].values
    for row in zip(EE_NN_df['cat-id'],EE_NN_df['nearest_neighbor_id']):
        id_index = np.where(EE_cat_ids==row[0])[0]
        if(len(id_index) != 0):
            NN_index = np.where(EE_cat_ids==row[1])[0][0]
            csv_results.iloc[id_index[0],2:len(AORC_met_vars)+2] = csv_results.loc[NN_index,AORC_met_vars]
    return csv_results


def get_date_time(path):
    """
    Extract the date-time from the file path
    """
    path = Path(path)
    name = path.stem
    date_time = name.split('.')[0]
    date_time = date_time.split('_')[2]  #this index may depend on the naming format of the forcing data
    date_time = re.sub('\D','',date_time)
    return date_time

def process_csv_ids(data : dict, lock: Lock, num: int, csv_dir):
    # Get the number of unique catchment ids in each thread
    cat_ids = np.unique(data['cat_ids'][:,0])
    num_cats = len(cat_ids)

    # Loop through each catchment and create/save csv
    for i in range(num_cats):
        csv_df = pd.DataFrame([])
        csv_df['Time'] = data['Time'][i,:]
        csv_df['RAINRATE'] = data['APCP_surface'][i,:]/3600.0
        csv_df['Q2D'] = data['SPFH_2maboveground'][i,:]
        csv_df['T2D'] = data['TMP_2maboveground'][i,:]
        csv_df['U2D'] = data['UGRD_10maboveground'][i,:]
        csv_df['V2D'] = data['VGRD_10maboveground'][i,:]
        csv_df['LWDOWN'] = data['DLWRF_surface'][i,:]
        csv_df['SWDOWN'] = data['DSWRF_surface'][i,:]
        csv_df['PSFC'] = data['PRES_surface'][i,:]
        csv_df = csv_df.sort_values(by=['Time'])
        NextGen_csv = join(csv_dir,str(data['cat_ids'][i,0])+'.csv')
        csv_df.to_csv(NextGen_csv,index=False)

        if(i == 0):
            csv_length = len(csv_df)
        else:
            if(len(csv_df) != csv_length):
                print(cat_ids[i] + ' likely missing some of time series')

        print(f'\rThread {num}: {(i+1)/num_cats*100:.1f}% complete updating catchment csvs', end='', flush=True)
    # collect the garbage within the thread before sending results back to main
    # thread to maximize our RAM as much as possible
    gc.collect()

def process_annual_csv_ids(data : dict, lock: Lock, num: int, csv_dir):
    # Get the number of unique catchment ids in each thread
    cat_ids = np.unique(data['cat_ids'])
    num_cats = len(cat_ids)
    year = pd.DatetimeIndex([data['Time'][0]]).year[0]

    print(f"Thread {num}: Processing {len(data['cat_ids'])} records for {num_cats} unique catchments")
    print(f"Thread {num}: Time range: {data['Time'].min()} to {data['Time'].max()}")
    print(f"Thread {num}: Unique timestamps: {len(np.unique(data['Time']))}")

    # Loop through each catchment and create/save csv
    
    for i in range(num_cats):
        idx = np.where(data['cat_ids'] == cat_ids[i])[0]
        csv_df = pd.DataFrame([])
        csv_df['Time'] = data['Time'][idx]
        csv_df['RAINRATE'] = data["APCP_surface"][idx]/3600.0
        csv_df['Q2D'] = data['SPFH_2maboveground'][idx]
        csv_df['T2D'] = data['TMP_2maboveground'][idx]
        csv_df['U2D'] = data['UGRD_10maboveground'][idx]
        csv_df['V2D'] = data['VGRD_10maboveground'][idx]
        csv_df['LWDOWN'] = data['DLWRF_surface'][idx]
        csv_df['SWDOWN'] = data['DSWRF_surface'][idx]
        csv_df['PSFC'] = data['PRES_surface'][idx]
        csv_df = csv_df.sort_values(by=['Time'])
        NextGen_csv = join(csv_dir,str(cat_ids[i])+ '_' + str(year) + '.csv')
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

def process_cats(data : dict, lock: Lock, num: int, csv_dir):
    # get number of catchments for a given thread
    num_cats = len(data['cat-ids'])

    for i in range(num_cats):
        # Get annual csv catchment files to combine together
        # and sort them for sequential concatenation
        datafile_path = join(csv_dir, str(data['cat-ids'][i]) +  "*.csv")
        datafiles = glob.glob(datafile_path)
        datafiles.sort()
        # Initalize catchment dataframe
        cat_df = pd.DataFrame([])
        # loop through files and concatenate annual
        # csv datafile and then remove file from directory
        for annual_file in datafiles:
            annual_cat_df = pd.read_csv(annual_file)
            cat_df = pd.concat([cat_df,annual_cat_df])
            os.remove(annual_file)

        # Create and save NextGen catchment csv file
        NextGen_csv = join(csv_dir,str(data['cat-ids'][i])+'.csv')
        cat_df.to_csv(NextGen_csv,index=False)

        # Print how far thread is along with regridding all AORC data
        print("Thread is " + str((i+1)/num_cats*100) + '% complete for creating final csv catchment file')

def create_ngen_cat_csv_CONUS(csv_dir, nextgen_cat_ids, num_processes):
    """
    Combine annual catchment csv files and
    Create NextGen csv files with specified format
    """

    # generate the data objects for child processes for catchment ids
    cat_groups = np.array_split(nextgen_cat_ids, num_processes)

    process_data = []
    process_list = []
    lock = Lock()

    for i in range(num_processes):
        # fill the dictionary with needed at
        data = {}
        data["cat-ids"] = cat_groups[i]
        #append to the list
        process_data.append(data)

        p = Process(target=process_cats, args=(data, lock, i, csv_dir))

        process_list.append(p)

    ##start all processes
    for p in process_list:
        p.start()

    ##wait for termination
    for p in process_list:
        p.join()

    ## free threads and their resources
    for p in process_list:
        p.close()


def create_ngen_cat_csv_annual(final_df, csv_dir, num_processes):
    """
    Create NextGen csv files with specified format
    """

    final_df['Time'] = pd.Timestamp("1970-01-01 00:00:00") + pd.TimedeltaIndex(final_df['time'].values,'s')

    #generate the data objects for child processes for csv files
    id_groups = np.array_split(final_df['cat-id'].values, num_processes)
    time_groups = np.array_split(final_df['Time'].values, num_processes)
    precip_groups = np.array_split(final_df['APCP_surface'].values, num_processes)
    q_groups = np.array_split(final_df['SPFH_2maboveground'].values, num_processes)
    tmp_groups = np.array_split(final_df['TMP_2maboveground'].values, num_processes)
    ugrd_groups = np.array_split(final_df['UGRD_10maboveground'].values, num_processes)
    vgrd_groups = np.array_split(final_df['VGRD_10maboveground'].values, num_processes)
    lw_groups = np.array_split(final_df['DLWRF_surface'].values, num_processes)
    sw_groups = np.array_split(final_df['DSWRF_surface'].values, num_processes)
    pres_groups = np.array_split(final_df['PRES_surface'].values, num_processes)

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
        data["APCP_surface"] = precip_groups[i]  # Changed from RAINRATE
        data["SPFH_2maboveground"] = q_groups[i] # Changed from Q2D
        data["TMP_2maboveground"] = tmp_groups[i] # Changed from T2D
        data["UGRD_10maboveground"] = ugrd_groups[i] # Changed from U2D
        data["VGRD_10maboveground"] = vgrd_groups[i] # Changed from V2D
        data["DLWRF_surface"] = lw_groups[i] # Changed from LWDOWN
        data["DSWRF_surface"] = sw_groups[i] # Changed from SWDOWN
        data["PRES_surface"] = pres_groups[i] # Changed from PSFC
        #append to the list
        process_data.append(data)

        p = Process(target=process_annual_csv_ids, args=(data, lock, i, csv_dir))

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


def create_ngen_cat_csv(final_df, csv_dir, num_catchments,num_forcing_files, num_processes):
    """
    Create NextGen csv files with specified format
    """
    
    from concurrent.futures import ProcessPoolExecutor, as_completed

    final_df['Time'] = pd.Timestamp("1970-01-01 00:00:00") + pd.TimedeltaIndex(final_df['time'].values,'s')


    # Check if we need to restructure data
    expected_rows = num_catchments * num_forcing_files
    actual_rows = len(final_df)

    #print(f"final_df['cat-id'].values: {final_df['cat-id'].values}")
    if actual_rows == expected_rows:
        print("Data is already properly structured for reshaping")
        #generate the data objects for child processes for csv files
        id_groups = np.array_split(np.reshape(final_df['cat-id'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        time_groups = np.array_split(np.reshape(final_df['Time'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        precip_groups = np.array_split(np.reshape(final_df['APCP_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        q_groups = np.array_split(np.reshape(final_df['SPFH_2maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        tmp_groups = np.array_split(np.reshape(final_df['TMP_2maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        ugrd_groups = np.array_split(np.reshape(final_df['UGRD_10maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        vgrd_groups = np.array_split(np.reshape(final_df['VGRD_10maboveground'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        lw_groups = np.array_split(np.reshape(final_df['DLWRF_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        sw_groups = np.array_split(np.reshape(final_df['DSWRF_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)
        pres_groups = np.array_split(np.reshape(final_df['PRES_surface'].values,(num_catchments,num_forcing_files)), num_processes,axis=0)

    else:
        print(f"Restructuring irregular data: {actual_rows} records -> {num_catchments}*{num_forcing_files} structure")
        
        # Get unique catchments and timestamps
        unique_catchments = sorted(final_df['cat-id'].unique())
        unique_times = sorted(final_df['time'].unique())
        
        # Verify dimensions match expected
        if len(unique_catchments) != num_catchments:
            print(f"Warning: Expected {num_catchments} catchments, found {len(unique_catchments)}")
            num_catchments = len(unique_catchments)
        if len(unique_times) != num_forcing_files:
            print(f"Warning: Expected {num_forcing_files} timesteps, found {len(unique_times)}")
            num_forcing_files = len(unique_times)
        
        # Create mapping dictionaries for fast lookup
        cat_to_idx = {cat: i for i, cat in enumerate(unique_catchments)}
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        
        # Initialize 2D arrays with proper fill values
        cat_ids_2d = np.empty((num_catchments, num_forcing_files), dtype=object)
        time_2d = np.empty((num_catchments, num_forcing_files), dtype=object)
        apcp_2d = np.full((num_catchments, num_forcing_files), np.nan)
        spfh_2d = np.full((num_catchments, num_forcing_files), np.nan)
        tmp_2d = np.full((num_catchments, num_forcing_files), np.nan)
        ugrd_2d = np.full((num_catchments, num_forcing_files), np.nan)
        vgrd_2d = np.full((num_catchments, num_forcing_files), np.nan)
        dlwrf_2d = np.full((num_catchments, num_forcing_files), np.nan)
        dswrf_2d = np.full((num_catchments, num_forcing_files), np.nan)
        pres_2d = np.full((num_catchments, num_forcing_files), np.nan)
        
        # Fill 2D arrays
        for _, row in final_df.iterrows():
            cat_idx = cat_to_idx[row['cat-id']]
            time_idx = time_to_idx[row['time']]
            
            cat_ids_2d[cat_idx, time_idx] = row['cat-id']
            time_2d[cat_idx, time_idx] = row['Time']
            apcp_2d[cat_idx, time_idx] = row['APCP_surface']
            spfh_2d[cat_idx, time_idx] = row['SPFH_2maboveground']
            tmp_2d[cat_idx, time_idx] = row['TMP_2maboveground']
            ugrd_2d[cat_idx, time_idx] = row['UGRD_10maboveground']
            vgrd_2d[cat_idx, time_idx] = row['VGRD_10maboveground']
            dlwrf_2d[cat_idx, time_idx] = row['DLWRF_surface']
            dswrf_2d[cat_idx, time_idx] = row['DSWRF_surface']
            pres_2d[cat_idx, time_idx] = row['PRES_surface']
        
        print("Data restructuring complete")
        
        # Split the 2D arrays across processes
        id_groups = np.array_split(cat_ids_2d, num_processes, axis=0)
        time_groups = np.array_split(time_2d, num_processes, axis=0)
        precip_groups = np.array_split(apcp_2d, num_processes, axis=0)
        q_groups = np.array_split(spfh_2d, num_processes, axis=0)
        tmp_groups = np.array_split(tmp_2d, num_processes, axis=0)
        ugrd_groups = np.array_split(ugrd_2d, num_processes, axis=0)
        vgrd_groups = np.array_split(vgrd_2d, num_processes, axis=0)
        lw_groups = np.array_split(dlwrf_2d, num_processes, axis=0)
        sw_groups = np.array_split(dswrf_2d, num_processes, axis=0)
        pres_groups = np.array_split(pres_2d, num_processes, axis=0)

    worker_args = []
    for i in range(num_processes):
        data = {
            "cat_ids": id_groups[i],
            "Time": time_groups[i],
            "APCP_surface": precip_groups[i],
            "SPFH_2maboveground": q_groups[i],
            "TMP_2maboveground": tmp_groups[i],
            "UGRD_10maboveground": ugrd_groups[i],
            "VGRD_10maboveground": vgrd_groups[i],
            "DLWRF_surface": lw_groups[i],
            "DSWRF_surface": sw_groups[i],
            "PRES_surface": pres_groups[i]
        }
        worker_args.append((data, None, i, csv_dir))

    # Delete main data to save RAM
    del(final_df)
    gc.collect()

    executor = ProcessPoolExecutor(max_workers=num_processes)
    try:
        print(f"Submitting {num_processes} CSV writing tasks...")
        futures = [executor.submit(process_csv_ids, *args) for args in worker_args]
        
        # Wait for completion and handle any errors
        for i, future in enumerate(futures):
            try:
                future.result()
                #print(f"Process {i} completed successfully")
            except Exception as e:
                print(f"Process {i} failed with error: {e}")

    finally:
        # Explicit cleanup to prevent garbage collection issues
        executor.shutdown(wait=True)
        
        # Force cleanup of internal thread pool
        if hasattr(executor, '_threads'):
            executor._threads.shutdown(wait=True)
        
        # Delete the executor reference
        del executor

    print("All CSV writing processes completed")

def create_ngen_netcdf_CONUS(aorc_ncfile, netcdf_dir):
    """
    Now combine the annual netcdf files and 
    Create NextGen netcdf file with specified format
    """

    # get datetime of forcing file to append
    # to ExactExtract csv output file
    start_time = get_date_time(aorc_ncfile)

    #create output netcdf file name
    output_path = join(netcdf_dir, "NextGen_forcing_final_"+start_time+".nc")

    # If user requested a multi-month CONUS
    # dataset, then seperate netcdf files were produced
    # and we must combine them into a single file
    if(len(os.listdir(netcdf_dir)) > 1):

        # Now we are going to use xarray to open all of the netcdf files
        #i at once and combine them together and save the contents to the final file
        print("Xarray opening and combining NextGen annual forcing files")
        ds = xr.open_mfdataset(netcdf_dir + "NextGen_forcing*.nc",combine='nested',concat_dim='time', data_vars='minimal', coords='minimal', compat='override')

        # Define Time variable properly to concatenate and save data to a single netcdf file
        ds['Time'] = ds['Time'].astype(np.float64)
        ds.Time.attrs['units'] = "seconds"

        # Define netcdf compression encoding for variables
        # when saving final netcdf file
        comp = dict(zlib=True, complevel=1, shuffle=True, _FillValue=-99999.)
        encoding = {var: comp for var in ['Time','APCP_surface','precip_rate','T2D','U2D','Q2D','U2D','V2D','PSFC','SWDOWN','LWDOWN']}
        print("Saving combined netcdf xarray dataset to netcdf file")
        ds.to_netcdf(output_path,encoding=encoding)
        del(ds)
    # If only one file exists, then rename the batch file
    # as the final file for the given script execution
    else:
        monthly_file = os.listdir(netcdf_dir)[0]
        os.rename(monthly_file,output_path)


def create_ngen_netcdf(aorc_ncfile, final_df, netcdf_dir, num_files, num_catchments):
    """
    Create NextGen netcdf file with specified format
    """

    # get datetime of forcing file to append
    # to ExactExtract csv output file
    start_time = get_date_time(aorc_ncfile)

    # first read AORC metadata in to save to NextGen forcing file
    ds = nc4.Dataset(aorc_ncfile)

    #create output netcdf file name
    output_path = join(netcdf_dir, "NextGen_forcing_"+start_time+".nc")

    #make the data set
    filename = output_path
    filename_out = output_path

     
    # write data to netcdf files
    filename_out = output_path
    ncfile_out = nc4.Dataset(filename_out, 'w', format='NETCDF4')

    #add the dimensions
    time_dim = ncfile_out.createDimension('time', num_files)
    catchment_id_dim = ncfile_out.createDimension('catchment-id', num_catchments)
    string_dim =ncfile_out.createDimension('str_dim', 1)

    # create variables
    cat_id_out = ncfile_out.createVariable('ids', 'str', ('catchment-id'))
    time_out = ncfile_out.createVariable('Time', 'double', ('catchment-id','time',), fill_value=-99999, 
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    APCP_surface_out = ncfile_out.createVariable('APCP_surface', 'f4', ('catchment-id', 'time',), fill_value=-99999,
        chunksizes=(num_catchments,1), zlib=True, complevel=1, shuffle=True)
    precip_rate_out = ncfile_out.createVariable('precip_rate', 'f4', ('catchment-id', 'time',), fill_value=-99999,
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

    #set output netcdf file atributes
    varout_dict = {'time':time_out,
                   'APCP_surface':APCP_surface_out, 'DLWRF_surface':DLWRF_surface_out, 'DSWRF_surface':DSWRF_surface_out,
                   'PRES_surface':PRES_surface_out, 'SPFH_2maboveground':SPFH_2maboveground_out, 'TMP_2maboveground':TMP_2maboveground_out,
                   'UGRD_10maboveground':UGRD_10maboveground_out, 'VGRD_10maboveground':VGRD_10maboveground_out}


    #copy all attributes from input netcdf file
    for name, variable in ds.variables.items():
        if name == 'latitude' or name == 'longitude':
            pass
        else:
            varout_name = varout_dict[name]
            for attrname in variable.ncattrs():
                if name == "time" and attrname == "units":
                    #slight hack here to be compatible with current NetCDFPerFeatureDataProvider
                    # ... change it instead?
                    #see also https://www.unidata.ucar.edu/software/netcdf/time/recs.html
                    setattr(varout_name, "units", "seconds")
                    setattr(varout_name, "epoch_start", "01/01/1970 00:00:00")
                elif (attrname != '_FillValue' and attrname != 'missing_value'):
                    setattr(varout_name, attrname, getattr(variable, attrname))

    #drop the scale_factor and add_offset from the output netcdf forcing file attributes
    for key, varout_name in varout_dict.items():
        if key != 'time':
            try:
                del varout_name.scale_factor
            except: 
                print("No scale factor in forcing files. No keys to tweak for output netcdf")
            try:
                del varout_name.add_offset
            except:
                print("No add offset in forcing files. No keys to tweak for output netcdf")


    #####################################################################

    # set attributes manually for precipitation rate
    setattr(precip_rate_out, 'long_name', 'Precipitation Rate')
    setattr(precip_rate_out, 'short_name', 'RAINRATE')
    setattr(precip_rate_out, 'units', 'mm s-1')
    setattr(precip_rate_out, 'level', 'surface')
    
    #set attributes for additional variables
    setattr(cat_id_out, 'description', 'catchment_id')
   

    # get number of unique NextGen catchments
    # to reshape arrays in dataframe for netcdf variables
    cat_ids = final_df['cat-id'].unique()
    cat_id_out[:] = cat_ids
    num_cats = len(cat_ids)

    # Assign netcdf variables reshaped arrays
    # from Met variables of interest
    print("Reshaping annual arrays for netcdf variables")
    time5 = datetime.datetime.now()
    time_out[:,:] = np.reshape(final_df['time'].values,(num_cats,num_files))
    APCP_surface_out[:,:] = np.reshape(final_df['APCP_surface'].values,(num_cats,num_files))
    precip_rate_out[:,:] = np.reshape(final_df['APCP_surface'].values/3600,(num_cats,num_files))    
    DLWRF_surface_out[:,:] = np.reshape(final_df['DLWRF_surface'].values,(num_cats,num_files))
    DSWRF_surface_out[:,:] = np.reshape(final_df['DSWRF_surface'].values,(num_cats,num_files))    
    PRES_surface_out[:,:] = np.reshape(final_df['PRES_surface'].values,(num_cats,num_files))
    SPFH_2maboveground_out[:,:] = np.reshape(final_df['SPFH_2maboveground'].values,(num_cats,num_files))
    TMP_2maboveground_out[:,:] = np.reshape(final_df['TMP_2maboveground'].values,(num_cats,num_files))
    UGRD_10maboveground_out[:,:] = np.reshape(final_df['UGRD_10maboveground'].values,(num_cats,num_files))
    VGRD_10maboveground_out[:,:] = np.reshape(final_df['VGRD_10maboveground'].values,(num_cats,num_files))
    time6 = datetime.datetime.now()

    ncfile_out.close()
    ds.close()
    print("Finish NextGen annual file creation")

def Python_ExactExtract_Coverage_Fraction_Weights(aorc_file, hyfab_file, AORC_met_vars, output_dir):
    
    # Generate cache filename based on hydrofabric file
    
    #hyfab_test = os.path.splitext(os.path.basename(hyfab_file))[1]
    #print(f"Hyfab_test = {hyfab_test}")
    gage_id = output_dir.split('/')[-1]

    hyfab_name = os.path.splitext(os.path.basename(hyfab_file))[0]
    weights_file = os.path.join(os.path.dirname(output_dir), f"AORC_ExactExtract_Weights_{gage_id}.csv")
    
    # If weights already exist, return the path
    if os.path.exists(weights_file):
        print(f"Using existing weights file: {weights_file}")
        return weights_file

    if aorc_file.startswith('s3://'): #is_zarr
        
        print("Calculating weights file.")
    
        # Set up zarr access
        _s3 = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=aorc_file, s3=_s3, check=False)
        ds = xr.open_dataset(store, engine='zarr', chunks={'time': 'auto'})
        
        # Get grid information
        lats = ds.latitude.values
        lons = ds.longitude.values
        
        # Create temporary raster file for ExactExtract
        temp_raster = os.path.join(output_dir, "temp_raster.tif")
        
        # Get the first variable's data for grid structure
        var_data = ds[AORC_met_vars[0]].isel(time=0).values
        
        # Create geotransform for the raster
        xmin, xmax = lons.min(), lons.max()
        ymin, ymax = lats.min(), lats.max()
        dx = (xmax - xmin) / (len(lons) - 1)
        dy = (ymax - ymin) / (len(lats) - 1)
        geotransform = [xmin, dx, 0, ymax, 0, -dy]
        
        # Create GDAL raster
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = var_data.shape
        dataset = driver.Create(temp_raster, cols, rows, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geotransform)

        # Write data
        dataset.GetRasterBand(1).WriteArray(var_data)
        dataset.FlushCache()
        dataset = None
        ds.close()
        
        # Use existing ExactExtract functionality with temporary raster
        rsw = GDALRasterWrapper(temp_raster)
        dsw = GDALDatasetWrapper(hyfab_file)
        writer = CoverageWriter(weights_file, dsw)
        
        # Process the data and produce weights file
        processor = CoverageProcessor(dsw, writer, rsw)
        processor.process()
        
        # Clean up
        writer = None
        os.remove(temp_raster)
        
    else:
    
        # load AORC netcdf file into gdal dataframe to
        # partition out meterological variables into rasters
        aorc = gdal.Open(aorc_file)
        # Get gdal sub-datasets, which will seperate each AORC
        # variable into their own raster wrapper
        nc_rasters = aorc.GetSubDatasets()
        # Get variable name in netcdf file
        variable = nc_rasters[-1][0].split(":")[-1]
        # Get the gdal netcdf syntax for netcdf variable
        # Example syntax: 'NETCDF:"AORC-OWP_2012050100z.nc4":APCP_surface'
        nc_dataset_name = nc_rasters[-1][0]
        # Define raster wrapper for AORC meteorological variable
        # and specify nc_file attribute to be True. Otherwise,
        # this function will expect a .tif file. Assign data for dict variable
        rsw = GDALRasterWrapper(nc_dataset_name)
        # For each AORC met variable, we must redefine the
        # hydrofabric raster dataset to regrid forcings
        # based on user operation below
        dsw = GDALDatasetWrapper(hyfab_file)
        # Define output writer and coverage fraction weights 
        # output file
        #weights_file = os.path.join(output_dir + "AORC_ExactExtract_Weights.csv")
        writer = CoverageWriter(weights_file, dsw)        
        
        # Original netCDF handling
        aorc = gdal.Open(aorc_file)
        nc_rasters = aorc.GetSubDatasets()
        variable = nc_rasters[-1][0].split(":")[-1]
        nc_dataset_name = nc_rasters[-1][0]
        
        # Use existing ExactExtract functionality
        rsw = GDALRasterWrapper(nc_dataset_name)
        dsw = GDALDatasetWrapper(hyfab_file)
        writer = CoverageWriter(weights_file, dsw)
        
        processor = CoverageProcessor(dsw, writer, rsw)
        processor.process()
    
        # Process the data and produce the coverage fraction
        # weights file between the hydrofabric and AORC data
        processor = CoverageProcessor(dsw, writer, rsw)
        processor.process()

        # Flush changes to disk
        writer = None

    # Both conditionals produce the same format weights file at weights_file path
    if not os.path.exists(weights_file):
        raise RuntimeError(f"Failed to generate weights file at {weights_file}")
      
    return weights_file    

def python_ExactExtract_Manual_Weights(aorc_file, AORC_weights, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, NN_table, gapfill):
    
    # load AORC netcdf file into gdal dataframe to
    # partition out meterological variables into rasters
    aorc = gdal.Open(aorc_file)
    # Get gdal sub-datasets, which will seperate each AORC
    # variable into their own raster wrapper
    nc_rasters = aorc.GetSubDatasets()
    # get datetime of forcing file to append
    # to ExactExtract csv output file
    date_time = get_date_time(aorc_file)

    # Open the netcdf file and get latitude and longitude dimensions to
    # initalize AORC data array
    nc_file = nc4.Dataset(aorc_file)
    lat_shape = nc_file.variables['latitude'].shape[0]
    lon_shape = nc_file.variables['longitude'].shape[0]
    nc_file.close()
    # Initalize variables needed for manual aerial weight calculation
    AORC_data = np.zeros((len(AORC_met_vars),lat_shape,lon_shape),dtype=float)
    EE_data_sum = np.zeros((len(AORC_met_vars),len(AORC_weights)),dtype=float)
    EE_coverage_fraction_sum = EE_data_sum.copy()[0,:]

    # loop over each meteorological variable and get data
    # from each gdal raster to manually regrid
    for i in np.arange(len(AORC_met_vars)):
        # Get variable name in netcdf file
        variable = nc_rasters[i][0].split(":")[-1]
        # Get the gdal netcdf syntax for netcdf variable
        # Example syntax: 'NETCDF:"AORC-OWP_2012050100z.nc4":APCP_surface'
        nc_dataset_name = nc_rasters[i][0]
        # Grab data from each netcdf raster to thread AORC data array
        AORC_data[i,:,:] = gdal.Open(nc_dataset_name).ReadAsArray()
    # Now loop through EE weights and raster indices and
    # calculate coverage fraction summation
    for row in zip(AORC_weights.index, AORC_weights['row'], AORC_weights['col'], AORC_weights['coverage_fraction']):
        # Flag to discard missing AORC grid cell
        # data from aerial weight average
        if(AORC_data[0,int(row[1]),int(row[2])] != AORC_missing_value):
            # Loop over each AORC met variable and calculate
            # coverage fraction summation (value * coverage fraction)
            for var in np.arange(len(AORC_met_vars)):
                EE_data_sum[var,row[0]] += (AORC_data[var,int(row[1]),int(row[2])]*row[3])
            # Account for coverage fraction with available data
            EE_coverage_fraction_sum[row[0]] += row[3]
    # Once summation is finished for all met variables
    # then we groupby the catchment ids and  calculate
    # coverage fraction weighted mean (summation/coverage fraction total)
    # over each met variable
    var_loop = 0
    for var in AORC_met_vars:
        AORC_weights[var] = EE_data_sum[var_loop,:]
        var_loop += 1
    # Add coverage fraction that accounted for only available data
    # to the dataframe before grouping by catchment ids
    AORC_weights['EE_coverage_fraction'] = EE_coverage_fraction_sum[:]
    # Groupby the catchment ids and sum up the values as part of the aerial
    # weighted calculation used here
    AORC_weights = AORC_weights.groupby('divide_id').sum()
    AORC_weights['cat-id'] = np.array(AORC_weights.index.values)
    # get seconds since AORC reference date for time array in
    # pandas dataframe
    time = np.zeros(len(AORC_weights))
    time[:] = (pd.Timestamp(datetime.datetime.strptime(date_time,'%Y%m%d%H')) - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()
    AORC_weights['time'] = time
    met_loop = 0
    # Loop over and finalize aerial weighted calculation while
    # accounting for scale factor and add offset values in AORC
    # data that is not accounted for within the gdal library
    for var in AORC_met_vars:
        AORC_weights[var] = (AORC_weights[var]/AORC_weights['EE_coverage_fraction'])*scale_factor[met_loop] + add_offset[met_loop]
        met_loop += 1
    # Now sort the dataframe columnds and drop the groupby
    # index before we return the dataframe to thread
    csv_results = AORC_weights[['cat-id','time','APCP_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]
    csv_results = csv_results.reset_index(drop=True)

    # call the nearest neighbor function to gapfill
    # coastal catchments and islands with the nearest
    # NextGen catchment with available AORC data if
    # the given hydrofabric file indicated catchments with
    # missing AORC data
    if(gapfill):
        csv_results = nearest_neighbor_correction(csv_results, NN_table, AORC_met_vars)

    return csv_results



def subset_zarr_by_bounds(ds, hyfabfile, buff=0.1):
    """
    Subset zarr data by geographical bounds with buffer
    bounds: (minx, miny, maxx, maxy)
    buff: degrees
    """
    t0 = time.time()
    gdf = gpd.read_file(hyfabfile)
    t1 = time.time()
    print(f"Reading hyfabfile: {t1-t0:.3f}s")
    
    bounds = gdf.total_bounds    
    
    #t0=time.time()
    bounds_with_buffer = (
        bounds[0] - buff,
        bounds[1] - buff, 
        bounds[2] + buff,
        bounds[3] + buff
    )
    t2=time.time()
    print(f"Computing bounds: {t2-t1:.3f}s")
    
    #print(f"About to select with bounds: {bounds_with_buffer}")
    #print(f"Dataset coords before selection: {ds.coords}")
    
    t3 = time.time()
    subset=ds.sel(
    #return ds.sel(
        latitude=slice(bounds_with_buffer[1], bounds_with_buffer[3]),
        longitude=slice(bounds_with_buffer[0], bounds_with_buffer[2])
    )
    t4 = time.time()
    
    print(f"Selection operation: {t4-t3:.3f}s")
    
    return subset
    #t2 = time.time()
    
    #print(f"Bounds calc: {t1-t0:.3f}s")
    #print(f"Selection op: {t2-t1:.3f}s")
    #print(f"Subset shape: {subset.dims}")
    
    #lat_idx, lon_idx = get_spatial_indices(hyfabfile)
    #return ds.isel(latitude=lat_idx, longitude=lon_idx)


def python_ExactExtract_zarr(aorc_file, hyfabfile, add_offset, scale_factor, AORC_met_vars, NN_table, gapfill, timestamp=None, total_iterations=1):
    # time logging for performance metrics
    #t0 = time.time()
    
    if not hasattr(python_ExactExtract_zarr, "last_time"):
        python_ExactExtract_zarr.last_time = time.time()
        python_ExactExtract_zarr.elapsed_times = np.zeros(100)
        python_ExactExtract_zarr.index = 0
        python_ExactExtract_zarr.count = 0
        python_ExactExtract_zarr.current_iteration = 0
    
    current_time = time.time()
    elapsed = current_time - python_ExactExtract_zarr.last_time
    
    avg_time = np.mean(python_ExactExtract_zarr.elapsed_times)

    #use cached subset
    ds_subset = get_cached_subset(timestamp.year, hyfabfile, AORC_met_vars)
    all_vars_data = ds_subset.sel(time=timestamp.strftime('%Y-%m-%d %H:%M:%S'))
    
    lats = all_vars_data.latitude.values
    lons = all_vars_data.longitude.values   
        
    # Set geotransform once
    xmin, xmax = lons.min(), lons.max()
    ymin, ymax = lats.min(), lats.max()
    dx = (xmax - xmin) / (len(lons) - 1)
    dy = (ymax - ymin) / (len(lats) - 1)
    geotransform = [xmin, dx, 0, ymax, 0, -dy]
    
    # Create process-specific temporary directory
    process_id = current_process().pid
    temp_dir = f"/tmp/aorc_process_{process_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Initialize dictionaries for raster wrappers and operations
        rsw_dict = {}
        op_dict = {}
        
        gdal_start = time.time()
        # Loop over each meteorological variable
        
       
        for i, var in enumerate(AORC_met_vars):
            # Create temporary GeoTIFF for this variable
            temp_file = os.path.join(temp_dir, f"var_{i}.tif")
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = all_vars_data[var].shape
            dataset = driver.Create(temp_file, cols, rows, 1, gdal.GDT_Float32)
            
            # Set projection and geotransform
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            dataset.SetProjection(srs.ExportToWkt())
            dataset.SetGeoTransform(geotransform)
            
            # Write variable data
            dataset.GetRasterBand(1).WriteArray(all_vars_data[var].values)
            dataset.FlushCache()
            dataset = None
            
            # Create raster wrapper and operation
            rsw_dict[f"rsw{i}"] = GDALRasterWrapper(temp_file)
            op_dict[f"op{i}"] = Operation.from_descriptor(f'mean({var})', raster=rsw_dict[f"rsw{i}"])
            
       
        gdal_setup_end = time.time()
        
        # Process features
        #process_start = time.time()
        dsw = GDALDatasetWrapper.from_descriptor(f"{hyfabfile}[hyfabfile_final]", field_name='divide_id')
        writer = MapWriter()
        processor = FeatureSequentialProcessor(dsw, writer, list(op_dict.values()))
        processor.process()  
        #process_end = time.time()  

        # Process results
        #results_start = time.time()
        csv_results = pd.DataFrame(writer.output.values(), columns=AORC_met_vars)
        for i, column in enumerate(csv_results):
            csv_results[column] = csv_results[column] * scale_factor[i] + add_offset[i]
            #print(f"Scale factor: {scale_factor}, i: {i}")
        
          
        csv_results['cat-id'] = writer.output.keys()
        csv_results['time'] = (timestamp - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()
        
        if(gapfill):
            csv_results = nearest_neighbor_correction(csv_results, NN_table, AORC_met_vars)
        
        #results_end = time.time()
        
        #total_time = time.time() - t0
        
        #print(f"Dataset retrieval: {t1-t0:.3f}s")
        #print(f"Spatial subsetting: {t2-t1:.3f}s")
        #print(f"Time selection and compute: {t3-t2:.3f}s")
        #print(f"GDAL setup: {gdal_setup_end-gdal_start:.3f}s")
        #print(f"Feature processing: {process_end-process_start:.3f}s")
        #print(f"Results processing: {results_end-results_start:.3f}s")
        #print(f"Total function time: {total_time:.3f}s")
        
        # processing time metrics, continually updating
        python_ExactExtract_zarr.elapsed_times[python_ExactExtract_zarr.index] = elapsed
        python_ExactExtract_zarr.index = (python_ExactExtract_zarr.index + 1) % 100
        python_ExactExtract_zarr.count = min(python_ExactExtract_zarr.count + 1, 100)
       
        avg_time = np.mean(python_ExactExtract_zarr.elapsed_times[:python_ExactExtract_zarr.count])
        python_ExactExtract_zarr.current_iteration += 1
        percent_done = (python_ExactExtract_zarr.current_iteration / total_iterations) * 100

        output = f"\r Processed {timestamp}, current: {elapsed:.4f}s, Avg: {avg_time:.4f}s, {percent_done:3.1f}%"
        print(f"\r{output:<60}", end='', flush=True)
        python_ExactExtract_zarr.last_time = current_time
        
        return csv_results
        
    finally:
        # Clean up resources
        for rsw in rsw_dict.values():
            del rsw
        #ds.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

def python_ExactExtract(aorc_file, hyfabfile, add_offset, scale_factor, AORC_met_vars, NN_table, gapfill):
    
    # load AORC netcdf file into gdal dataframe to
    # partition out meterological variables into rasters
    aorc = gdal.Open(aorc_file)
    # Get gdal sub-datasets, which will seperate each AORC
    # variable into their own raster wrapper
    nc_rasters = aorc.GetSubDatasets()
    # get datetime of forcing file to append
    # to ExactExtract csv output file
    date_time = get_date_time(aorc_file)
    #print(f"date_time: {date_time}")
    # Define gdal writer to only return ExactExtract
    # regrid results as a python dict
    writer = MapWriter()
    # Define operation and raster dictionaries for each AORC met variable
    op_dict = {}
    rsw_dict = {}
    # loop over each meteorological variable and call
    # ExactExtract to regrid raster to lumped sum for
    # a given NextGen catchment
    for i in np.arange(len(AORC_met_vars)):
        # Get variable name in netcdf file
        variable = nc_rasters[i][0].split(":")[-1]
        # Get the gdal netcdf syntax for netcdf variable
        # Example syntax: 'NETCDF:"AORC-OWP_2012050100z.nc4":APCP_surface'
        nc_dataset_name = nc_rasters[i][0]
        # Define raster wrapper for AORC meteorological variable
        # and specify nc_file attribute to be True. Otherwise,
        # this function will expect a .tif file. Assign data for dict variable
        rsw_dict["rsw{0}".format(i)] = GDALRasterWrapper(nc_dataset_name)
        # Define operation to use for raster and assign data for dict variable
        op_dict["op{0}".format(i)] = Operation.from_descriptor('mean('+variable+')', raster=rsw_dict["rsw{0}".format(i)])
    # For each AORC met variable, we must redefine the
    # hydrofabric raster dataset to regrid forcings
    # based on user operation below
    dsw = GDALDatasetWrapper(hyfabfile)
    # Process the data and write results to writer instance
    processor = FeatureSequentialProcessor(dsw, writer, list(op_dict.values()))
    processor.process()
    # convert dict results to pandas dataframe
    csv_results = pd.DataFrame(writer.output.values(),columns = AORC_met_vars)
    # Loop through AORC met variables and acount for their scale factor
    # and offset (if any) within the netcdf metadata
    met_loop = 0
    for column in csv_results:
        csv_results[column] = csv_results[column]*scale_factor[met_loop] + add_offset[met_loop]
        met_loop += 1
    # Assign catchment ids of regridded data from writer output keys
    csv_results['cat-id'] = writer.output.keys()
    # get seconds since AORC reference date for time array in
    # pandas dataframe
    time = np.zeros(len(csv_results))
    time[:] = (pd.Timestamp(datetime.datetime.strptime(date_time,'%Y%m%d%H')) - pd.Timestamp("1970-01-01 00:00:00")).total_seconds()
    # Assign the timestamp for the regridded data pandas dataframe
    csv_results['time'] = time
    # Flush changes to disk
    writer = None

    # reset the column indices for consistency with formatting
    csv_results = csv_results[['cat-id','time','APCP_surface','DLWRF_surface','DSWRF_surface','PRES_surface','SPFH_2maboveground','TMP_2maboveground','UGRD_10maboveground','VGRD_10maboveground']]

    # call the nearest neighbor function to gapfill
    # coastal catchments and islands with the nearest
    # NextGen catchment with available AORC data if
    # the given hydrofabric file indicated catchments with
    # missing AORC data
    if(gapfill):
        csv_results = nearest_neighbor_correction(csv_results, NN_table, AORC_met_vars)

    return csv_results

def process_sublist(data : dict, lock: Lock, shared_results, num: int, met_dataset_pathway, output_root, hyfabfile, weights, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, aorc_ncfile, NN_table, gapfill, zarr_data):

    try:
        if data.get('is_zarr', False):
            EE_df_final = pd.DataFrame()
            time_subset = data["time_periods"]
            weights_data = data.get("weights_data") if data.get("weights_data") is not None else pd.read_csv(weights)
            total_iterations=len(time_subset)
            
            print(f"Thread {num} processing zarr data at {time.strftime('%H:%M:%S')}.")
            print(f"Thread {num} will process {total_iterations} time periods.")
               
            _s3 = s3fs.S3FileSystem(anon=True)
            
            for year in zarr_data['years']:
                year_start = time.time()
                store_path = f"s3://noaa-nws-aorc-v1-1-1km/{year}.zarr"
                store = s3fs.S3Map(root=store_path, s3=_s3, check=False)
                       
                try:
                    ds = xr.open_dataset(store, engine='zarr', chunks={'time': 'auto'})
                         
                    year_periods = time_subset[time_subset.year == year]
                    CHUNK_SIZE = 24
                       
                    for chunk_start in range(0, len(year_periods), CHUNK_SIZE):
                        chunk_end = min(chunk_start + CHUNK_SIZE, len(year_periods))
                        chunk_periods = year_periods[chunk_start:chunk_end]
                           
                        chunk_start_time = pd.Timestamp(chunk_periods[0].to_timestamp())
                        chunk_end_time = pd.Timestamp(chunk_periods[-1].to_timestamp())
                          
                           
                        for timestamp in chunk_periods:
                            EE_df = python_ExactExtract_zarr(
                                met_dataset_pathway, #aorc_file
                                hyfabfile, 
                                add_offset,
                                scale_factor,
                                zarr_data['variables'], #AORC_met_vars
                                NN_table,
                                gapfill,
                                timestamp.to_timestamp(),
                                total_iterations
                            )
                            EE_df_final = pd.concat([EE_df_final, EE_df])
                               
                finally:
                    if 'ds' in locals():
                        ds.close()
                        del ds
                    if 'store' in locals():
                        del store
                    gc.collect()
                       
            shared_results.append(EE_df_final)
        else:
            # existing non-zarr processing:
            # Get number of files in each thread to loop through
            num_files = len(data["url_forcing_files"])    

            # Initalize pandas dataframe to save the
            # regridded AORC ExactExtract results from
            # each AORC file we loop through
            EE_df_final = pd.DataFrame()

            # Read in ExactExtract coverage fraction weights file
            weights = pd.read_csv(weights)
            #print(f"weights = {weights}")
            for i in range(num_files):
                # extract forcing url file and file name
                aorc_url_file = data["url_forcing_files"][i]
                aorc_filename = data["aorc_filename"][i]

                # Flag to indicate user requested to 
                # download AORC data off ERRDAP server
                if(met_dataset_pathway == None):
                    # Initalize AORC netcdf filename pathway
                    datafile_path = join(output_root, aorc_filename)

                    # Generate dummy certificate for quicker access to ERRDAP server
                    ssl._create_default_https_context = ssl._create_unverified_context

                    if(datafile_path != aorc_ncfile):
                        # Call wget to download AORC forcing file for ExactExtract regridding
                        filename = wget.download(aorc_url_file,out=output_root)

                else:
                    # Initalize AORC netcdf filename pathway
                    datafile_path = join(met_dataset_pathway, aorc_filename)

                # Call python ExactExtract routine to directly extract 
                # AORC regridded results to global AORC variables
                EE_df = python_ExactExtract_Manual_Weights(datafile_path, weights.copy(), add_offset, scale_factor, AORC_met_vars, AORC_missing_value, NN_table, gapfill)
                # concatenate the regridded data to threads final dataframe
                EE_df_final = pd.concat([EE_df_final,EE_df])

            # collect the garbage within the thread before sending results back to main
            # thread to maximize our RAM as much as possible
            gc.collect()

            # Put regridded results into thread queue to return to main thread
            #print(f"Thread {num} completed successfully: {len(EE_df_final)} records, {EE_df_final['time'].nunique()} unique timestamps")
            shared_results.put(EE_df_final)
            
    except Exception as e:
        print(f"Thread {num} FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        shared_results.put(pd.DataFrame()) 


def CONUS_NGen_files(pr, met_dataset_pathway, datafiles, aorc_filenames, output_root, netcdf_dir, csv_dir, netcdf, csv, hyfabfile, weights, num_processes, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, num_files, num_catchments, aorc_ncfile, nextgen_cat_ids, NN_table, gapfill):

    #Get the years of data that we will need to loop through
    years = pr.year.unique()

    # loop through each year of data, and create unique netcdf/csv files
    for year in years:
        print("Loop through CONUS AORC forcing data for year " + str(year))

        if met_dataset_pathway is not None and met_dataset_pathway.startswith('s3://'):
            # For zarr, we just need the time periods for this year
            pr_yearly = pr[pr.year == year]
            annual_months = pr_yearly.month.unique()
        else:
            # Slice list based on url and filenames within a particular year
            yearly_filenames = [s for s in aorc_filenames if "AORC-OWP_" + str(year) in s]
            yearly_datafiles = [s for s in datafiles if "AORC-OWP_" + str(year) in s]

            # partition out the months within the year of data
            # we are looping through
            pr_yearly = pr[pr.year == year]
            annual_months = pr_yearly.month.unique()

        # seperate batches of months in either monthly batches from 1-3 months
        # which is the amount we can handle on the NWC servers currently
        if(len(annual_months) <= 3):
            monthly_batches = np.array_split(annual_months,1)
        elif(len(annual_months) > 3 and len(annual_months) <=6):
            monthly_batches = np.array_split(annual_months,2)
        elif(len(annual_months) > 6 and len(annual_months) <=9):
            monthly_batches = np.array_split(annual_months,3)
        else:
            monthly_batches = np.array_split(annual_months,4)

        # Loops over each monthly batch and create CONUS netcdf file for data
            # Create monthly AORC file extension strings to grab the
            # correct monthly AORC files for this monthly batch
        for months in monthly_batches:
            print("Loop through CONUS AORC forcing data for year " + str(year) + " and months" + str(months.values))
            if is_zarr:
                # Get time periods for these months
                monthly_periods = pr_yearly[pr_yearly.month.isin(months)]
                num_forcing_files_monthly = len(monthly_periods)
                
                # Split time periods among processes
                time_groups = np.array_split(monthly_periods, num_processes)
                
                process_data = []
                process_list = []
                lock = Lock()

                for i in range(num_processes):
                    data = {
                        "time_periods": time_groups[i],
                        "year": year,
                        "months": months
                    }
                    process_data.append(data)
                    
                    p = Process(
                        target=process_sublist, 
                        args=(data, lock, EE_results, i, met_dataset_pathway, output_root, 
                              hyfabfile, weights, add_offset, scale_factor, AORC_met_vars, 
                              AORC_missing_value, aorc_ncfile, NN_table, gapfill, zarr_data)
                    )
                    process_list.append(p)
            else:
                monthly_file_str = []
                for value in months:
                    if(value < 10):
                        monthly_file_str.append("AORC-OWP_" + str(year) + '0' + str(value))
                    else:
                        monthly_file_str.append("AORC-OWP_" + str(year) + str(value))

                # Slice list based on url and filenames within a particular year
                monthly_filenames = [s for s in yearly_filenames if any(xs in s for xs in monthly_file_str)]
                monthly_datafiles = [s for s in yearly_datafiles if any(xs in s for xs in monthly_file_str)]


                num_forcing_files_monthly = len(monthly_datafiles)

                #generate the data objects for child processes
                url_file_groups = np.array_split(np.array(monthly_datafiles), num_processes)
                aorc_file_name_groups = np.array_split(np.array(monthly_filenames), num_processes)

                # If there's no dataset pathway specified
                # then we assume user wants to download
                # AORC data off the server
                if(met_dataset_pathway == None):
        
                    # Grab first datafile for given year to save for netcdf creation
                    aorc_ncfile_monthly = join(output_root,monthly_filenames[0])

                    # Check to see if file exsits since we've already previously downloaded
                    # the first datafile to extract netcdf metadata
                    if(os.path.exists(aorc_ncfile_monthly) == False):
                        filename = wget.download(monthly_datafiles[0],out=output_root)
                else:
                    # Grab first datafile for given year to save for netcdf creation
                    aorc_ncfile_monthly = join(met_dataset_pathway,monthly_filenames[0])


                process_data = []
                process_list = []
                lock = Lock()

                # Initalize thread storage to return to main program
                EE_results = Queue()

                for i in range(num_processes):
                    # fill the dictionary with needed at
                    data = {}
                    data["url_forcing_files"] = url_file_groups[i]
                    data["aorc_filename"] = aorc_file_name_groups[i]

                    #append to the list
                    process_data.append(data)

                    p = Process(target=process_sublist, args=(data, lock, EE_results, i, met_dataset_pathway, output_root, hyfabfile, weights, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, aorc_ncfile_monthly, NN_table, gapfill))

                    process_list.append(p)

            #start all processes
            for p in process_list:
                p.start()

            print("Gathering regridded data to main thread")
            # Before we terminate threads, aggregate thread
            # regridded results together and save to main thread
            final_df = pd.DataFrame()
            for i in range(num_processes):
                result = EE_results.get()
                final_df = pd.concat([final_df,result])
                print("Collecting regridded data from thread " + str(i))

            #wait for termination
            for p in process_list:
                p.join()

            # delete variables to free up RAM
            del(EE_results)
            del(url_file_groups)
            del(aorc_file_name_groups)
            del(result)

            # Collect garbage from main program to save RAM
            gc.collect()

            print("Sorting data by catchment ids and time")
            # Sort aggregated data based on cat-id and timestamp
            final_df = final_df.sort_values(by=['cat-id','time'])

            time1=datetime.datetime.now()
            if (netcdf):
                #generate single NextGen netcdf file from aggregated regridded data
                print('Now Generating NextGen monthly (' + str(year) + ')' + str(months.values) + 'netcdf file for CONUS')
                create_ngen_netcdf(aorc_ncfile_monthly, final_df, netcdf_dir, num_forcing_files_monthly, num_catchments)

            time2=datetime.datetime.now()
            if (csv):
                # generate catchment csv files from aggregated regridded data
                print('Now Generating NextGen yearly (' + str(year) + ')' + str(months.values) + ' csv files for CONUS')
                create_ngen_cat_csv_annual(final_df, csv_dir, num_processes)
                #final_df.index =  final_df['cat-id'].values
            del(final_df)
            gc.collect()
            # Finally, once we've finished producing netcdf/csv files, remove the inital
            # AORC file used for netcdf metadata and AORC variable names
            time3=datetime.datetime.now()

            # If user is downloading AORC data
            # off of ERRDAP, then remove file
            if(met_dataset_pathway == None):
                os.remove(aorc_ncfile_monthly)


    print("Finished with creating monthly netcdf/csv files, now time to combine them")

    time4=datetime.datetime.now()
    if (csv):
        # generate NextGen catchment csv files from aggregated regridded data
        print('Now Combining monthly NextGen csv files for CONUS')
        create_ngen_cat_csv_CONUS(csv_dir, nextgen_cat_ids, num_processes) 

    time5=datetime.datetime.now()
    if (netcdf and len(os.listdir(netcdf_dir)) > 1):
        # generate NextGen netcdf file from aggregated regridded data
        print('Now Combining monthly NextGen netcdf file for CONUS')
        create_ngen_netcdf_CONUS(aorc_ncfile, netcdf_dir)
    time6=datetime.datetime.now()


def VPU_NGen_files(met_dataset_pathway, datafiles, aorc_filenames, output_root, netcdf_dir, csv_dir, netcdf, csv, hyfabfile, weights, num_processes, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, num_files, num_catchments, aorc_ncfile, NN_table, gapfill, pr, zarr_data):
    
    weights_data = pd.read_csv(weights) if weights else None
    
    if met_dataset_pathway is not None and met_dataset_pathway.startswith('s3://'):
        # Modified zarr processing with year chunks
        years = sorted(pr.year.unique())
        chunk_size = min(num_processes, len(years))
        
        manager = Manager()
        shared_results = manager.list()
        
        # Process years in chunks equal to number of processes
        for i in range(0, len(years), chunk_size):
            years_chunk = years[i:i + chunk_size]
            process_list = []
            
            # Create process for each year in chunk
            for j, year in enumerate(years_chunk):
                year_periods = pr[pr.year == year]
                data = {
                    "time_periods": year_periods,
                    "weights_data" : weights_data,
                    "is_zarr": True
                }
                
                p = Process(
                    target=process_years_chunk,
                    args=([year], data, None, shared_results, j, 
                          met_dataset_pathway, output_root, hyfabfile,
                          weights, add_offset, scale_factor, AORC_met_vars,
                          AORC_missing_value, aorc_ncfile, NN_table, 
                          gapfill, zarr_data)
                )
                process_list.append(p)
            
            # Start and wait for processes in current chunk
            for p in process_list:
                p.start()
            for p in process_list:
                p.join()
            
            # Clear cache after chunk is complete
            gc.collect()
    else:
        url_file_groups = np.array_split(np.array(datafiles), num_processes)
        aorc_file_name_groups = np.array_split(np.array(aorc_filenames), num_processes)
        
        result_queue = Queue()
        process_list = []

        #print(f"About to create {num_processes} processes for {len(datafiles)} files")
        #print(f"Files per process: {[len(group) for group in url_file_groups]}")

        for i in range(num_processes):
            data = {
                "url_forcing_files": url_file_groups[i],
                "aorc_filename": aorc_file_name_groups[i],
                "is_zarr": False
            }

            #print(f"Process {i}: {len(data['url_forcing_files'])} files assigned")
            #if len(data['aorc_filename']) > 0:
            #    print(f"  First: {data['aorc_filename'][0]}")
            #    print(f"  Last: {data['aorc_filename'][-1]}")

            p = Process(target=process_sublist, 
                       args=(data, None, result_queue, i, met_dataset_pathway, 
                            output_root, hyfabfile, weights, add_offset, 
                            scale_factor, AORC_met_vars, AORC_missing_value, 
                            aorc_ncfile, NN_table, gapfill, zarr_data))
            process_list.append(p)

        print(f"Created {len(process_list)} processes")

        # Start all processes
        for p in process_list:
            p.start()

        print("All processes started")
        
        # DEBUG: Check process status before collecting results
        #import time
        #time.sleep(2)  # Give processes a moment to start
        #alive_count = sum(1 for p in process_list if p.is_alive())
        #print(f"Processes alive after start: {alive_count}/{num_processes}")

        # Collect results with better error handling and monitoring
        results = []
        collected_count = 0
        max_wait_time = 60  # seconds to wait for each result
        
        print("Starting result collection (may take some time)...")
        while collected_count < num_processes:
            try:
                # Check if queue has items before trying to get
                if not result_queue.empty():
                    result = result_queue.get_nowait()
                    results.append(result)
                    collected_count += 1
                else:
                    # Check if any processes are still alive
                    alive_processes = [p for p in process_list if p.is_alive()]
                    
                    if len(alive_processes) == 0:
                        # Try to get any remaining results
                        while not result_queue.empty():
                            result = result_queue.get_nowait()
                            results.append(result)
                            collected_count += 1
                        break
                    # Wait a bit and try again
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error collecting result {collected_count + 1}: {e}")
                # Add empty DataFrame for failed process
                results.append(pd.DataFrame())
                collected_count += 1

        print(f"Result collection complete. Got {len(results)} results")

        # Join processes with timeout to prevent hanging
        print("Joining processes...")
        for i, p in enumerate(process_list):
            try:
                #print(f"Joining process {i}...")
                p.join(timeout=30)  # 30 second timeout
                if p.is_alive():
                    print(f"Process {i} didn't finish, terminating...")
                    p.terminate()
                    time.sleep(1)
                    if p.is_alive():
                        print(f"Process {i} still alive after terminate, killing...")
                        p.kill()
                #print(f"Process {i} joined successfully")
            except Exception as e:
                print(f"Error joining process {i}: {e}")

    if met_dataset_pathway is not None and met_dataset_pathway.startswith('s3://'):
        final_df = pd.concat(list(shared_results))
        
    else:
        print(f"Collected {len(results)} results from queue")
        
        # Filter out empty results and concatenate
        valid_results = [r for r in results if len(r) > 0]
        print(f"Valid results: {len(valid_results)}")
        
        if len(valid_results) == 0:
            print("WARNING: No valid results collected!")
            return pd.DataFrame()
        
        final_df = pd.concat(valid_results)
    
    final_df = final_df.sort_values(by=['cat-id','time'])

    if netcdf:
        print('Generating NextGen netcdf file for single VPU')
        create_ngen_netcdf(aorc_ncfile, final_df, netcdf_dir, num_files, num_catchments)

    if csv:
        print('Generating NextGen csv files for single VPU')

        # Debug the final_df before processing
        print(f"Final dataframe shape: {final_df.shape}")
        print(f"Unique catchments: {final_df['cat-id'].nunique()}")
        print(f"Unique timestamps: {final_df['time'].nunique()}")

        create_ngen_cat_csv(final_df, csv_dir, num_catchments, num_files, num_processes)

    if met_dataset_pathway is None:
        os.remove(aorc_ncfile)
        print(f"met_dataset_pathway = None")

    return final_df


def NextGen_Forcings_AORC(output_root, met_dataset_pathway, AORC_start_time, AORC_end_time, netcdf, csv, CONUS, hyfabfile, weights_file, num_processes : int):
   
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
        
    #Checking which data source type we're dealing with
    is_zarr = met_dataset_pathway is not None and met_dataset_pathway.startswith('s3://')
    is_local = met_dataset_pathway is not None and not is_zarr
    is_erddap = met_dataset_pathway is None
    
    # Remove AORC files if they're already existing within the 
    # output directory from a previous run and they're using
    # the option to extract data off the ERRDAP server
    if(met_dataset_pathway == None and not is_zarr):
        os.system('rm ' + join(output_root,'AORC-*.nc*'))

    # get number of catchments and their ids from hydrofabric
    cat_df_full = gpd.read_file(hyfabfile,layer='divides')
    nextgen_cat_ids = [i for i in cat_df_full.divide_id]
    num_catchments = len(nextgen_cat_ids)
    print("number of catchments = {}".format(num_catchments))
    # Now delete instance of hydrofabric file
    # and respective variables to save RAM
    del(cat_df_full)

    # Create time series based on user input
    pr = pd.period_range(start=AORC_start_time,end=AORC_end_time,freq='h')
    pr = pr[0:len(pr)-1]

    if is_zarr:
        datafiles = None
        aorc_filenames = None
        aorc_ncfile = None
        num_forcing_files = len(pr)
    
        _s3 = s3fs.S3FileSystem(anon=True)
    
        # Get unique years needed
        years = pr.year.unique()
    

    
        # Create zarr store for first year to get metadata
        first_year = min(years)
        aorc_ncfile = f"s3://noaa-nws-aorc-v1-1-1km/{first_year}.zarr"
        first_store = s3fs.S3Map(
            root=aorc_ncfile,
            #root=f"s3://noaa-nws-aorc-v1-1-1km/{first_year}.zarr",
            s3=_s3,
            check=False
        )
    
        # Open first dataset to get metadata (similar to how other paths use first file)
        ds = xr.open_dataset(
            first_store,
            engine='zarr',
        chunks={'time': 'auto'}
        )
    
        # Get AORC variable names from zarr data (similar to NetCDF extraction)
        AORC_met_vars = [var for var in ds.variables if '_' in var]
    
        # Get scale factors and offsets (similar to NetCDF handling)
        add_offset = np.zeros([len(AORC_met_vars)])
        scale_factor = np.ones([len(AORC_met_vars)])
        
        metadata = json.loads(first_store['.zmetadata'].decode('utf-8'))
        
        for i, var in enumerate(AORC_met_vars):
            var_attrs = metadata['metadata'][f'{var}/.zattrs']
            add_offset[i] = 0.0
            scale_factor[i] = 1.0
            #print(f"{var} scale_factor: {scale_factor[i]}, add_offset: {add_offset[i]}")
        AORC_missing_value = ds[AORC_met_vars[0]].attrs.get('missing_value', -9999.0)
            
        # Store zarr info for later processing
        zarr_data = {
            'years': years,
            'variables': AORC_met_vars,
            'add_offset': add_offset,
            'scale_factor': scale_factor,
            'missing_value': AORC_missing_value
        }
        ds.close()
    elif is_local:
        # Local file handling
        aorc_filenames = []
        datafiles = []

        aorc_end = ".nc4"
        aorc_new_end = ".nc4"
        #TODO: Add if_alaska check
        aorc_beg = "AK_AORC-OWP_"

        for dt in pr:
            year = dt.strftime('%Y')
            month = dt.strftime('%m')
            day = dt.strftime('%d')
            hour = dt.strftime('%H')
        
            # Use same filename convention as ERRDAP, based on original code
            if(int(year) > 2019):
                aorc_end_final = aorc_new_end
            else:
                aorc_end_final = aorc_end
            
            file_name = aorc_beg + year + month + day + hour + aorc_end_final
            file_path = join(met_dataset_pathway, file_name)
        
            # Only add if file exists
            if os.path.exists(file_path):
                aorc_filenames.append(file_name)
                datafiles.append(file_path)
            else:
                print(f"Warning: Expected file not found: {file_path}")
            
        num_forcing_files = len(datafiles)
        if num_forcing_files == 0:
            raise FileNotFoundError(f"No AORC files found in {met_dataset_pathway}")
        else:
            print(f"Number of forcing files: {num_forcing_files}")
        
        #print(f"aorc_filenames: {aorc_filenames}")

        aorc_ncfile = join(met_dataset_pathway, aorc_filenames[0])
        zarr_data = None
        
        # Extract variable names from AORC netcdf data
        nc_file = nc4.Dataset(aorc_ncfile)
        # Get variable list from AORC file
        nc_vars = list(nc_file.variables.keys())
        # Get indices corresponding to Meteorological data
        indices = [nc_vars.index(i) for i in nc_vars if '_' in i]
        # Make array with variable names to use for ExactExtract module
        AORC_met_vars = np.array(nc_vars)[indices]

        # get scale_factor and offset keys if available
        # (AORC-OWP files for HUC01 scenario has this metadata)
        add_offset = np.zeros([len(AORC_met_vars)])
        scale_factor = np.zeros([len(AORC_met_vars)])
        i = 0
        for key in AORC_met_vars:
            try:
                scale_factor[i] = nc_file.variables[key].scale_factor
            except AttributeError as e:
                scale_factor[i] = 1.0
            try:
                add_offset[i] = nc_file.variables[key].add_offset
            except AttributeError as e:
                add_offset[i] = 0.0
            i += 1

        #Get AORC missing value
        AORC_missing_value = nc_file.variables[AORC_met_vars[0]].missing_value
        
        # Close netcdf file
        nc_file.close()
        
    else: #is_erddap
    
        # Initialize ssl default context to connect to url server when downloading files
        ssl._create_default_https_context = ssl._create_unverified_context
        
        aorc_beg = "AORC-OWP_"
        aorc_end = "z.nc4"
        aorc_new_end = ".nc4"
        base_url = 'https://nwcal-wrds-ti01.nwc.nws.noaa.gov/aorc_erddap/erddap/files/erddap_dadf_1012_a549/'

        datafiles = []
        aorc_filenames = []
        for dt in pr:
            year = dt.strftime('%Y')
            month = dt.strftime('%m')
            day = dt.strftime('%d')
            hour = dt.strftime('%H')
            sub_dir = year + month + '/'
            # flag to change ending of AORC file 
            # based on system file changes after 2019
            if(int(year) > 2019):
                aorc_end_final = aorc_new_end
            else:
                aorc_end_final = aorc_end
            
            file_name = aorc_beg + year + month + day + hour + aorc_end_final
            aorc_file = base_url + sub_dir + file_name
            aorc_filenames.append(file_name)
            datafiles.append(aorc_file)
        
        #Get number of AORC forcing files
        num_forcing_files = len(datafiles)
        print("number of AORC forcing files = {}".format(num_forcing_files))
        
        #Download first file for metadata
        aorc_ncfile = join(met_dataset_pathway, aorc_filenames[0])
        filename = wget.download(datafiles[0], out=output_root)
        zarr_data = None
 
        # Extract variable names from AORC netcdf data
        nc_file = nc4.Dataset(aorc_ncfile)
        # Get variable list from AORC file
        nc_vars = list(nc_file.variables.keys())
        # Get indices corresponding to Meteorological data
        indices = [nc_vars.index(i) for i in nc_vars if '_' in i]
        # Make array with variable names to use for ExactExtract module
        AORC_met_vars = np.array(nc_vars)[indices]

        # get scale_factor and offset keys if available
        # (AORC-OWP files for HUC01 scenario has this metadata)
        add_offset = np.zeros([len(AORC_met_vars)])
        scale_factor = np.zeros([len(AORC_met_vars)])
        i = 0
        for key in AORC_met_vars:
            try:
                scale_factor[i] = nc_file.variables[key].scale_factor
            except AttributeError as e:
                scale_factor[i] = 1.0
            try:
                add_offset[i] = nc_file.variables[key].add_offset
            except AttributeError as e:
                add_offset[i] = 0.0
            i += 1

        #Get AORC missing value
        AORC_missing_value = nc_file.variables[AORC_met_vars[0]].missing_value
        
        # Close netcdf file
        nc_file.close()

    # Read in divides layer of the hydrofabric geopackage file
    # that is used to calculate lumped forcings
    hyfab_data = gpd.read_file(hyfabfile,layer='divides')

    # Need to reproject the hydrofabric crs to the meteorological forcing
    # dataset crs for ExactExtract to properly regrid the data and save the
    # file for ExactExtract bindings to rasterize
    hyfabfile_final = join(output_root,"hyfabfile_final.json")
    hyfab_data = hyfab_data.to_crs('WGS84')
    hyfab_data.to_file(hyfabfile_final,driver="GeoJSON")


    # Flag to see if user has already provided an ExactExtract
    # coverage weights file, otherwise go ahead and produce the file
    if(weights_file != None):
        weights = weights_file
    else:
        # Generate the ExactExtract Coverage Fraction Weights File
        weights = Python_ExactExtract_Coverage_Fraction_Weights(aorc_ncfile, hyfabfile_final, AORC_met_vars, output_root)    

    # Generate nearest neighbor table for a given hydrofabric domain 
    # which will provide a gap-fill method for coastal catchments outside
    # of the AORC domain
    NN_table = generate_nearest_neighbor_correction_table(hyfabfile_final, aorc_ncfile, add_offset, scale_factor, AORC_met_vars)

    # Boolean flag to indicate if we need to gapfill missing AORC catchment data
    if(type(NN_table) == type(None)):    
        gapfill = False
    else:
        gapfill = True

    # Call function based on type of hydrofabric file
    if(CONUS):
        print("Calling CONUS_NGen_files")
        CONUS_NGen_files(pr, met_dataset_pathway, datafiles, aorc_filenames, output_root, netcdf_dir, csv_dir, netcdf, csv, hyfabfile_final, weights, num_processes, add_offset, scale_factor, AORC_met_vars, AORC_missing_value, num_forcing_files, num_catchments, aorc_ncfile, nextgen_cat_ids, NN_table, gapfill)
    else:
        print("Calling VPU_NGen_files")
        VPU_NGen_files(met_dataset_pathway, datafiles, aorc_filenames, output_root, netcdf_dir, csv_dir, netcdf, csv, hyfabfile_final, weights, num_processes, add_offset, scale_factor, AORC_met_vars, AORC_missing_value,num_forcing_files, num_catchments,aorc_ncfile, NN_table, gapfill, pr, zarr_data)
    # Now clean up I/O files from the script to free up memory for the user
    # Remove the temporary hydrofabric file
    os.remove(hyfabfile_final)


