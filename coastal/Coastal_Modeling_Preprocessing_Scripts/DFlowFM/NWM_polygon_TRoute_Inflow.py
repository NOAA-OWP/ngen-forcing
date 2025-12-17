# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:23:19 2021

@author: jon.allen
"""
from shapely import geometry
from shapely.geometry import Point
import glob
from common.io import read_polygon, read_csv
import datetime
import pathlib
import numpy as np
import netCDF4 as nc
import cftime
import argparse
import operator
import argparse
import pathlib
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from os.path import join
from scipy import spatial

def extract_pois(feature_id):
    feature_id_index = np.empty(len(feature_id),dtype=float)
    number = False
    for i in range(len(feature_id_index)):
        if(feature_id[i] != 'nan'):
            if(number == False):
                poi = list(feature_id[i])
                for j in range(len(poi)):
                    if(number != True):
                        try:
                            int(poi[j])
                            domain_slice = ''.join(poi[0:j])
                            number = True
                        except:
                            pass
                feature_id_index[i] = float(feature_id[i].split(domain_slice)[-1])
            else:
                feature_id_index[i] = float(feature_id[i].split(domain_slice)[-1])
        else:
            feature_id_index[i] = np.nan
    return feature_id_index


class BCFileWriter:
    functions = {'timeseries'}

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self._filehandle = open(self.filename, 'w')
        self._filehandle.write("[General]\n")
        self._filehandle.write("    fileVersion           = 1.01\n")
        self._filehandle.write("    fileType              = boundConds\n")
        self._filehandle.write("\n")
        return self

    def __exit__(self, type, value, traceback):
        self._filehandle.close()

    def add_forcing(self, name, function, units, data):
        """Add forcing
        Args:
            name (str): Name
            function (str): One of BCFileWriter.functions
            units (list[tuples]): A list of tuples mapping column name to column units.
                The ordering should match the ordering of the data columns.
            data (Iterable of lists): Number of columns in data and len(units) must match.
                Data will be iterated thru row by row
        Returns:
            None
        """
        if function not in self.functions:
            raise ValueError("Invalid function")

        fh = self._filehandle
        fh.write("[forcing]\n")
        fh.write(f"    name                  = {name}\n")
        fh.write(f"    function              = {function}\n")
        fh.write(f"    time-interpolation    = linear\n")
        for i, (col, unit) in enumerate(units):
            fh.write(f"    quantity              = {col}\n")
            fh.write(f"    unit                  = {unit}\n")
        if isinstance(data, np.ndarray):
            np.savetxt(fh, data, fmt='%f', delimiter=' ')
        else:
            for row in data:
                fh.write(" ".join(map(str, row)))
                fh.write("\n")

        fh.write("\n")

def binary_isin(elements, test_elements, assume_sorted=False, return_indices=False):
    """Test if values of elements are present in test_elements.
    Returns a boolean array: True if element is present in test_elements, False otherwise.
    If return_indices=True, return an array of indexes that transforms the elements of
    test_elements to same order of elements (for the elements that are present in test_elements).
    ie, for every True in the returned mask, also return the index such that test_elements[indices] == elements[mask]
    """
    elements = np.asarray(elements)
    test_elements = np.asarray(test_elements)
    if assume_sorted:
        idxs = np.searchsorted(test_elements, elements)
    else:
        asorted = test_elements.argsort()
        idxs = np.searchsorted(test_elements, elements, sorter=asorted)
    valid = idxs != len(test_elements)
    test_selector = idxs[valid]
    if not assume_sorted:
        test_selector = asorted[test_selector]
    mask = np.zeros(elements.shape, dtype=bool)
    mask[valid] = test_elements[test_selector] == elements[valid]
    #indices are the array of indexes that transform the elements in
    # test_elements to match the order of elements
    if return_indices:
        return mask, test_selector[mask[valid]]
    else:
        return mask

# this function extracts the offsest values for the requested ids
def extract_offsets(current_netcdf_filename, selected_ids):
    """Return a mask on feature_id for the selected_ids
    Args:
        current_netcdf_filename (str): NetCDF file to read
        selected_ids (list_like): ids to select.

    Returns:
        (ndarray): Indexes *in order* of selected comm_ids.
    """
    with nc.Dataset(current_netcdf_filename) as ncdata:
        feature_id_index = ncdata['POI'][:]
        #feature_id_index = [str(i).split('-')[-1] for i in feature_id_index]
        feature_id_index = np.where(feature_id_index=='nan',np.nan,feature_id_index)
        feature_id_index = np.array(feature_id_index,dtype=float)

        #feature_id_index = ncdata['feature_id'][:]

        #print("Size of feature id data is ", len(feature_id_index))
        fmask, fidx = binary_isin(feature_id_index, selected_ids, return_indices=True)
        fvals = feature_id_index[fmask]
        return fmask, fvals, fidx


def extract_inflow_NWM(current_netcdf_filename,NWM_masks):
    with nc.Dataset(current_netcdf_filename) as ncdataa:

        # extract NWM Common ID lateral discharge and return values
        stream_flow_vals = ncdataa['flow'][:].data[NWM_masks,:]
        timestamps = nc.num2date(ncdataa['time'][:],units='seconds since '+ncdataa.file_reference_time,only_use_cftime_datetimes=False)
    
    return stream_flow_vals, timestamps.data

def create_boundary_files(output_file: pathlib.Path, data: dict, ref_time, date_index):
    """Create DFlow boundary files in bc text file format

    Args:
        output_dir (pathlib.Path): Output directory
        data (dict):

    Raises:
        FileNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    units = [("time", "seconds since "+ ref_time.strftime("%Y-%m-%d %H:%M:%S")),
             ("lateral_discharge", "m^3/s")]
    #date_index = cftime.date2num(data['row_index'], units[0][1], calendar='julian')
    #date_index = np.insert(cftime.date2num(data['row_index'], units[0][1], calendar='julian'),0,cftime.date2num(data['row_index'][0], units[0][1], calendar='julian')-60)
    values = data['inflow']
    ids = data['ids']

    with BCFileWriter(output_file) as bcwriter:
        #for i, commid in enumerate(data['col_index']):
        for i, commid in enumerate(data['col_index']):
            v = values[i,:].data
        # Missing value, set to zero
            #if v.mask.any():
            #    #continue
            #    v[:] = 0.0
            #v = np.insert(v,0,0.0)
            tab = np.empty(len(v),dtype='<U4')
            tab[:] = '   '
            ts = zip(tab, date_index, v)
            print("Writing station:", data['col_index'][i], end='\r')
            bcwriter.add_forcing(data['col_index'][i], "timeseries", units, ts)


def get_inputfiles(input_path, start, end):
    """Generate list of files between start and end timestamps.

    Args:
        input_path (pathlib.Path): Directory to process
        start (datetime.datetime): Start date
        end (datetime.datetime): End date

    Raises:
        FileNotFoundError: Raised when a missing file is encountered

    Returns:
        (list): List of files matching date pattern
    """
    files = {}
    fdate = start
    one_hour = datetime.timedelta(hours=1)
    # Filter for variable NWM channel route output
    # file extensions
    chrt = next(input_path.glob("troute_output_*"))
    troute_name = "troute_output_"
    extension = ".nc"
    while fdate <= end:
        cand = chrt.with_name(fdate.strftime(troute_name + '%Y%m%d%H%M'+extension))
        if cand.exists():
            files[fdate] = cand
        else:
            raise FileNotFoundError(cand)
        fdate += one_hour
    return files

# function to grab and return the polygon
# latitude and longitude coordinates within shape file
def get_NWM_poly(NWM_ID,NWM_poly):

    idx = NWM_poly['hl_link'] == NWM_ID
    print(NWM_ID)
    try:
        # Extract coordinates if data is a polygon
        x, y = NWM_poly['geometry'][np.where(idx==True)[0][0]].exterior.coords.xy
    except:
        # Extract coordinates if data is a multipolygon
        x, y = NWM_poly['geometry'][np.where(idx==True)[0][0]].geoms._get_geom_item(0).exterior.xy

    return np.array(x,dtype=float), np.array(y,dtype=float), NWM_ID

# open and read streamline csv and append
# boolean flag for NWM common IDS within
# user specified polygon
def import_streamline_csv(hyfab_csv_file):
    hyfab = pd.read_csv(hyfab_csv_file)
    #hyfab['POIS'] = extract_pois(hyfab.hl_link.values)
    hyfab['POIS'] = np.array([i.split('-')[-1] for i in hyfab.flowpath_id.values],dtype='int')
    #hyfab['POIS'] = hyfab.poi_id.values

    #idx = hyfab.comment != "outside domain"
    #hyfab = hyfab.loc[idx,:]
    idx = hyfab.outside == 0
    hyfab = hyfab.loc[idx,:]
    return hyfab


# filter points by user specified polygon
def _point_in_polygon(test_polygon, longitude, latitude): #DONE
    line = geometry.LineString(test_polygon)
    point = geometry.Point(longitude, latitude)
    polygon = geometry.Polygon(line)
    result = polygon.contains(point)
    return result #True or False

# Now create the D-FlowFM formated lateral discharge
# boundary conditions within a newly created .ext file
# from NWM polygon files
def NWM_polygon_ext(ext_file, HUC_NWM_agg_sliced, NWM_polygon_file, output_filename):
    # Create beginning of D-FlowFM ext file. This format
    # here is needed to properly initalize lateral
    # discharges over polygon areas in sourc code
    print("Creating pure NWM polygon lateral discharge ext file.\n")
    ext_file = open(ext_file, 'a')
    # Read HUC-12 polygon shape file
    NWM_polygon = gpd.read_file(NWM_polygon_file)
    # Convert to WGS-84 to ensure crs is same for NWM
    NWM_polygon = NWM_polygon.to_crs('WGS-84')


    for i, row in HUC_NWM_agg_sliced.iterrows():
        # Grab HUC-12 coordinates
        lons, lats, NWM_ID = get_NWM_poly(row['hl_link'],NWM_polygon)
        #NWM_ID = row['poi_id']
        NWM_ID = row['POIS']
        # Write out coordinate strings of HUC-12 polygon
        # for D-FlowFM ext file format
        num_coords = str(len(lons))
        x_coord = ''
        y_coord = ''
        for i in np.arange(len(lons)):
            # Flag for iteration appending syntax
            # for coordinate strings
            if(i==0):
                x_coord = x_coord + str(lons[i])
                y_coord = y_coord + str(lats[i])
            else:
                x_coord = x_coord + ' ' + str(lons[i])
                y_coord = y_coord + ' ' + str(lats[i])
        # Write out lateral discharge template format
        ext_file.write("[lateral]"+'\n')
        ext_file.write("id          = "+ str(NWM_ID) + '\n')
        ext_file.write("name        = "+ str(NWM_ID) + '\n')
        ext_file.write("type        = discharge"+'\n')
        ext_file.write("LocationType = all"+ '\n')
        ext_file.write("numCoordinates = "+ num_coords + '\n')
        ext_file.write("xCoordinates = "+ x_coord + '\n')
        ext_file.write("yCoordinates = "+ y_coord + '\n')
        ext_file.write("discharge   = " + str(output_filename).split('/')[-1]+'\n')
        ext_file.write('\n')

    ext_file.close()
    return

# user defined options
def get_options():
    parser = argparse.ArgumentParser(description='Create lateral discharge ext DFlow subdomain file and extract aggregated data to .bc file based on user specified Polygon.')
    parser.add_argument('--comm_ids', dest='comm_id_path', required=True,
                    help='The file list the comm ids to be extracted. Two columns map the comm id to an identifier string (used for boundary condition output).')
    parser.add_argument('--NWM_polygon', dest='NWM_polygon', required=True, type=pathlib.Path, default=None,
                    help='GIS produced shape file that speifies the NWM polygon coordinates based on the D-Flow extent of its meshes.')
    parser.add_argument('--polygon', dest='polygon', required=True, type=pathlib.Path,
                    help='The path of the polygon file defining the region of interest.')
    parser.add_argument('--start', dest='start_time', required=True,
                    help='The date and time to begin making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--stop', dest='stop_time', required=True,
                    help='The date and time to stop making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--troute_input', dest='troute_input_dir', required=True, type=pathlib.Path,
                    help='The directory that stores NWM streamflow input files')
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path('.'), help="Path to bc output directory. Default is current directory")
    parser.add_argument("-ext", "--ext", type=pathlib.Path, default=pathlib.Path('.'), help="Path to ext output file. Default is current directory")

    args = parser.parse_args()

    args.start_time = datetime.datetime.strptime(args.start_time, '%Y-%m-%d_%H:%M:%S')
    args.stop_time = datetime.datetime.strptime(args.stop_time, '%Y-%m-%d_%H:%M:%S')

    # Validate that output_dir is an existing directory
    if not args.output_dir.exists():
        raise FileNotFoundError(args.output_dir)
    elif not args.output_dir.is_dir():
        raise NotADirectoryError(args.output_dir)

    return args


def main(args):

    ############### BEGIN LATQ D-FlowFM .ext file creation ##################

    # Read user defined domain polygon file
    domain_polygon = read_polygon(args.polygon)

    # Read csv file with HUC-12 ID corresponding to NWM Common IDs
    NWM_streams = import_streamline_csv(args.comm_ids)
    # create latq ext file using "new" ext file format
    NWM_polygon_ext(args.ext, NWM_streams, args.NWM_polygon, args.output)


    ############### BEGIN LATQ NWM file extraction ##################
    #get the path for the first input file
    files = get_inputfiles(args.troute_input, args.start_time, args.stop_time)
    print("Processing", len(files), "files in", args.troute_input)
    f0 = next(iter(files.values()))

    #troute_data = nc.Dataset(f0)
    #ref_time = datetime.datetime.strptime(troute_data.file_reference_time, '%Y-%m-%d_%H:%M:%S')
    #timestamps = [args.start_time + datetime.timedelta(minutes=n*5) for n in range(1,13)] * (len(files)*12//12)
    #print(len(timestamps))
    #timestamps = pd.to_datetime(timestamps)
    timestamps = pd.date_range(start=args.start_time.strftime('%Y-%m-%d %H:%M:%S'),end=args.stop_time.strftime('%Y-%m-%d %H:%M:%S'),freq='5min')
    time_final = (timestamps - timestamps[0]).total_seconds()
    #troute_data.close()

    # mask = nwm streamflow mask for selected commids
    # fidx = array of indices to reorder selected commids to nwm streamflow order
    # (ie comm_ids[fidx] corresponds to data[mask]
    #mask, feature_ids, fidx = extract_offsets(f0, NWM_streams['POIS'].values)

    #feature_ids = [float(i.split('-')[-1]) for i in NWM_streams.flowpath_id.values]
    feature_ids = np.array(NWM_streams['poi_id'].values,dtype=float)
    #feature_ids = np.array(NWM_streams['POIS'].values,dtype=float)

    ncdata = nc.Dataset(f0)
    #feature_ids_index = ncdata.variables['feature_id'][:]
    feature_ids_index = ncdata.variables['POI'][:]
    feature_ids_index = np.where(feature_ids_index=='nan',-9999.,feature_ids_index)
    feature_ids_index = np.array(feature_ids_index,dtype=float)

    feature_ids_inds = np.empty((len(feature_ids_index),2))
    feature_ids_inds[:,0] = feature_ids_index
    feature_ids_inds[:,1] = feature_ids_index


    troute_source_ids = np.empty((len(feature_ids),2))
    troute_source_ids[:,0] = feature_ids
    troute_source_ids[:,1] = feature_ids

    distance, mask = spatial.KDTree(feature_ids_inds).query(troute_source_ids)


    # add nwmcommid as string for

    #ids_final = np.array(NWM_streams.poi_id.values,dtype=str)
    ids_final = np.array(NWM_streams.POIS.values,dtype=str)

    #extract lat long and streamflow for the ids with stored offsets
    inflows_timeseries = np.ma.masked_array(np.zeros((len(feature_ids), len(files)*12)), fill_value=0.0)

    timestamps_timeseries = []#np.empty((len(files), 12))

    inflows = np.ma.masked_array(np.zeros((len(feature_ids), len(time_final))), fill_value=0.0)

    data = {'ids': ids_final,
            'col_index': ids_final,
            'row_index': list(files.keys())}

    print("Reading NWM polygon lateral discharges...")
    ffiles = len(files)
    troute_index = 0
    for i, f in enumerate(files.values()):
        inflows_timeseries[:,troute_index:troute_index+12], timestamps_file = extract_inflow_NWM(f,mask)
        troute_index += 12
        timestamps_timeseries.append(timestamps_file)
        print("{}/{} ({:.2%})".format(i, ffiles, i/ffiles).ljust(20), end="\r")
    
    timestamps_timeseries = pd.to_datetime([x for xs in timestamps_timeseries for x in xs])
    for i in range(len(time_final)):
        #time_diff = np.abs([date - timestamps[i] for date in timestamps_timeseries])
        index = abs((timestamps_timeseries - timestamps[i]).total_seconds()).argmin(0)
        inflows[:,i] = inflows_timeseries[:,index]*NWM_streams.FlowDir.values
    # Set streamflow data
    data["inflow"] = inflows

    # create latq .bc file using specified format
    create_boundary_files(args.output, data, args.start_time, time_final)

# Run main when this file is run
if __name__ == "__main__":
    args = get_options()
    main(args)
