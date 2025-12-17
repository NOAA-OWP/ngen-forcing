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
        feature_id_index = ncdata['feature_id'][:]
        #print("Size of feature id data is ", len(feature_id_index))
        fmask, fidx = binary_isin(feature_id_index, selected_ids, return_indices=True)
        fvals = feature_id_index[fmask]
        return fmask, fvals, fidx

# extract the lat long and streamflow for each comm ids
def extract_lat_lon(current_netcdf_filename, idxs):
    with nc.Dataset(current_netcdf_filename) as ncdata:
        # extract the streamflow
        lat_vals = ncdata['latitude'][:][idxs]
        lon_vals = ncdata['longitude'][:][idxs]
        return (lat_vals, lon_vals)


# extract and sum all lateral discharges within HUC polygon mask
def extract_latq_HUCS(current_netcdf_filename, HUC_ids, HUC_masks):
    with nc.Dataset(current_netcdf_filename) as ncdataa:
        # extract the streamflow
        stream_flow_vals = []

        # extract NWM Common IDs within each HUC mask and sum values
        for mask in HUC_masks:
            stream_flow_vals.append(np.nansum(ncdataa['q_lateral'][:][mask],axis=0))
            #stream_flow_vals.append(np.nansum(ncdataa['qSfcLatRunoff'][:][mask],axis=0))
        #stream_flow_vals = ncdata['qSfcLatRunoff'][:][idxs]
        #stream_flow_vals = ncdata['qBucket'][:][idxs]

        return np.array(stream_flow_vals)

# extract and sum all lateral discharges within HUC polygon mask
def extract_latq_HUCS_Lake(current_netcdf_filename, HUC_ids, HUC_masks_lake, HUC_masks_no_lake):
    with nc.Dataset(current_netcdf_filename) as ncdataa:
        # extract the streamflow
        stream_flow_vals = []

        # extract NWM Common IDs within each HUC mask and sum values
        for i in range (len(HUC_masks_lake)):
            #feature_ids = ncdataa['feature_id'][:][mask]
            latq = np.nansum(ncdataa['q_lateral'][:][HUC_masks_no_lake[i]],axis=0) + np.nansum(ncdataa['qSfcLatRunoff'][:][HUC_masks_lake[i]],axis=0)
            #stream_flow_vals.append(np.nansum(ncdataa['q_lateral'][:][mask],axis=0))
            stream_flow_vals.append(latq)
        #stream_flow_vals = ncdata['qSfcLatRunoff'][:][idxs]
        #stream_flow_vals = ncdata['qBucket'][:][idxs]

        return np.array(stream_flow_vals)

# extract the lat long and streamflow for each comm ids
def extract_latq_NWM(current_netcdf_filename, NWM_masks):
    with nc.Dataset(current_netcdf_filename) as ncdataa:
   
        # extract NWM Common ID lateral discharge and return values
        stream_flow_vals = ncdata['q_lateral'][:][NWM_masks]
        #stream_flow_vals = ncdata['qSfcLatRunoff'][:][idxs]
        #stream_flow_vals = ncdata['qBucket'][:][idxs]

    return stream_flow_vals

def extract_inflow_NWM(current_netcdf_filename, NWM_masks):
    with nc.Dataset(current_netcdf_filename) as ncdataa:

        # extract NWM Common ID lateral discharge and return values
        stream_flow_vals = ncdataa['streamflow'][:][NWM_masks]
        #stream_flow_vals = ncdata['qSfcLatRunoff'][:][idxs]
        #stream_flow_vals = ncdata['qBucket'][:][idxs]

    return stream_flow_vals

def create_boundary_files(output_file: pathlib.Path, data: dict):
    """Create DFlow boundary files in bc text file format

    Args:
        output_dir (pathlib.Path): Output directory
        data (dict):

    Raises:
        FileNotFoundError: [description]

    Returns:
        [type]: [description]
    """
    units = [("time", "minutes since 2000-01-01 00:00:00"),
             ("lateral_discharge", "m^3/s")]
    #date_index = cftime.date2num(data['row_index'], units[0][1], calendar='julian')
    date_index = np.insert(cftime.date2num(data['row_index'], units[0][1], calendar='julian'),0,cftime.date2num(data['row_index'][0], units[0][1], calendar='julian')-60)
    values = data['lateral_q']
    ids = data['ids']

    with BCFileWriter(output_file) as bcwriter:
        #for i, commid in enumerate(data['col_index']):
        for i, commid in enumerate(data['col_index']):
            v = values[:, i]
        # Missing value, set to zero
            if v.mask.any():
                #continue
                v[:] = 0.0
            v = np.insert(v,0,0.0)
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
    try:
        chrt = next(input_path.glob("*.CHRTOUT_DOMAIN1"))
        extension = "CHRTOUT_DOMAIN1"
    except:
        chrt = next(input_path.glob("*.CHRTOUT_DOMAIN1.comp"))
        extension = "CHRTOUT_DOMAIN1.comp"
    while fdate <= end:
        cand = chrt.with_name(fdate.strftime('%Y%m%d%H%M.'+extension))
        if cand.exists():
            files[fdate] = cand
        else:
            raise FileNotFoundError(cand)
        fdate += one_hour
    return files

# function to grab and return the polygon
# latitude and longitude coordinates within shape file
def get_NWM_poly(NWM_ID,NWM_poly):

    idx = NWM_poly['ID'] == NWM_ID

    try:
        # Extract coordinates if data is a polygon
        x, y = NWM_poly['geometry'][np.where(idx==True)[0][0]].exterior.coords.xy
    except:
        # Extract coordinates if data is a multipolygon
        x, y = NWM_poly['geometry'][np.where(idx==True)[0][0]].geoms._get_geom_item(0).exterior.xy

    return np.array(x,dtype=float), np.array(y,dtype=float), NWM_ID

# function to grab and return the polygon
# latitude and longitude coordinates within shape file
def get_HUC_poly(NWM_lon,NWM_lat, HUC_poly):
    latq = geometry.Point((NWM_lon, NWM_lat))
    idx = HUC_poly.geometry.intersects(latq)
    if(HUC_poly['huc12'][np.where(idx==True)[0][0]][0] == '0'):
        HUC_ID = HUC_poly['huc12'][np.where(idx==True)[0][0]][1:]
    else:
        HUC_ID = HUC_poly['huc12'][np.where(idx==True)[0][0]]
    try:
        # Extract coordinates if data is a polygon
        x, y = HUC_poly['geometry'][np.where(idx==True)[0][0]].exterior.coords.xy
    except:
        # Extract coordinates if data is a multipolygon
        x, y = HUC_poly['geometry'][np.where(idx==True)[0][0]].geoms._get_geom_item(0).exterior.xy

    return np.array(x,dtype=float), np.array(y,dtype=float), HUC_ID
    #try:
    #    poly = np.array(HUC_poly['geometry'][np.where(idx==True)[0][0]].boundary.coords)
    #except:
    #    try:
    #        poly = np.array(HUC_poly['geometry'][np.where(idx==True)[0][0]].exterior.coords[:-1])
    #    except:
    #        poly = np.array([point for polygon in HUC_poly['geometry'][np.where(idx==True)[0][0]] for point in polygon.exterior.coords[:-1]])
    #return poly[:,0], poly[:,1], HUC_ID


# open and read streamline csv and append
# boolean flag for NWM common IDS within
# user specified polygon
def import_streamline_csv(HUC_csv_file):
    HUC_NWM_agg = pd.read_csv(HUC_csv_file,names=['rownum','NWMcommID','lon','lat','HUC_ID'])
    #HUC_NWM_agg = pd.read_csv(HUC_csv_file)
    #HUC_NWM_agg['NWMcommID'] = HUC_NWM_agg['NWMCommID']
    #HUC_NWM_agg = pd.read_csv(HUC_csv_file,names=['rownum','NWMcommID','lon','lat','HUC_ID','Lake'])
    #HUC_NWM_agg = pd.read_csv(HUC_csv_file,names=['rownum','NWMcommID','HUC_ID'])
    HUC_NWM_agg['Inside_Domain'] = False
    return HUC_NWM_agg

def import_streamline_csv2(fn):
    data = pd.read_csv(fn)
    data['Inside_Domain'] = False
    data = data.rename(columns={'HUC12_ID': "HUC_ID", 'NWM_ID': 'NWMcommID'})
    return data

# filter points by user specified polygon
def _point_in_polygon(test_polygon, longitude, latitude): #DONE
    line = geometry.LineString(test_polygon)
    point = geometry.Point(longitude, latitude)
    polygon = geometry.Polygon(line)
    result = polygon.contains(point)
    return result #True or False

# filter HUC csv file and append to dataframe
# which NWM common IDs are within user specifed
# polygon, then return dataframe with those
# only within user domain
def _filter_by_polygon(user_polygon, HUC_NWM_agg): #DONE
    for i, row in HUC_NWM_agg.iterrows():
        if (_point_in_polygon(user_polygon, row['lon'], row['lat']) == True):
            HUC_NWM_agg.at[i,'Inside_Domain'] = True
    HUC_NWM_agg_sliced = HUC_NWM_agg.loc[HUC_NWM_agg['Inside_Domain'] == True,:]
    old_tot = str(len(HUC_NWM_agg))
    new_tot = str(len(HUC_NWM_agg_sliced))
    print("With selection polygon, " + new_tot + " of initial " + old_tot + " streamlines remain.")
    return HUC_NWM_agg_sliced

# Now create the D-FlowFM formated lateral discharge
# boundary conditions within a newly created .ext file
# from HUC polygon files
def latq_ext_HUC(ext_file, HUC_NWM_agg_sliced, HUC_polygon_file, bc_file):
    # Create beginning of D-FlowFM ext file. This format
    # here is needed to properly initalize lateral
    # discharges over polygon areas in sourc code
    print("Creating pure HUC-12 polygon lateral discharge ext file.\n")
    ext_file = open(ext_file, 'a')
    # Read HUC-12 polygon shape file
    HUC_polygon = gpd.read_file(HUC_polygon_file)
    # Convert to WGS-84 to ensure crs is same for NWM
    HUC_polygon = HUC_polygon.to_crs('WGS-84')

    # Sort for the unique HUC-12 IDs within user domain polygon
    # and grab the polygon geospatial data
    HUC_NWM_agg_unique = HUC_NWM_agg_sliced.drop_duplicates(subset=['HUC_ID'])

    for i, row in HUC_NWM_agg_unique.iterrows():
        # Grab HUC-12 coordinates
        lons, lats, HUC_ID = get_HUC_poly(row['lon'],row['lat'],HUC_polygon)

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
        ext_file.write("id          = "+ str(HUC_ID) + '\n')
        ext_file.write("name        = "+ str(HUC_ID) + '\n')
        ext_file.write("type        = discharge"+'\n')
        ext_file.write("LocationType = all"+ '\n')
        ext_file.write("numCoordinates = "+ num_coords + '\n')
        ext_file.write("xCoordinates = "+ x_coord + '\n')
        ext_file.write("yCoordinates = "+ y_coord + '\n')
        ext_file.write("discharge   = "+str(bc_file).split('/')[-1]+'\n')
        ext_file.write('\n')

    ext_file.close()
    return

# Now create the D-FlowFM formated lateral discharge
# boundary conditions within a newly created .ext file
# from NWM polygon files
def latq_ext_NWM(ext_file, HUC_NWM_agg_sliced, NWM_polygon_file):
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
        lons, lats, NWM_ID = get_NWM_poly(row['NWMcommID'],NWM_polygon)

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
        ext_file.write("discharge   = BoundaryConditions.bc"+'\n')
        ext_file.write('\n')

    ext_file.close()
    return

# boundary conditions within a newly created .ext file
# from NWM polygon files
def latq_ext_NWM_inflow(ext_file, HUC_NWM_agg_sliced):
    # Create beginning of D-FlowFM ext file. This format
    # here is needed to properly initalize lateral
    # discharges over polygon areas in sourc code
    print("Creating pure NWM polygon lateral discharge ext file.\n")
    ext_file = open(ext_file, 'a')

    for i, row in HUC_NWM_agg_sliced.iterrows():
        # Grab HUC-12 coordinates
        #lon = str(row['long'])
        #lat = str(row['lat'])
        lon = str(row['END_X'])
        lat = str(row['END_Y'])
        NWM_ID = row['NWMcommID']

        # Write out lateral discharge template format
        ext_file.write("[lateral]"+'\n')
        ext_file.write("id          = "+ str(NWM_ID) + '\n')
        ext_file.write("name        = "+ str(NWM_ID) + '\n')
        ext_file.write("type        = discharge"+'\n')
        ext_file.write("LocationType = all"+ '\n')
        ext_file.write("numCoordinates = 1"+ '\n')
        ext_file.write("xCoordinates = "+ lon + '\n')
        ext_file.write("yCoordinates = "+ lat + '\n')
        ext_file.write("discharge   = BoundaryConditions.bc"+'\n')
        ext_file.write('\n')

    ext_file.close()
    return


# user defined options
def get_options():
    parser = argparse.ArgumentParser(description='Create lateral discharge ext DFlow subdomain file and extract aggregated data to .bc file based on user specified Polygon.')
    parser.add_argument('--HUC_agg_csv', dest='HUC_agg_csv', required=True, type=pathlib.Path, default=None,
                    help='The path to the GIS analysis csv file containing NWMCommIDs corresponding to each HUC-12 ID.')
    parser.add_argument('--HUC_polygon', dest='HUC_polygon', required=True, type=pathlib.Path, default=None,
                    help='GIS produced shape file that speifies the HUC-12 polygon coordinates based on the D-Flow extent of its meshes.')
    parser.add_argument('--NWM_polygon', dest='NWM_polygon', required=True, type=pathlib.Path, default=None,
                    help='GIS produced shape file that speifies the NWM polygon coordinates based on the D-Flow extent of its meshes.')
    parser.add_argument('--polygon', dest='polygon', required=True, type=pathlib.Path,
                    help='The path of the polygon file defining the region of interest.')
    parser.add_argument('--start', dest='start_time', required=True,
                    help='The date and time to begin making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--stop', dest='stop_time', required=True,
                    help='The date and time to stop making timeslice files for YYYY-mm-dd_HH:MM:SS')
    parser.add_argument('--input', dest='input_dir', required=True, type=pathlib.Path,
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

    if(args.HUC_polygon != None):
        # Read csv file with HUC-12 ID corresponding to NWM Common IDs
        HUC_NWM_agg_sliced = import_streamline_csv(args.HUC_agg_csv)
        # create latq ext file using "new" ext file format
        latq_ext_HUC(args.ext, HUC_NWM_agg_sliced, args.HUC_polygon,args.output)
    else:
        # Read csv file with HUC-12 ID corresponding to NWM Common IDs
        HUC_NWM_agg_sliced = import_streamline_csv(args.HUC_agg_csv)
        # create latq ext file using "new" ext file format
        latq_ext_NWM(args.ext, HUC_NWM_agg_sliced, args.NWM_polygon)
        #latq_ext_NWM_inflow(args.ext, HUC_NWM_agg_sliced)


    ############### BEGIN LATQ NWM file extraction ##################
    #get the path for the first input file
    files = get_inputfiles(args.input_dir, args.start_time, args.stop_time)
    print("Processing", len(files), "files in", args.input_dir)
    f0 = next(iter(files.values()))

    # mask = nwm streamflow mask for selected commids
    # fidx = array of indices to reorder selected commids to nwm streamflow order
    # (ie comm_ids[fidx] corresponds to data[mask]
    mask, feature_ids, fidx = extract_offsets(f0, HUC_NWM_agg_sliced['NWMcommID'].values)

    if(args.HUC_polygon != None):
        # add nwmcommid as string for
        ids_final = np.array(feature_ids,dtype=str)

        HUC_ids = HUC_NWM_agg_sliced['HUC_ID'].unique()

        HUC_ids = np.array(HUC_ids,dtype=float)
        HUC_ids_str = np.array(HUC_ids,dtype=int)
        HUC_ids_str = np.array(HUC_ids_str,dtype=str)

        # Create a mask for each unique HUC ID
        # indiciating which NWM common IDs are located
        # for each HUC within the CHRTOUT files
        HUC_masks = []
        for HUC_id in HUC_ids:
            HUCS_NWM_ids = HUC_NWM_agg_sliced.loc[HUC_NWM_agg_sliced.HUC_ID==HUC_id,'NWMcommID'].values
            idxs, feature_ids, fidx = extract_offsets(f0, HUCS_NWM_ids)
            HUC_masks.append(idxs)

        #HUC_masks_lake = []
        #HUC_masks_no_lake = []
        #lake = HUC_NWM_agg_sliced.loc[HUC_NWM_agg_sliced['Lake'] == True, :]
        #no_lake = HUC_NWM_agg_sliced.loc[HUC_NWM_agg_sliced['Lake'] == False, :]
        #for HUC_id in HUC_ids:
        #    HUCS_NWM_ids = no_lake.loc[no_lake.HUC_ID==HUC_id,'NWMcommID'].values
        #    idxs, feature_ids, fidx = extract_offsets(f0, HUCS_NWM_ids)
        #    HUC_masks_no_lake.append(idxs)
        #    HUCS_NWM_ids = lake.loc[lake.HUC_ID==HUC_id,'NWMcommID'].values
        #    idxs, feature_ids, fidx = extract_offsets(f0, HUCS_NWM_ids)
        #    HUC_masks_lake.append(idxs)

        #extract lat long and streamflow for the ids with stored offsets
        latq = np.ma.masked_array(np.zeros((len(files), len(HUC_ids))), fill_value=-9999)

        data = {'ids': HUC_ids_str,
                'col_index': HUC_ids_str,
                'row_index': list(files.keys())}

        print("Reading HUC aggregation lateral discharges...")
        ffiles = len(files)
        for i, f in enumerate(files.values()):
            #extract streamflow for the comm ids with stored offsets
            latq[i] = extract_latq_HUCS(f,HUC_ids,HUC_masks)
            #latq[i] = extract_latq_HUCS_Lake(f,HUC_ids, HUC_masks_lake, HUC_masks_no_lake)
            print("{}/{} ({:.2%})".format(i, ffiles, i/ffiles).ljust(20), end="\r")

    else:
        # add nwmcommid as string for
        ids_final = np.array(feature_ids,dtype=str)

        #extract lat long and streamflow for the ids with stored offsets
        latq = np.ma.masked_array(np.zeros((len(files), len(feature_ids))), fill_value=-9999)

        data = {'ids': ids_final,
                'col_index': ids_final,
                'row_index': list(files.keys())}

        print("Reading NWM polygon lateral discharges...")
        ffiles = len(files)
        for i, f in enumerate(files.values()):
            #extract streamflow for the comm ids with stored offsets
            #latq[i] = extract_latq_NWM(f,mask)
            latq[i] = extract_inflow_NWM(f,mask)
            print("{}/{} ({:.2%})".format(i, ffiles, i/ffiles).ljust(20), end="\r")

    # Set streamflow data
    data["lateral_q"] = latq

    # create latq .bc file using specified format
    create_boundary_files(args.output, data)

# Run main when this file is run
if __name__ == "__main__":
    args = get_options()
    main(args)
