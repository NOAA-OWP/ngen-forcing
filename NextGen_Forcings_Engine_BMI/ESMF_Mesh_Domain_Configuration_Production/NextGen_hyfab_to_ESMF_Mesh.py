import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import argparse
import pathlib
import scipy
gpd.options.display_precision=16
np.set_printoptions(precision=128)

"""
Script to perform a conversion between the NextGen Hydrofabric geopackage and an ESMF Unstructured Grid Format,
with the option of including hydrofabric model attribute data from a parquet file that allows the NextGen
hydrofabric domain configuration to utilize downscaling methods in the NextGen Forcings Engine

Example Usage:  python NextGen_hyfab_to_ESMF_Mesh.py ./nextgen_01.gpkg -parquet ./vpu1.parquet ./NextGen_VPU01_Mesh.nc
"""

def main(args):
    # Open hydrofabric geopackage file and
    # save copy of original cartesian coordinate system
    # for orientation properties since there are issues
    # with geopandas for converting crs and translating
    # orientation of polygon from original dataset
    hyfab = gpd.read_file(args.hyfab_gpkg,layer='divides')
    hyfab_cart = hyfab
    # convert hydrofabric data to spherical coordiantes
    hyfab = hyfab.to_crs('WGS84')

    # Eventually, we'll add code to slice catchment ids
    # but for now just use feature ids
    element_ids = np.array(np.array([elem.split('-')[1] for elem in np.array(hyfab.divide_id.values,dtype=str)],dtype=float),dtype=int)
    hyfab_coords = np.empty((len(element_ids),2),dtype=float)
    hyfab_coords[:,0] = element_ids
    hyfab_coords[:,1] = element_ids


    # Sort data by feature id and reset index
    hyfab['element_id'] = element_ids
    hyfab_cart['element_id'] = element_ids
    hyfab = hyfab.sort_values(by=['element_id'])
    hyfab_cart = hyfab_cart.sort_values(by=['element_id'])
    hyfab = hyfab.reset_index()
    hyfab_cart = hyfab_cart.reset_index()

    # Flag to see if user specified the hydrofabric parquet file for either VPU, subset, of CONUS
    if(args.parquet != None):

        # Open hydrofabric v2 parquet file containing the forcing
        # metadata that highlights catchment characteristics that
        # are needed to implement NCAR bias calibration and
        # downscaling methods within the forcings engine
        forcing_metadata = pd.read_parquet(args.parquet)
        forcing_metadata = forcing_metadata[['divide_id', 'elevation_mean', 'slope_mean','aspect_c_mean','X', 'Y']]
        forcing_metadata = forcing_metadata.sort_values('divide_id')
        forcing_metadata = forcing_metadata.reset_index()

        element_ids_parquet = np.array(np.array([elem.split('-')[1] for elem in np.array(forcing_metadata.divide_id.values,dtype=str)],dtype=float),dtype=int)
        parquet_coords = np.empty((len(element_ids_parquet),2),dtype=float)
        parquet_coords[:,0] = element_ids_parquet
        parquet_coords[:,1] = element_ids_parquet

        dist, idx = scipy.spatial.KDTree(parquet_coords).query(hyfab_coords)

        hyfab['elevation'] = forcing_metadata.elevation_mean.values[idx]
        hyfab['slope'] = forcing_metadata.slope_mean.values[idx]
        hyfab['slope_azmuith'] = forcing_metadata.aspect_c_mean.values[idx]

        # remove metadata file to clear space
        del(forcing_metadata)

    # Get element count
    element_count = len(hyfab.element_id)


    # find the number of nodes in first element
    # based on geometry type
    if(hyfab.geometry[0].geom_type == "Polygon"):
        dup_df = pd.DataFrame([])
        dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[0].exterior.coords.xy
        dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
        elem_max_nodes = len(dup_df)
    else:
        dup_df = pd.DataFrame([])
        dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[0].geoms._get_geom_item(0).exterior.xy
        dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
        elem_max_nodes = len(dup_df)

    # Allocate element arrays for center point calculations
    # within hyrofabric data
    element_num_nodes = np.empty(element_count,dtype=np.int32)
    element_x_coord = np.empty(element_count,dtype=np.double)
    element_y_coord = np.empty(element_count,dtype=np.double)
    if(args.parquet != None):
        element_elevation = np.empty(element_count,dtype=np.double)
        element_slope = np.empty(element_count,dtype=np.double)
        element_slope_azmuith = np.empty(element_count,dtype=np.double)

    # Get the total number of nodes
    # throughout the entire hydrofabric domain
    # based on geometry type
    total_num_nodes = 0
    for i in range(element_count):
        if(hyfab.geometry[i].geom_type == "Polygon"):
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].exterior.coords.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
            total_num_nodes += len(dup_df)
        else:
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].geoms._get_geom_item(0).exterior.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
            total_num_nodes += len(dup_df)

    # assign current node id and allocate node arrays to extract
    # data from hydrofabric below
    node_id = np.arange(total_num_nodes)+1
    node_x_coord = np.empty(total_num_nodes,dtype=np.double)
    node_y_coord = np.empty(total_num_nodes,dtype=np.double)
    node_start = 0

    # Extract node coordinates, calculate element data,
    # calculate max element of nodes through hydrofabric, and
    # flip node coordinates based on orientation of polygons
    # from the original cartesian coordinate system
    for i in range(element_count):
        if(hyfab.geometry[i].geom_type == "Polygon"):
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].exterior.coords.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
            node_x = dup_df.node_x.values
            node_y = dup_df.node_y.values
            ccw = hyfab_cart.geometry[i].exterior.is_ccw
        else:
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].geoms._get_geom_item(0).exterior.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x','node_y'],keep='first')
            node_x = dup_df.node_x.values
            node_y = dup_df.node_y.values
            ccw = hyfab_cart.geometry[i].geoms._get_geom_item(0).exterior.is_ccw

        num_nodes = len(node_x)
        element_num_nodes[i] = num_nodes
        if(num_nodes > elem_max_nodes):
            elem_max_nodes = num_nodes

        element_x_coord[i] = hyfab.geometry[i].centroid.coords.xy[0][0]
        element_y_coord[i] = hyfab.geometry[i].centroid.coords.xy[1][0]

        if(args.parquet != None):
            element_elevation[i] = hyfab.elevation[i]
            element_slope[i] = hyfab.slope[i]
            element_slope_azmuith[i] = hyfab.slope_azmuith[i]

        if(ccw):
            node_x_coord[node_start:node_start+num_nodes] = np.array(node_x,dtype=np.double)
            node_y_coord[node_start:node_start+num_nodes] = np.array(node_y,dtype=np.double)
        else:
            node_x_coord[node_start:node_start+num_nodes] = np.array(np.concatenate([[node_x[0]],np.flip(node_x[1:])]),dtype=np.double)
            node_y_coord[node_start:node_start+num_nodes] = np.array(np.concatenate([[node_y[0]],np.flip(node_y[1:])]),dtype=np.double)
        node_start += num_nodes

    # Assign node data to pandas dataframe
    # and calculate the duplicate nodes throughout
    # the hydrofabric geometry network
    node_connectivity = pd.DataFrame([])
    node_connectivity['node_x'] = node_x_coord
    node_connectivity['node_y'] = node_y_coord

    duplicates = node_connectivity[node_connectivity.duplicated(keep='first')]

    # Create array to assign duplicate nodes as
    # zeroes, while creating unique ids for only
    # the first instance of the unique node
    duplicates_index = duplicates.index
    node_id_connectivity = np.empty(len(node_id),dtype=np.int32)
    node_count = 1
    for i in range(len(node_id)):
        if(i in duplicates_index):
            node_id_connectivity[i] = 0
        else:
            node_id_connectivity[i] = node_count
            node_count += 1

    # Assign new node id network to dataframe
    node_connectivity['node_id'] = node_id_connectivity

    # calculate the node id network to include its duplicate ids
    # for each instance of the node coordinates
    ESMF_node_id_connectivity = node_connectivity.groupby(['node_x','node_y']).node_id.transform('max')

    node_connectivity['node_id_connectivity'] = ESMF_node_id_connectivity.values

    node_connectivity_final = node_connectivity.node_id_connectivity.values

    # Extract only the unique node id network and respective coordinates
    node_connectivity = node_connectivity.drop_duplicates('node_id_connectivity')
    node_count = len(node_connectivity)
    node_x_coord_final = node_connectivity.node_x.values
    node_y_coord_final = node_connectivity.node_y.values

    # Calculate element connectivity from node id
    # network that includes duplicates
    elementConn = np.empty((element_count,elem_max_nodes),dtype=np.int32)
    elementConn[:,:] = -1
    start_index = 0
    end_index = 0
    for i in range(element_count):
        end_index += element_num_nodes[i]
        elementConn[i,0:element_num_nodes[i]] = node_connectivity_final[start_index:end_index]
        start_index = end_index

    # Create ESMF mesh netcdf file
    nc = netCDF4.Dataset(args.esmf_mesh_output, "w", format="NETCDF4")
    node_count_dim = nc.createDimension("nodeCount", node_count)
    elem_count_dim = nc.createDimension("elementCount", element_count)
    elem_conn_count_dim = nc.createDimension("connectionCount", len(node_connectivity_final))
    node_count_dim = nc.createDimension("coordDim", 2)
    node_coords_var = nc.createVariable("nodeCoords",'f8',("nodeCount","coordDim"))
    node_coords_var.units = "degrees"
    elem_id = nc.createVariable("element_id","i","elementCount")
    elem_id.long_name = "Catchment ID for hydrofabric"
    elem_conn_var = nc.createVariable("elementConn","i4",("connectionCount"))
    elem_conn_var.long_name = "Node Indices that define the element connectivity"
    num_elem_conn_var = nc.createVariable("numElementConn","i","elementCount")
    num_elem_conn_var.long_name = "Number of nodes per element"
    center_coords_var = nc.createVariable("centerCoords",'f8',("elementCount","coordDim"))
    center_coords_var.units = "degrees"
    nc.gridType = "unstructured"
    nc.version = "0.9"

    # Flag to whether include hydrofabric metadata if parquet file was specified
    if(args.parquet != None):
        hgt_elem_var = nc.createVariable("Element_Elevation","f8",("elementCount"))
        hgt_elem_var.long_name = "Catchment height above sea level"
        hgt_elem_var.units = "meters"
        slope_elem_var = nc.createVariable("Element_Slope","f8",("elementCount"))
        slope_elem_var.long_name = "Catchment slope"
        slope_elem_var.units = "meters"
        slope_azi_elem_var = nc.createVariable("Element_Slope_Azmuith","f8",("elementCount"))
        slope_azi_elem_var.long_name = "Catchment slope azmuith angle"
        slope_azi_elem_var.units = "Degrees"
        hgt_elem_var[:] = element_elevation
        slope_elem_var[:] = element_slope
        slope_azi_elem_var[:] = element_slope_azmuith

    node_coords_var[:, 0] = node_x_coord_final
    node_coords_var[:, 1] = node_y_coord_final
    elem_conn_var[:] = node_connectivity_final
    num_elem_conn_var[:] = element_num_nodes
    center_coords_var[:,0] = element_x_coord
    center_coords_var[:,1] = element_y_coord
    elem_id[:] = hyfab.element_id.values
    nc.close()

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('hyfab_gpkg', type=pathlib.Path, help="Hydrofabric geopackage file pathway")
    parser.add_argument('-parquet',type=pathlib.Path, nargs='?', default = None, help="Hydrofabric parquet file pathway containing the model-attributes of the VPU or subset. This is only required if a user wants to utilize downscaling methods within the NextGen Forcings Engine")
    parser.add_argument("esmf_mesh_output", type=pathlib.Path, help="File pathway to save ESMF netcdf mesh file for hydrofabric")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_options()
    main(args)
