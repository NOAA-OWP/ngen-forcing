import geopandas as gpd
import netCDF4
import numpy as np
import pandas as pd
import argparse
import pathlib
import os
import uuid

gpd.options.display_precision = 16
np.set_printoptions(precision=128)

"""
Script to perform a conversion between the NextGen Hydrofabric geopackage and an ESMF Unstructured Grid Format,
including hydrofabric model attribute data from the hydrofabric geopackage that allows the NextGen
hydrofabric domain configuration to utilize downscaling methods in the NextGen Forcings Engine

Example Usage:  python NextGen_hyfab_to_ESMF_Mesh.py ./nextgen_01.gpkg ./NextGen_VPU01_Mesh.nc
"""


def convert_hyfab_to_esmf(hyfab_gpkg: pathlib.Path, esmf_mesh_output: pathlib.Path):
    """
    Convert NextGen Hydrofabric geopackage into ESMF Mesh format

    :param hyfab_gpkg: Path to the hydrofabric geopackage file
    :param esmf_mesh_output: Path to the output ESMF mesh file
    :param parquet: Optional parquet file with hydrofabric model attributes
    """

    # Open hydrofabric geopackage file and
    # save copy of original cartesian coordinate system
    # for orientation properties since there are issues
    # with geopandas for converting crs and translating
    # orientation of polygon from original dataset
    hyfab = gpd.read_file(hyfab_gpkg, layer='divides')
    hyfab_cart = hyfab
    # convert hydrofabric data to spherical coordiantes
    hyfab = hyfab.to_crs('WGS84')

    # Eventually, we'll add code to slice catchment ids
    # but for now just use feature ids
    element_ids = np.array(hyfab.div_id.values, dtype=int)
    hyfab_coords = np.empty((len(element_ids), 2), dtype=float)
    hyfab_coords[:, 0] = element_ids
    hyfab_coords[:, 1] = element_ids

    # Sort data by feature id and reset index
    hyfab['element_id'] = element_ids
    hyfab_cart['element_id'] = element_ids
    hyfab = hyfab.sort_values(by=['element_id'])
    hyfab_cart = hyfab_cart.sort_values(by=['element_id'])
    hyfab = hyfab.reset_index()
    hyfab_cart = hyfab_cart.reset_index()

    # Get element count
    element_count = len(hyfab.element_id)

    # find the number of nodes in first element
    # based on geometry type
    if (hyfab.geometry[0].geom_type == "Polygon"):
        dup_df = pd.DataFrame([])
        dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[0].exterior.coords.xy
        dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
        elem_max_nodes = len(dup_df)
    else:
        dup_df = pd.DataFrame([])
        dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[0].geoms._get_geom_item(0).exterior.xy
        dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
        elem_max_nodes = len(dup_df)

    # Allocate element arrays for center point calculations
    # within hyrofabric data
    element_num_nodes = np.empty(element_count, dtype=np.int32)
    element_x_coord = np.empty(element_count, dtype=np.double)
    element_y_coord = np.empty(element_count, dtype=np.double)
    element_elevation = np.empty(element_count, dtype=np.double)
    element_slope = np.empty(element_count, dtype=np.double)
    element_slope_azmuith = np.empty(element_count, dtype=np.double)

    # Get the total number of nodes
    # throughout the entire hydrofabric domain
    # based on geometry type
    total_num_nodes = 0
    for i in range(element_count):
        if (hyfab.geometry[i].geom_type == "Polygon"):
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].exterior.coords.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
            total_num_nodes += len(dup_df)
        else:
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].geoms._get_geom_item(0).exterior.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
            total_num_nodes += len(dup_df)

    # assign current node id and allocate node arrays to extract
    # data from hydrofabric below
    node_id = np.arange(total_num_nodes) + 1
    node_x_coord = np.empty(total_num_nodes, dtype=np.double)
    node_y_coord = np.empty(total_num_nodes, dtype=np.double)
    node_start = 0

    # Extract node coordinates, calculate element data,
    # calculate max element of nodes through hydrofabric, and
    # flip node coordinates based on orientation of polygons
    # from the original cartesian coordinate system
    for i in range(element_count):
        if (hyfab.geometry[i].geom_type == "Polygon"):
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].exterior.coords.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
            node_x = dup_df.node_x.values
            node_y = dup_df.node_y.values
            ccw = hyfab_cart.geometry[i].exterior.is_ccw
        else:
            dup_df = pd.DataFrame([])
            dup_df['node_x'], dup_df['node_y'] = hyfab.geometry[i].geoms._get_geom_item(0).exterior.xy
            dup_df = dup_df.drop_duplicates(subset=['node_x', 'node_y'], keep='first')
            node_x = dup_df.node_x.values
            node_y = dup_df.node_y.values
            ccw = hyfab_cart.geometry[i].geoms._get_geom_item(0).exterior.is_ccw

        num_nodes = len(node_x)
        element_num_nodes[i] = num_nodes
        if (num_nodes > elem_max_nodes):
            elem_max_nodes = num_nodes

        element_x_coord[i] = hyfab.geometry[i].centroid.coords.xy[0][0]
        element_y_coord[i] = hyfab.geometry[i].centroid.coords.xy[1][0]

        element_elevation[i] = hyfab.elevation_mean[i]
        element_slope[i] = hyfab.slope1km_mean[i]
        element_slope_azmuith[i] = hyfab.aspect_circmean[i]  # NHF aspect is currently in radians, may need to be converted to degrees

        if (ccw):
            node_x_coord[node_start:node_start + num_nodes] = np.array(node_x, dtype=np.double)
            node_y_coord[node_start:node_start + num_nodes] = np.array(node_y, dtype=np.double)
        else:
            node_x_coord[node_start:node_start + num_nodes] = np.array(np.concatenate([[node_x[0]], np.flip(node_x[1:])]), dtype=np.double)
            node_y_coord[node_start:node_start + num_nodes] = np.array(np.concatenate([[node_y[0]], np.flip(node_y[1:])]), dtype=np.double)
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
    node_id_connectivity = np.empty(len(node_id), dtype=np.int32)
    node_count = 1
    for i in range(len(node_id)):
        if (i in duplicates_index):
            node_id_connectivity[i] = 0
        else:
            node_id_connectivity[i] = node_count
            node_count += 1

    # Assign new node id network to dataframe
    node_connectivity['node_id'] = node_id_connectivity

    # calculate the node id network to include its duplicate ids
    # for each instance of the node coordinates
    ESMF_node_id_connectivity = node_connectivity.groupby(['node_x', 'node_y']).node_id.transform('max')

    node_connectivity['node_id_connectivity'] = ESMF_node_id_connectivity.values

    node_connectivity_final = node_connectivity.node_id_connectivity.values

    # Extract only the unique node id network and respective coordinates
    node_connectivity = node_connectivity.drop_duplicates('node_id_connectivity')
    node_count = len(node_connectivity)
    node_x_coord_final = node_connectivity.node_x.values
    node_y_coord_final = node_connectivity.node_y.values

    # Calculate element connectivity from node id
    # network that includes duplicates
    elementConn = np.empty((element_count, elem_max_nodes), dtype=np.int32)
    elementConn[:, :] = -1
    start_index = 0
    end_index = 0
    for i in range(element_count):
        end_index += element_num_nodes[i]
        elementConn[i, 0:element_num_nodes[i]] = node_connectivity_final[start_index:end_index]
        start_index = end_index

    out_dir = os.path.dirname(esmf_mesh_output)
    base = os.path.basename(esmf_mesh_output)

    # Format: .<filename>.tmp.<UUID>
    # Hidden temp file tied to the final filename, guaranteed unique
    temp_path = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

    # Create ESMF mesh netcdf file
    nc = netCDF4.Dataset(temp_path, "w", format="NETCDF4")
    node_count_dim = nc.createDimension("nodeCount", node_count)
    elem_count_dim = nc.createDimension("elementCount", element_count)
    elem_conn_count_dim = nc.createDimension("connectionCount", len(node_connectivity_final))
    node_count_dim = nc.createDimension("coordDim", 2)
    node_coords_var = nc.createVariable("nodeCoords", 'f8', ("nodeCount", "coordDim"))
    node_coords_var.units = "degrees"
    elem_id = nc.createVariable("element_id", "i", "elementCount")
    elem_id.long_name = "Catchment ID for hydrofabric"
    elem_conn_var = nc.createVariable("elementConn", "i4", ("connectionCount"))
    elem_conn_var.long_name = "Node Indices that define the element connectivity"
    num_elem_conn_var = nc.createVariable("numElementConn", "i", "elementCount")
    num_elem_conn_var.long_name = "Number of nodes per element"
    center_coords_var = nc.createVariable("centerCoords", 'f8', ("elementCount", "coordDim"))
    center_coords_var.units = "degrees"
    nc.gridType = "unstructured"
    nc.version = "0.9"

    hgt_elem_var = nc.createVariable("Element_Elevation", "f8", ("elementCount"))
    hgt_elem_var.long_name = "Catchment height above sea level"
    hgt_elem_var.units = "meters"
    slope_elem_var = nc.createVariable("Element_Slope", "f8", ("elementCount"))
    slope_elem_var.long_name = "Catchment slope"
    slope_elem_var.units = "meters"
    slope_azi_elem_var = nc.createVariable("Element_Slope_Azmuith", "f8", ("elementCount"))
    slope_azi_elem_var.long_name = "Catchment slope azmuith angle"
    slope_azi_elem_var.units = "Degrees"
    hgt_elem_var[:] = element_elevation
    slope_elem_var[:] = element_slope
    slope_azi_elem_var[:] = element_slope_azmuith

    node_coords_var[:, 0] = node_x_coord_final
    node_coords_var[:, 1] = node_y_coord_final
    elem_conn_var[:] = node_connectivity_final
    num_elem_conn_var[:] = element_num_nodes
    center_coords_var[:, 0] = element_x_coord
    center_coords_var[:, 1] = element_y_coord
    elem_id[:] = hyfab.element_id.values

    nc.sync()
    nc.close()

    try:
        os.link(temp_path, esmf_mesh_output)

        # Give up the temporary name. The underlying file remains,
        # because 'esmf_mesh_output' now points to the same inode.
        os.remove(temp_path)

    except FileExistsError:
        # Another process already published the file.
        os.remove(temp_path)


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('hyfab_gpkg', type=pathlib.Path, help="Hydrofabric geopackage file pathway")
    parser.add_argument("esmf_mesh_output", type=pathlib.Path, help="File pathway to save ESMF netcdf mesh file for hydrofabric")

    return parser.parse_args()


def main():
    args = get_options()
    convert_hyfab_to_esmf(
        hyfab_gpkg=args.hyfab_gpkg,
        esmf_mesh_output=args.esmf_mesh_output,
    )


if __name__ == "__main__":
    main()
