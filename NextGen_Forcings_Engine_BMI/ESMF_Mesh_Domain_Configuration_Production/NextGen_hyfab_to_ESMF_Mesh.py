import geopandas as gpd
from shapely.ops import orient
import netCDF4
import numpy as np
import pandas as pd
import argparse
import pathlib
import scipy
import os
import uuid

gpd.options.display_precision = 16
np.set_printoptions(precision=128)

"""
Script to perform a conversion between the NextGen Hydrofabric geopackage and an ESMF Unstructured Grid Format,
with the option of including hydrofabric model attribute data from a parquet file that allows the NextGen
hydrofabric domain configuration to utilize downscaling methods in the NextGen Forcings Engine

Example Usage:  python NextGen_hyfab_to_ESMF_Mesh.py ./nextgen_01.gpkg -parquet ./vpu1.parquet ./NextGen_VPU01_Mesh.nc
"""


def convert_hyfab_to_esmf(hyfab_gpkg: pathlib.Path, esmf_mesh_output: pathlib.Path, parquet: pathlib.Path | None = None):
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

    # convert hydrofabric data to spherical coordiantes
    hyfab = hyfab.to_crs('WGS84')

    # Eventually, we'll add code to slice catchment ids
    # but for now just use feature ids
    element_ids = np.array(np.array([elem.split('-')[1] for elem in np.array(hyfab.divide_id.values, dtype=str)], dtype=float), dtype=int)
    hyfab_coords = np.empty((len(element_ids), 2), dtype=float)
    hyfab_coords[:, 0] = element_ids
    hyfab_coords[:, 1] = element_ids

    # Sort data by feature id and reset index
    hyfab['element_id'] = element_ids
    hyfab = hyfab.sort_values(by=['element_id']).reset_index(drop=True)

    # Flag to see if user specified the hydrofabric parquet file for either VPU, subset, of CONUS
    if parquet is not None and os.path.isfile(parquet):

        # Open hydrofabric v2 parquet file containing the forcing
        # metadata that highlights catchment characteristics that
        # are needed to implement NCAR bias calibration and
        # downscaling methods within the forcings engine
        forcing_metadata = pd.read_parquet(parquet)
        forcing_metadata = forcing_metadata[['divide_id', 'elevation_mean', 'slope_mean', 'aspect_c_mean', 'X', 'Y']]
        forcing_metadata = forcing_metadata.sort_values('divide_id')
        forcing_metadata = forcing_metadata.reset_index()

        element_ids_parquet = np.array(np.array([elem.split('-')[1] for elem in np.array(forcing_metadata.divide_id.values, dtype=str)], dtype=float), dtype=int)
        parquet_coords = np.empty((len(element_ids_parquet), 2), dtype=float)
        parquet_coords[:, 0] = element_ids_parquet
        parquet_coords[:, 1] = element_ids_parquet

        dist, idx = scipy.spatial.KDTree(parquet_coords).query(hyfab_coords)

        hyfab['elevation'] = forcing_metadata.elevation_mean.values[idx]
        hyfab['slope'] = forcing_metadata.slope_mean.values[idx]
        hyfab['slope_azmuith'] = forcing_metadata.aspect_c_mean.values[idx]

        # remove metadata file to clear space
        del (forcing_metadata)

    # Get element count
    element_count = len(hyfab.element_id)

    # Array for describing number of nodes per element
    element_num_nodes = np.empty(element_count, dtype=np.int32)

    #Initialize the total number of nodes variable
    total_num_nodes = 0

    # Sanitized geometry list to ensure ESMF Mesh compliance
    # with the given hydrofabric
    sanitized_geoms = []

    for i in range(element_count):
        # Extract geometry for each element
        geom = hyfab.geometry[i]
    
        # If the catchment consists of multiple disconnected parts, keep only the
        # largest contiguous polygon to ensure a valid single-part ESMF element.
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda a: a.area)
        
        # Check for self-intersecting geometries (e.g., bowties) 
        # and apply a zero-buffer fix to re-compute valid topology.
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.geom_type == "MultiPolygon":
                geom = max(geom.geoms, key=lambda a: a.area)

        # Check for clockwise winding order and force a counter-clockwise orientation.
        # This ensures consistent surface normals and positive area calculations,
        # preventing triangulation failures during ESMF area-weighted regridding.
        if not geom.exterior.is_ccw:
            geom = orient(geom, sign=1.0)


        # Count nodes (exclude closure point)
        nodes = len(geom.exterior.coords) - 1
        total_num_nodes += nodes
        element_num_nodes[i] = nodes
        # Get ESMF compliant geometries
        sanitized_geoms.append(geom)


    # Extract Coordinates sequentially
    node_x_coord = np.empty(total_num_nodes, dtype=np.double)
    node_y_coord = np.empty(total_num_nodes, dtype=np.double)
    element_x_coord = np.empty(element_count, dtype=np.double)
    element_y_coord = np.empty(element_count, dtype=np.double)

    node_ptr = 0
    # Loop over sanitized geometries
    for i, geom in enumerate(sanitized_geoms):
        # Get geometry for the given element (catchment)
        coords = np.array(geom.exterior.coords)[:-1]
        # Get number of nodes associated with element
        num_nodes = element_num_nodes[i]

        # Assign the x and y node coordinates for the given element
        node_x_coord[node_ptr:node_ptr + num_nodes] = coords[:, 0]
        node_y_coord[node_ptr:node_ptr + num_nodes] = coords[:, 1]

        # Calculate centers
        element_x_coord[i] = geom.centroid.x
        element_y_coord[i] = geom.centroid.y

        # Iterate over number of nodes associated with previous element
        node_ptr += num_nodes

    # Global Node Unification : matches shared nodes between catchments to reduce file size
    node_df = pd.DataFrame({'x': node_x_coord, 'y': node_y_coord})
    # Group identical coordinates and assign a unique global ID
    node_df['global_id'] = node_df.groupby(['x', 'y']).ngroup() + 1

    # Final node connectivity of the mesh
    node_connectivity = node_df['global_id'].values.astype(np.int32)

    # Extract unique coordinates for the nodeCoords variable
    unique_nodes = node_df.drop_duplicates('global_id').sort_values('global_id')
    node_count = len(unique_nodes)
    node_x = unique_nodes['x'].values
    node_y = unique_nodes['y'].values

    # Get the pathway of the output directory and associated base name of file
    out_dir = os.path.dirname(esmf_mesh_output)
    base = os.path.basename(esmf_mesh_output)

    # Format: .<filename>.tmp.<UUID>
    # Hidden temp file tied to the final filename, guaranteed unique
    temp_path = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

    # Create ESMF mesh netcdf file
    nc = netCDF4.Dataset(temp_path, "w", format="NETCDF4")
    node_count_dim = nc.createDimension("nodeCount", node_count)
    elem_count_dim = nc.createDimension("elementCount", element_count)
    elem_conn_count_dim = nc.createDimension("connectionCount", len(node_connectivity))
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


    # Flag to whether include hydrofabric metadata if parquet file was specified
    if parquet is not None and os.path.isfile(parquet):
        hgt_elem_var = nc.createVariable("Element_Elevation", "f8", ("elementCount"))
        hgt_elem_var.long_name = "Catchment height above sea level"
        hgt_elem_var.units = "meters"
        slope_elem_var = nc.createVariable("Element_Slope", "f8", ("elementCount"))
        slope_elem_var.long_name = "Catchment slope"
        slope_elem_var.units = "meters"
        slope_azi_elem_var = nc.createVariable("Element_Slope_Azmuith", "f8", ("elementCount"))
        slope_azi_elem_var.long_name = "Catchment slope azmuith angle"
        slope_azi_elem_var.units = "Degrees"
        hgt_elem_var[:] = hyfab.elevation.values
        slope_elem_var[:] = hyfab.slope.values
        slope_azi_elem_var[:] = hyfab.slope_azmuith.values

    node_coords_var[:, 0] = node_x
    node_coords_var[:, 1] = node_y
    elem_conn_var[:] = node_connectivity
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
    parser.add_argument('-parquet', type=pathlib.Path, nargs='?', default=None, help="Hydrofabric parquet file pathway containing the model-attributes of the VPU or subset. This is only required if a user wants to utilize downscaling methods within the NextGen Forcings Engine")
    parser.add_argument("esmf_mesh_output", type=pathlib.Path, help="File pathway to save ESMF netcdf mesh file for hydrofabric")

    return parser.parse_args()


def main():
    args = get_options()
    convert_hyfab_to_esmf(
        hyfab_gpkg=args.hyfab_gpkg,
        esmf_mesh_output=args.esmf_mesh_output,
        parquet=args.parquet
    )


if __name__ == "__main__":
    main()
