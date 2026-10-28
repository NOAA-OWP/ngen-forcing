import geopandas as gpd
import shapely
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
def extract_largest_polygon(geom):
    """
    Guarantees that the returned object is a single Shapely Polygon.
    Extracts the polygon with the largest area from MultiPolygons 
    or GeometryCollections.
    """
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return geom
    elif geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda a: a.area)
    elif geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if polys:
            return max(polys, key=lambda a: a.area)
        return None
    return None

def clean_geometry_for_esmf(geom, i, grid_size=1e-6, simplify_tol=1e-6):
    """
    Cleans GIS vector geometries for safe 3D spherical triangulation in ESMF.
    Always returns a single Polygon object with valid .exterior attributes.
    """
    if geom is None or geom.is_empty:
        return None

    geom = extract_largest_polygon(geom)
    if geom is None:
        return None

    if not geom.is_valid:
        print(f"element {i+1} has an invalid topology (e.g., bowtie)")
        geom = shapely.make_valid(geom)
        geom = extract_largest_polygon(geom)
        if geom is None:
            return None

    geom = shapely.set_precision(geom, grid_size=grid_size)
    geom = extract_largest_polygon(geom)

    geom = geom.simplify(simplify_tol, preserve_topology=True)
    geom = extract_largest_polygon(geom)

    if geom is not None and hasattr(geom, "exterior") and not geom.exterior.is_ccw:
        print(f"element {i+1} is clockwise orientation, switching orientation.")
        geom = orient(geom, sign=1.0)

    return geom

def convert_hyfab_to_esmf(hyfab_gpkg: pathlib.Path, esmf_mesh_output: pathlib.Path, parquet: pathlib.Path | None = None):
    """
    Convert NextGen Hydrofabric geopackage into ESMF Mesh format

    :param hyfab_gpkg: Path to the hydrofabric geopackage file
    :param esmf_mesh_output: Path to the output ESMF mesh file
    :param parquet: Optional parquet file with hydrofabric model attributes
    """
    hyfab = gpd.read_file(hyfab_gpkg, layer='divides')
    hyfab = hyfab.to_crs('WGS84')

    element_ids = np.array(np.array([elem.split('-')[1] for elem in np.array(hyfab.divide_id.values, dtype=str)], dtype=float), dtype=int)
    hyfab_coords = np.empty((len(element_ids), 2), dtype=float)
    hyfab_coords[:, 0] = element_ids
    hyfab_coords[:, 1] = element_ids

    hyfab['element_id'] = element_ids
    hyfab = hyfab.sort_values(by=['element_id']).reset_index(drop=True)

    if parquet is not None and os.path.isfile(parquet):
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

        del (forcing_metadata)

    element_count = len(hyfab.element_id)
    element_num_nodes = np.empty(element_count, dtype=np.int32)
    total_num_nodes = 0
    sanitized_geoms = []

    for i in range(element_count):
        geom = hyfab.geometry[i]
        geom = clean_geometry_for_esmf(geom, i)
        sanitized_geoms.append(geom)
        
        nodes = len(geom.exterior.coords) - 1
        total_num_nodes += nodes
        element_num_nodes[i] = nodes

    node_x_coord = np.empty(total_num_nodes, dtype=np.double)
    node_y_coord = np.empty(total_num_nodes, dtype=np.double)
    element_x_coord = np.empty(element_count, dtype=np.double)
    element_y_coord = np.empty(element_count, dtype=np.double)

    node_ptr = 0
    for i, geom in enumerate(sanitized_geoms):
        coords = np.array(geom.exterior.coords)[:-1]
        num_nodes = element_num_nodes[i]

        node_x_coord[node_ptr:node_ptr + num_nodes] = coords[:, 0]
        node_y_coord[node_ptr:node_ptr + num_nodes] = coords[:, 1]

        element_x_coord[i] = geom.centroid.x
        element_y_coord[i] = geom.centroid.y

        node_ptr += num_nodes

    node_df = pd.DataFrame({'x': node_x_coord, 'y': node_y_coord})
    node_df['global_id'] = node_df.groupby(['x', 'y']).ngroup() + 1

    node_connectivity = node_df['global_id'].values.astype(np.int32)

    unique_nodes = node_df.drop_duplicates('global_id').sort_values('global_id')
    node_count = len(unique_nodes)
    node_x = unique_nodes['x'].values
    node_y = unique_nodes['y'].values

    out_dir = os.path.dirname(esmf_mesh_output)
    base = os.path.basename(esmf_mesh_output)
    temp_path = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

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
        os.remove(temp_path)
    except FileExistsError:
        os.remove(temp_path)
