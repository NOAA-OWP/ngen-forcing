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
