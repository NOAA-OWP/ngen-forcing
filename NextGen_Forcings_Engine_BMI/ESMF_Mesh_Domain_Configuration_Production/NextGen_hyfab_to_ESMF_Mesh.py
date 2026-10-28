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
