from __future__ import annotations

import argparse
import os
import pathlib
import uuid
import typing

import geopandas as gpd
import netCDF4
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial
import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import orient

gpd.options.display_precision = 16
np.set_printoptions(precision=128)


def extract_largest_polygon(geom: BaseGeometry | None) -> shapely.Polygon | None:
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return typing.cast(shapely.Polygon, geom)
    elif geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda a: a.area)
    elif geom.geom_type == "GeometryCollection":
        polys: list[shapely.Polygon] = [g for g in geom.geoms if g.geom_type == "Polygon"]
        if polys:
            return max(polys, key=lambda a: a.area)
        return None
    return None


def clean_geometry_for_esmf(
    geom: BaseGeometry | None,
    i: int,
    grid_size: float = 1e-6,
    simplify_tol: float = 1e-6,
) -> shapely.Polygon | None:
    if geom is None or geom.is_empty:
        return None

    poly: shapely.Polygon | None = extract_largest_polygon(geom)
    if poly is None:
        return None

    if not poly.is_valid:
        print(f"element {i + 1} has an invalid topology (e.g., bowtie)")
        poly = extract_largest_polygon(shapely.make_valid(poly))
        if poly is None:
            return None

    poly = extract_largest_polygon(shapely.set_precision(poly, grid_size=grid_size))
    if poly is None:
        return None

    poly = extract_largest_polygon(poly.simplify(simplify_tol, preserve_topology=True))
    if poly is not None and hasattr(poly, "exterior") and not poly.exterior.is_ccw:
        print(f"element {i + 1} is clockwise orientation, switching orientation.")
        poly = orient(poly, sign=1.0)

    return poly
