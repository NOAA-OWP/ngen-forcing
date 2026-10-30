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
