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


def convert_hyfab_to_esmf(
    hyfab_gpkg: pathlib.Path,
    esmf_mesh_output: pathlib.Path,
    parquet_path: pathlib.Path | None = None,
) -> None:
    hyfab: gpd.GeoDataFrame = gpd.read_file(hyfab_gpkg, layer="divides")
    hyfab = hyfab.to_crs("WGS84")

    element_ids: npt.NDArray[np.int64] = np.array(
        [int(elem.split("-")[1]) for elem in hyfab.divide_id.values],
        dtype=np.int64,
    )
    hyfab_coords: npt.NDArray[np.float64] = np.empty((len(element_ids), 2), dtype=np.float64)
    hyfab_coords[:, 0] = element_ids
    hyfab_coords[:, 1] = element_ids

    hyfab["element_id"] = element_ids
    hyfab = hyfab.sort_values(by=["element_id"]).reset_index(drop=True)

    if parquet_path is not None and os.path.isfile(parquet_path):
        forcing_metadata: pd.DataFrame = pd.read_parquet(parquet_path)
        forcing_metadata = forcing_metadata[
            ["divide_id", "elevation_mean", "slope_mean", "aspect_c_mean", "X", "Y"]
        ]
        forcing_metadata = forcing_metadata.sort_values("divide_id").reset_index(drop=True)

        element_ids_parquet: npt.NDArray[np.int64] = np.array(
            [int(elem.split("-")[1]) for elem in forcing_metadata.divide_id.values],
            dtype=np.int64,
        )
        parquet_coords: npt.NDArray[np.float64] = np.empty((len(element_ids_parquet), 2), dtype=np.float64)
        parquet_coords[:, 0] = element_ids_parquet
        parquet_coords[:, 1] = element_ids_parquet

        dist: npt.NDArray[np.float64]
        idx: npt.NDArray[np.int64]
        dist, idx = scipy.spatial.KDTree(parquet_coords).query(hyfab_coords)

        hyfab["elevation"] = forcing_metadata.elevation_mean.values[idx]
        hyfab["slope"] = forcing_metadata.slope_mean.values[idx]
        hyfab["slope_azmuith"] = forcing_metadata.aspect_c_mean.values[idx]
        del forcing_metadata
