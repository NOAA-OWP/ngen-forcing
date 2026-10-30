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

    element_count: int = len(hyfab.element_id)
    element_num_nodes: npt.NDArray[np.int32] = np.empty(element_count, dtype=np.int32)
    total_num_nodes: int = 0
    sanitized_geoms: list[shapely.Polygon] = []

    for i in range(element_count):
        raw_geom: BaseGeometry = hyfab.geometry[i]
        cleaned_geom: shapely.Polygon | None = clean_geometry_for_esmf(raw_geom, i)
        if cleaned_geom is None:
            continue
        sanitized_geoms.append(cleaned_geom)

        nodes: int = len(cleaned_geom.exterior.coords) - 1
        total_num_nodes += nodes
        element_num_nodes[i] = nodes

    node_x_coord: npt.NDArray[np.float64] = np.empty(total_num_nodes, dtype=np.float64)
    node_y_coord: npt.NDArray[np.float64] = np.empty(total_num_nodes, dtype=np.float64)
    element_x_coord: npt.NDArray[np.float64] = np.empty(element_count, dtype=np.float64)
    element_y_coord: npt.NDArray[np.float64] = np.empty(element_count, dtype=np.float64)

    node_ptr: int = 0
    for i, geom in enumerate(sanitized_geoms):
        coords: npt.NDArray[np.float64] = np.array(geom.exterior.coords)[:-1]
        num_nodes: int = int(element_num_nodes[i])

        node_x_coord[node_ptr : node_ptr + num_nodes] = coords[:, 0]
        node_y_coord[node_ptr : node_ptr + num_nodes] = coords[:, 1]
        element_x_coord[i] = geom.centroid.x
        element_y_coord[i] = geom.centroid.y
        node_ptr += num_nodes

    node_df: pd.DataFrame = pd.DataFrame({"x": node_x_coord, "y": node_y_coord})
    node_df["global_id"] = node_df.groupby(["x", "y"]).ngroup() + 1

    node_connectivity: npt.NDArray[np.int32] = node_df["global_id"].values.astype(np.int32)

    unique_nodes: pd.DataFrame = node_df.drop_duplicates("global_id").sort_values("global_id")
    node_count: int = len(unique_nodes)
    node_x: npt.NDArray[np.float64] = unique_nodes["x"].values
    node_y: npt.NDArray[np.float64] = unique_nodes["y"].values

    out_dir: str = os.path.dirname(esmf_mesh_output)
    base: str = os.path.basename(esmf_mesh_output)
    temp_path: str = os.path.join(out_dir, f".{base}.tmp.{uuid.uuid4()}")

    nc: netCDF4.Dataset = netCDF4.Dataset(temp_path, "w", format="NETCDF4")
    nc.createDimension("nodeCount", node_count)
    nc.createDimension("elementCount", element_count)
    nc.createDimension("connectionCount", len(node_connectivity))
    nc.createDimension("coordDim", 2)

    node_coords_var = nc.createVariable("nodeCoords", "f8", ("nodeCount", "coordDim"))
    node_coords_var.units = "degrees"
    elem_id = nc.createVariable("element_id", "i", "elementCount")
    elem_id.long_name = "Catchment ID for hydrofabric"
    elem_conn_var = nc.createVariable("elementConn", "i4", ("connectionCount",))
    elem_conn_var.long_name = "Node Indices that define the element connectivity"
    num_elem_conn_var = nc.createVariable("numElementConn", "i", "elementCount")
    num_elem_conn_var.long_name = "Number of nodes per element"
    center_coords_var = nc.createVariable("centerCoords", "f8", ("elementCount", "coordDim"))
    center_coords_var.units = "degrees"
    nc.gridType = "unstructured"
    nc.version = "0.9"

    if parquet_path is not None and os.path.isfile(parquet_path):
        hgt_elem_var = nc.createVariable("Element_Elevation", "f8", ("elementCount",))
        hgt_elem_var.long_name = "Catchment height above sea level"
        hgt_elem_var.units = "meters"
        slope_elem_var = nc.createVariable("Element_Slope", "f8", ("elementCount",))
        slope_elem_var.long_name = "Catchment slope"
        slope_elem_var.units = "meters"
        slope_azi_elem_var = nc.createVariable("Element_Slope_Azmuith", "f8", ("elementCount",))
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


def get_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hyfab_gpkg", type=pathlib.Path, help="Hydrofabric geopackage file pathway")
    parser.add_argument(
        "-parquet",
        dest="parquet_path",
        type=pathlib.Path,
        nargs="?",
        default=None,
        help="Hydrofabric parquet file pathway containing model-attributes.",
    )
    parser.add_argument("esmf_mesh_output", type=pathlib.Path, help="File pathway to save ESMF netcdf mesh file for hydrofabric")
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = get_options()
    convert_hyfab_to_esmf(
        hyfab_gpkg=args.hyfab_gpkg,
        esmf_mesh_output=args.esmf_mesh_output,
        parquet_path=args.parquet_path,
    )


if __name__ == "__main__":
    main()
