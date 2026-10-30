from __future__ import annotations

import argparse
import os
import pathlib
import typing

from ESMF_Mesh_Domain_Configuration_Production.NextGen_hyfab_to_ESMF_Mesh import (
    convert_hyfab_to_esmf,
)
from config_models import ConfigOptionsSubset


def create_mesh(cfg: ConfigOptionsSubset) -> None:
    """Create ESMF Mesh from geopackage file provided by the forcing engine config.

    :param cfg: ConfigOptionsSubset instance containing Geopackage, GeogridIn, and optional Parquet_Path
    """
    hyfab_name: str = cfg.Geopackage
    mesh_out_path: str = cfg.GeogridIn
    parquet_path: typing.Optional[str] = cfg.Parquet_Path

    if not os.path.exists(mesh_out_path):
        convert_hyfab_to_esmf(
            hyfab_gpkg=pathlib.Path(hyfab_name),
            esmf_mesh_output=pathlib.Path(mesh_out_path),
            parquet_path=pathlib.Path(parquet_path) if parquet_path else None,
        )
    else:
        print(f"ESMF mesh file already exists at {mesh_out_path}, skipping conversion.")
