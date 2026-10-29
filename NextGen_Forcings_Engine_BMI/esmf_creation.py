import argparse
import os
from types import SimpleNamespace

import yaml

from ESMF_Mesh_Domain_Configuration_Production.NextGen_hyfab_to_ESMF_Mesh import (
    convert_hyfab_to_esmf,
)


def create_mesh(cfg: "ConfigOptions"):
    """Create ESMF Mesh from geopackage file provided by the forcing engine config.

    :param cfg: Object with attributes:
                - geopackage: path to hydrofabric geopackage file
                - geogrid: path to desired ESMF mesh output file
    """
    # Set the mesh file name based on the hydrofabric file
    hyfab_name = cfg.geopackage
    mesh_outPath = cfg.geogrid
    parquet_data = cfg.parquet

    # Check if the mesh file already exists and skip conversion if it does
    if not os.path.exists(mesh_outPath):
        convert_hyfab_to_esmf(hyfab_gpkg=hyfab_name, esmf_mesh_output=mesh_outPath, parquet=parquet_data)
    else:
        print(f"ESMF mesh file already exists at {mesh_outPath}, skipping conversion.")
