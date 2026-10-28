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
