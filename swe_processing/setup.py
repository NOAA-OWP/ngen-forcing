from setuptools import setup, find_packages

setup(
    name="swe_processing",
    packages=find_packages(),
    install_requires=[
        'xarray',
        'matplotlib',
        'cartopy',
        'numpy',
        'scipy',
        'geopandas~=1.0',
        'shapely~=2.0',
        'fsspec',
        'pandas',
        's3fs',
        'dask[complete]',  # Includes Dask core + recommended dependencies
        'distributed',  # Adds support for parallel computing
        'requests'
    ],
    python_requires='>=3.8',
)
