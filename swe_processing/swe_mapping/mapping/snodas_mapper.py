import argparse
import time
import cartopy.crs as ccrs
import fsspec
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
from shapely.geometry import Point
from ..utility.swe_minmax import get_minmax
from ..utility.snotel_utils import SnotelDataLoader, SnotelCalculator, SnotelPlotter
from ..utility.geo_utils import GeoUtils
from ..utility.plot_utils import PlotUtils

class DataLoader:
    @staticmethod
    def snodas_path_constructor(date, s3_mount_point, direct_s3):
        """
        Construct the S3 path to SNODAS NetCDF file.
        
        Parameters
        ----------
        date : str
            Date string in format 'YYYY-MM-DD'
            
        Returns
        -------
        str
            S3 path to the SNODAS NetCDF file
        """
        file_date = date.replace('-', '')

        if not direct_s3:
            snodas_file = f"{s3_mount_point}/ngwpc-forcing/snodas_nc/zz_ssmv11034tS__T0001TTNATS{file_date}05HP001.nc"
            if os.path.exists(snodas_file):
                return snodas_file
            else:
                raise FileNotFoundError(f"File not found in local mount: {snodas_file}")
        else:
            snodas_file = f"s3://ngwpc-forcing/snodas_nc/zz_ssmv11034tS__T0001TTNATS{file_date}05HP001.nc"
            fs = fsspec.filesystem('s3')
            if fs.exists(snodas_file):
                return snodas_file
            else:
                raise FileNotFoundError(f"File not found in S3: {snodas_file}")

    @staticmethod
    def list_snotel_filenames(s3_mount_point, snotel_s3_path, direct_s3):
        """
        List SNOTEL CSV files available in the S3 bucket.
        """
        return SnotelDataLoader.list_snotel_filenames(s3_mount_point, snotel_s3_path, direct_s3)

    @staticmethod
    def parse_snotel_filenames(filenames):
        """
        Parse latitude and longitude from SNOTEL filenames and create a GeoDataFrame.
        """
        return SnotelDataLoader.parse_snotel_filenames(filenames)
    
    @staticmethod
    def load_snotel_data(stations_in_basin, date, fs, s3_mount_point, snotel_s3_path):
        """
        Load SNOTEL SWE data for stations within the basin for a specific date.
        """
        return SnotelDataLoader.load_snotel_data(stations_in_basin, date, fs, s3_mount_point, snotel_s3_path)

    @staticmethod
    def load_netcdf(snodas_file):
        """
        Load SNODAS NetCDF data from an S3 path with chunking.
        
        Parameters
        ----------
        snodas_file : str
            S3 path to the SNODAS NetCDF file
        
        Returns
        -------
        xarray.Dataset
            Loaded xarray Dataset containing SNODAS data
        """
        t0 = time.time()

        chunk_size = 100

        # Open the SNODAS NetCDF file with chunking to optimize performance and memory usage.
        print('Opening netCDF file', snodas_file)
        with fsspec.open(snodas_file, mode='rb') as f:
            snodas_ds = xr.open_dataset(f, chunks={"time": 1, "lat": chunk_size, "lon": chunk_size})

        print(f"   SNODAS NetCDF load time: {time.time() - t0:.2f}s")

        return snodas_ds

    @staticmethod
    def read_geo(gpkg_file):
        """
        Read the 'divides' layer from a geopackage file and ensure geographic CRS.
        """
        return GeoUtils.read_geo(gpkg_file)

    @staticmethod
    def get_basin_geometry(basin_gdf):
        """
        Extract a unified basin geometry and bounds from a GeoDataFrame.
        """
        return GeoUtils.get_basin_geometry(basin_gdf)


class Calculator:
    @staticmethod
    def subset_to_basin(ds, basin_geometry):
        """
        Subset a dataset to the basin extent and prepare for analysis.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing SNODAS data
        basin_geometry : shapely.geometry
            Basin geometry to use for subsetting
            
        Returns
        -------
        tuple
            (ds_subset, lons, lats) where:
                - ds_subset is the subsetted xarray Dataset
                - lons is a 2D numpy array of longitudes
                - lats is a 2D numpy array of latitudes
        """
        # Use bounding box to filter dataset
        bounds = basin_geometry.bounds
        lon_mask = (ds.lon >= bounds[0]) & (ds.lon <= bounds[2])
        lat_mask = (ds.lat >= bounds[1]) & (ds.lat <= bounds[3])

        # Apply the mask and compute to load required data
        ds_subset = ds.where(lon_mask & lat_mask, drop=True).compute()
        # `.compute()` ensures that the filtered dataset is fully materialized.

        # Convert units from millimeters to meters
        ds_subset['Band1'] = ds_subset.Band1 / 1000

        # Create meshgrid of lat/lon values
        lons, lats = np.meshgrid(ds_subset.lon, ds_subset.lat)

        return ds_subset, lons, lats

    @staticmethod
    def find_stations_in_basin(stations_gdf, basin_geometry):
        """
        Find SNOTEL stations that fall within the basin geometry.
        """
        return SnotelCalculator.find_stations_in_basin(stations_gdf, basin_geometry)

    @staticmethod
    def calculate_catchment_mean(ds, basin_geometry, gdf):
        """
        Calculate mean SWE for each catchment and apply to dataset.
        
        Parameters
        ----------
        ds : xarray.Dataset
            Input dataset containing SNODAS data
        basin_geometry : shapely.geometry
            Basin geometry containing all catchments
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with catchment boundaries
            
        Returns
        -------
        tuple
            (gdf_with_swe, ds_subset) where:
                - gdf_with_swe is the GeoDataFrame with added 'mean_swe' column
                - ds_subset is the xarray Dataset with catchment means applied
        """
        # Subset to basin and compute to ensure faster processing
        ds_subset, lons, lats = Calculator.subset_to_basin(ds, basin_geometry)

        # Convert meshgrid into Points for fast spatial operations
        points = np.array([Point(x, y) for x, y in zip(lons.ravel(), lats.ravel())])

        mean_values = []
        for _, row in gdf.iterrows():
            # Mask for each catchment using Shapely `contains`
            mask = np.array([row.geometry.contains(pt) for pt in points]).reshape(lons.shape)

            # Extract SWE data for catchment and compute mean
            catchment_data = ds_subset.Band1.where(mask)
            mean_swe = float(catchment_data.mean().compute()) 
            mean_values.append(mean_swe)

            # Apply computed mean value efficiently
            ds_subset['Band1'] = xr.where(mask, mean_swe, ds_subset.Band1)

        # Add computed mean values to GeoDataFrame
        gdf_with_swe = gdf.copy()
        gdf_with_swe['mean_swe'] = mean_values

        # Mask everything outside the basin
        basin_mask = np.array([basin_geometry.contains(pt) for pt in points]).reshape(lons.shape)
        ds_subset['Band1'] = ds_subset.Band1.where(basin_mask)

        return gdf_with_swe, ds_subset


class Plotter:
    @staticmethod
    def create_base_plot():
        """
        Create a base map plot with cartopy projection.
        """
        return PlotUtils.create_base_plot()

    @staticmethod
    def set_map_extent(ax, bounds, proj):
        """
        Set the map extent with appropriate buffers around bounds.
        """
        return PlotUtils.set_map_extent(ax, bounds, proj)

    @staticmethod
    def plot_catchment_boundaries(ax, gdf, proj):
        """
        Add catchment boundaries to a map plot.
        """
        return PlotUtils.plot_catchment_boundaries(ax, gdf, proj)

    @staticmethod
    def add_basin_overlay(ax, basin_geometry, proj):
        """
        Add the basin outline to a map plot.
        """
        return PlotUtils.add_basin_overlay(ax, basin_geometry, proj)

    @staticmethod
    def add_gridlines(ax):
        """
        Add gridlines to a map plot.
        """
        return PlotUtils.add_gridlines(ax)

    @staticmethod
    def add_colorbar(im, ax):
        """
        Add a colorbar to a map plot.
        """
        return PlotUtils.add_colorbar(im, ax)
 
    @staticmethod
    def add_snotel_overlay(ax, snotel_data, proj):
        """
        Add SNOTEL SWE data as text overlays on a map.
        """
        return SnotelPlotter.add_snotel_overlay(ax, snotel_data, proj)
    
    @staticmethod
    def plot_raw_swe(ax, ds, basin_geometry, gdf, proj):
        """
        Plot raw SWE values with the basin boundary.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on
        ds : xarray.Dataset
            Dataset containing SNODAS data
        basin_geometry : shapely.geometry
            Basin geometry for masking
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with catchment boundaries
        proj : cartopy.crs
            Projection to use for plot
            
        Returns
        -------
        tuple
            (ax, im, vmin, vmax) where:
                - ax is the updated matplotlib Axes
                - im is the plotted image
                - vmin, vmax are the colormap scale limits
        """
        # Subset dataset
        ds_subset, lons, lats = Calculator.subset_to_basin(ds, basin_geometry)

        # Convert lat/lon into Points for spatial operations
        from shapely.geometry import Point
        points = np.array([Point(x, y) for x, y in zip(lons.ravel(), lats.ravel())])
        basin_mask = np.array([basin_geometry.contains(pt) for pt in points]).reshape(lons.shape)

        # Mask invalid values & apply basin mask
        swe_data = ds_subset.Band1.where(ds_subset.Band1 != -9999).where(basin_mask)

        # Compute min/max values for colormap scaling
        vmin, vmax = get_minmax(swe_data.compute())

        # Create colormesh plot
        im = ax.pcolormesh(ds_subset.lon, ds_subset.lat.compute(), swe_data,
                           transform=proj,
                           cmap='Blues',
                           vmin=vmin,
                           vmax=vmax,
                           shading='auto')

        # Add catchment boundaries
        ax = Plotter.plot_catchment_boundaries(ax, gdf, proj)

        return ax, im, vmin, vmax

    @staticmethod
    def plot_polygon_swe(ax, gdf, proj):
        """
        Plot catchment polygons colored by mean SWE values.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to plot on
        gdf : geopandas.GeoDataFrame
            GeoDataFrame with catchment boundaries and 'mean_swe' column
        proj : cartopy.crs
            Projection to use for plot
            
        Returns
        -------
        tuple
            (ax, sm, vmin, vmax) where:
                - ax is the updated matplotlib Axes
                - sm is the ScalarMappable for colorbar
                - vmin, vmax are the colormap scale limits
        """
        # Use vmin and vmax to explicitly define a colorbar
        vmin, vmax = get_minmax(gdf['mean_swe'])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Blues
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Plot filled polygons
        for _, row in gdf.iterrows():
            ax.add_geometries([row.geometry], crs=proj,
                              facecolor=cmap(norm(row['mean_swe'])),
                              edgecolor='none')

        # Add catchment boundaries
        ax = Plotter.plot_catchment_boundaries(ax, gdf, proj)

        return ax, sm, vmin, vmax

    @staticmethod
    def save_figure(fig, output_file):
        """
        Save a figure to a file.
        """
        return PlotUtils.save_figure(fig, output_file)


class SNODASProcessor:
    def __init__(self, date=None, gpkg_file=None, output_file_raw=None, output_file_lumped=None, direct_s3=False):
        """
        Initialize the SNODAS Processor.
        
        Parameters
        ----------
        date : str, optional
            Date string in format 'YYYY-MM-DD'
        gpkg_file : str, optional
            Path to geopackage file with basin/catchment boundaries
        output_file_raw : str, optional
            Path where raw SWE visualization will be saved
        output_file_lumped : str, optional
            Path where catchment-averaged visualization will be saved
        """
        # Initialize input parameters
        self.date = date
        self.gpkg_file = gpkg_file
        self.output_file_raw = output_file_raw
        self.output_file_lumped = output_file_lumped
        self.direct_s3 = direct_s3
        
        # Initialize data attributes
        self.snodas_file = None
        self.snodas_ds = None
        self.basin_gdf = None
        self.basin_geometry = None
        self.bounds = None
        self.s3_mount_point = None
        
        # Initialize SNOTEL-related attributes
        self.snotel_filenames = None
        self.stations_gdf = None
        self.stations_in_basin = None
        self.snotel_data = None
        self.snotel_filesystem = None
        self.snotel_s3_path = None

        # Initialize plot attributes
        self.raw_fig = None
        self.raw_ax = None
        self.raw_im = None
        self.catchment_fig = None
        self.catchment_ax = None
        self.catchment_im = None
        self.proj = None
        self.ext = None

    def run(self):
        """
        Run the complete SNODAS processing pipeline.
        """
        self.setup_data()
        self.process_raw()
        self.process_catchment()
    
    def setup_data(self):
        """
        Load and prepare all required data for processing.
        """

        self.s3_mount_point = os.getenv('S3_MOUNT_POINT', os.path.join(os.path.expanduser("~"), 's3'))
        self.snotel_s3_path = 'ngwpc-forcing/snotel_csv'
        self.snodas_file = DataLoader.snodas_path_constructor(self.date, self.s3_mount_point, self.direct_s3)
        self.basin_gdf = DataLoader.read_geo(self.gpkg_file)
        self.basin_geometry, self.bounds = DataLoader.get_basin_geometry(self.basin_gdf)
        self.snodas_ds = DataLoader.load_netcdf(self.snodas_file)
        
        # For SNOTEL data
        self.snotel_filenames, self.snotel_filesystem = DataLoader.list_snotel_filenames(self.s3_mount_point, 
                                                                                         self.snotel_s3_path,
                                                                                         self.direct_s3)
        self.stations_gdf = DataLoader.parse_snotel_filenames(self.snotel_filenames)
        self.stations_in_basin = Calculator.find_stations_in_basin(self.stations_gdf, self.basin_geometry)

        # Load SNOTEL data if stations exist in basin
        if not self.stations_in_basin.empty:
            self.snotel_data = DataLoader.load_snotel_data(self.stations_in_basin, 
                                                           self.date, 
                                                           self.snotel_filesystem, 
                                                           self.s3_mount_point,
                                                           self.snotel_s3_path)        
    
    def process_raw(self):
        """
        Process and plot raw SNODAS data.
        """
        t3 = time.time()
        
        # Create base plot
        self.raw_fig, self.raw_ax, self.proj = Plotter.create_base_plot()
        self.ext = Plotter.set_map_extent(self.raw_ax, self.bounds, self.proj)
        self.raw_ax, self.raw_im, vmin, vmax = Plotter.plot_raw_swe(self.raw_ax, 
                                                                    self.snodas_ds, 
                                                                    self.basin_geometry, 
                                                                    self.basin_gdf, 
                                                                    self.proj)
        
        self.raw_ax = Plotter.add_basin_overlay(self.raw_ax, self.basin_geometry, self.proj)        
        cbar = Plotter.add_colorbar(self.raw_im, self.raw_ax)
        plt.title(f'Raw SNODAS Snow Water Equivalent\n {self.date} - 06z')
        gl = Plotter.add_gridlines(self.raw_ax)
        
        # Add SNOTEL data overlay if available
        if self.stations_in_basin is not None and not self.stations_in_basin.empty and self.snotel_data is not None:
            self.raw_ax = Plotter.add_snotel_overlay(self.raw_ax, self.snotel_data, self.proj)
        
        # Save figure if output file is specified
        if self.output_file_raw:
            t4 = time.time()
            Plotter.save_figure(self.raw_fig, self.output_file_raw)
            print(f"   Raw output time: {time.time() - t4:.2f}s")
        
        print(f"   Raw plotting time: {time.time() - t3:.2f}s")
    
    def process_catchment(self):
        """
        Process and plot catchment-averaged SNODAS data.
        """
        t5 = time.time()
        basin_gdf_with_swe, ds_catchment = Calculator.calculate_catchment_mean(self.snodas_ds, 
                                                                               self.basin_geometry, 
                                                                               self.basin_gdf)
        
        self.catchment_fig, self.catchment_ax, self.proj = Plotter.create_base_plot()
        self.ext = Plotter.set_map_extent(self.catchment_ax, self.bounds, self.proj)
        self.catchment_ax, self.catchment_im, vmin, vmax = Plotter.plot_polygon_swe(self.catchment_ax, 
                                                                                    basin_gdf_with_swe, 
                                                                                    self.proj)
        self.catchment_ax = Plotter.add_basin_overlay(self.catchment_ax, self.basin_geometry, self.proj)
        cbar = Plotter.add_colorbar(self.catchment_im, self.catchment_ax)
        plt.title(f'Lumped SNODAS Snow Water Equivalent\n {self.date} - 06z')
        gl = Plotter.add_gridlines(self.catchment_ax)

        # Add SNOTEL data overlay if available
        if self.stations_in_basin is not None and not self.stations_in_basin.empty and self.snotel_data is not None:
            self.catchment_ax = Plotter.add_snotel_overlay(self.catchment_ax, self.snotel_data, self.proj)
        
        # Save figure if output file is specified
        if self.output_file_lumped:
            t6 = time.time()
            Plotter.save_figure(self.catchment_fig, self.output_file_lumped)
            print(f"   Lumped output time: {time.time() - t6:.2f}s")
        
        print(f"   Lumped plotting time: {time.time() - t5:.2f}s")


def get_options(args_list=None):
    """
    Parse command-line arguments.
    
    Parameters
    ----------
    args_list : list, optional
        List of arguments to parse (defaults to command line arguments)
        
    Returns
    -------
    argparse.Namespace
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('date', type=str,
                        help="Date of SNODAS data to map.")
    parser.add_argument('gpkg_file', type=str,
                        help="Path to geopackage file.")
    parser.add_argument('output_file_raw', type=str,
                        help="Path where raw visualization output is saved.")
    parser.add_argument('output_file_lumped', type=str,
                        help="Path where catchment-averaged output is saved.")
    parser.add_argument('--direct_s3', action='store_true', 
                        help='Use direct S3 access instead of local mount')
    args_list = list(args_list)
    return parser.parse_args(args_list)


def main(args_list=None):
    """
    Main function to run the SNODAS processor.
    
    Parameters
    ----------
    args_list : list, optional
        List of arguments to parse (defaults to command line arguments)
    """
    args = get_options(args_list)
    
    # Create, then run, a processor instance
    processor = SNODASProcessor(
        date=args.date,
        gpkg_file=args.gpkg_file,
        output_file_raw=args.output_file_raw,
        output_file_lumped=args.output_file_lumped,
        direct_s3=args.direct_s3
    )

    processor.run()


if __name__ == "__main__":
    main()
