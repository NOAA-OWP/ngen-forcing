import os
import time
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import geopandas as gpd
import pandas as pd
import argparse
import fsspec
from ..utility.swe_minmax import get_minmax
from ..utility.snotel_utils import SnotelDataLoader, SnotelCalculator, SnotelPlotter
from ..utility.geo_utils import GeoUtils
from ..utility.plot_utils import PlotUtils

class DataLoader:
    @staticmethod
    def load_netcdf(netcdf_file):
        t0 = time.time()
        sim_ds = xr.open_dataset(netcdf_file)
        print(f"   NetCDF load time: {time.time() - t0:.2f}s")

        return sim_ds

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
    def read_geo(gpkg_file):
        """Read divides layer from .gpkg file to GeoDataFrame"""
        return GeoUtils.read_geo(gpkg_file)


class Calculator:
    @staticmethod
    def process_data(sim_ds, date_str, basin_gdf):
        """Load and process SWE data from NetCDF and geopackage files
        
        Args:
            netcdf_file: Path to NetCDF file
            gpkg_file: Path to geopackage file
            date_str: Date string from NetCDF time dim (ex: '2015-12-01')
        """

        swe_data = sim_ds.swe.sel(date=date_str).values

        # Create a mapping dictionary from catchment IDs to SWE values
        catchment_ids = sim_ds.catchment.values
        swe_dict = dict(zip(catchment_ids, swe_data))
        
        # Create catchment ID column and then lookup values from dict
        basin_gdf['catchment_id'] = basin_gdf['divide_id'].str.split('-').str[1].astype(int)
        basin_gdf['mean_swe'] = basin_gdf['catchment_id'].map(swe_dict).fillna(np.nan)
        # print(f"   SWE load/process time: {time.time() - t2:.2f}s")

        return basin_gdf
    
    @staticmethod
    def get_basin_geometry(basin_gdf):
        """
        Extract a unified basin geometry and bounds from a GeoDataFrame.
        """
        return GeoUtils.get_basin_geometry(basin_gdf)

    @staticmethod
    def find_stations_in_basin(stations_gdf, basin_geometry):
        """
        Find SNOTEL stations that fall within the basin geometry.
        """
        return SnotelCalculator.find_stations_in_basin(stations_gdf, basin_geometry)


class Plotter:
    @staticmethod
    def create_base_plot():
        """
        Create a base map plot with cartopy projection.
        """
        return PlotUtils.create_base_plot()

    @staticmethod
    def plot_catchment_boundaries(ax, gdf, proj):
        """Add catchment boundaries to plot"""
        return PlotUtils.plot_catchment_boundaries(ax, gdf, proj)

    @staticmethod
    def plot_polygon_simulated_swe(ax, gdf, proj):
        """Plot catchments filled with their simulated (lumped) SWE values"""
        
        # Set color scale based on min/max values
        #vmin = float(gdf['mean_swe'].min())
        vmin,vmax = get_minmax(gdf['mean_swe'])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.Blues
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        
        for _, row in gdf.iterrows():
            if not np.isnan(row['mean_swe']):
                ax.add_geometries([row.geometry], crs=proj,
                                facecolor=cmap(norm(row['mean_swe'])),
                                edgecolor='none')
        
        ax = Plotter.plot_catchment_boundaries(ax, gdf, proj)
        return ax, sm, vmin, vmax

    @staticmethod
    def set_map_extent(ax, bounds, proj):
        """
        Set the map extent with appropriate buffers around bounds.
        """
        return PlotUtils.set_map_extent(ax, bounds, proj)

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
    def plot_swe_map(netcdf_file, gpkg_file, date_str, output_file, mode='plot'):
        """Creates a map of simulated SWE values by catchment"""
        
        ds, gdf, basin_geometry, bounds = load_and_process_data(netcdf_file,
                                                                gpkg_file,
                                                                date_str)
        # t1 = time.time()
        
        # Call plot function
        ax, im, vmin, vmax = plot_polygon_simulated_swe(ax, gdf, proj)
        
        # Title bar with date
 
    @staticmethod
    def add_snotel_overlay(ax, snotel_data, proj):
        """
        Add SNOTEL SWE data as text overlays on a map.
        """
        return SnotelPlotter.add_snotel_overlay(ax, snotel_data, proj)

    @staticmethod
    def save_figure(fig, output_file):
        """
        Save a figure to a file.
        """
        return PlotUtils.save_figure(fig, output_file)


class SimSWEProcessor:
    def __init__(self, netcdf_file = None, gpkg_file = None, date=None, output_file=None, mode=None, direct_s3=False):
        
        # Initialize input parameters
        self.netcdf_file = netcdf_file
        self.gpkg_file = gpkg_file
        self.date = date
        self.output_file = output_file
        self.mode = mode
        self.direct_s3 = direct_s3
        
        # Initialize data attributes        
        self.basin_gdf = None
        self.swe_gdf = None
        self.bounds = None
        self.basin_geometry = None
        self.sim_ds = None

        # Initialize plot attributes
        self.vmin = None
        self.vmax = None
        self.sim_fig = None
        self.sim_ax = None
        self.sim_im = None
        self.proj = None
        self.ext = None

        # Initialize SNOTEL-related attributes
        self.snotel_filenames = None
        self.stations_gdf = None
        self.stations_in_basin = None
        self.snotel_data = None
        self.snotel_filesystem = None
        self.snotel_s3_path = None
        self.s3_mount_point = None

    def run(self):
        self.setup_data()
        self.process_data()
        self.plot_swe()

    def setup_data(self):
        self.snotel_s3_path = 'ngwpc-forcing/snotel_csv'
        self.s3_mount_point = os.getenv('S3_MOUNT_POINT', os.path.join(os.path.expanduser("~"), 's3'))
        self.basin_gdf = DataLoader.read_geo(self.gpkg_file)
        self.sim_ds = DataLoader.load_netcdf(self.netcdf_file)
        self.basin_geometry, self.bounds = Calculator.get_basin_geometry(self.basin_gdf)        
        
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

    def process_data(self):
        self.swe_gdf = Calculator.process_data( self.sim_ds, self.date, self.basin_gdf)

    def plot_swe(self):
        if self.mode == 'scan':
            return get_minmax(self.swe_gdf['mean_swe'])
        self.sim_fig, self.sim_ax, self.proj = Plotter.create_base_plot()
        self.ext = Plotter.set_map_extent(self.sim_ax, self.bounds, self.proj)
        self.sim_ax, self.sim_im, self.vmin, self.vmax = Plotter.plot_polygon_simulated_swe(self.sim_ax,
                                                                                            self.swe_gdf,
                                                                                            self.proj)
        self.sim_ax = Plotter.add_basin_overlay(self.sim_ax, self.basin_geometry, self.proj)
        cbar = Plotter.add_colorbar(self.sim_im, self.sim_ax)
        plt.title(f'Simulated Snow Water Equivalent (SWE)\n {self.date} - 06z')
        gl = Plotter.add_gridlines(self.sim_ax)
        # Add SNOTEL data overlay if available
        if self.stations_in_basin is not None and not self.stations_in_basin.empty and self.snotel_data is not None:
            self.sim_ax = Plotter.add_snotel_overlay(self.sim_ax, self.snotel_data, self.proj)
        if self.output_file is not None:
            Plotter.save_figure(self.sim_fig, self.output_file)

def get_options(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('netcdf_file', type=str,
                       help="Path to NetCDF file")
    parser.add_argument('gpkg_file', type=str,
                       help="Path to geopackage file")
    parser.add_argument('date', type=str,
                       help="Date to plot (ex: '2015-12-01')")
    parser.add_argument('--output_file', type=str, default=None,
                       help="Path where output image is saved")
    parser.add_argument('--mode', type=str, default='plot',
                       choices=['plot', 'scan'],
                       help="Operation mode: 'plot' or 'scan'")
    parser.add_argument('--direct_s3', action='store_true', 
                        help='Use direct S3 access instead of local mount')
    return parser.parse_args(args_list)

def main(args_list=None):
    args = get_options(args_list)
    processor = SimSWEProcessor(
        netcdf_file = args.netcdf_file, 
        gpkg_file = args.gpkg_file, 
        date = args.date, 
        output_file = args.output_file, 
        mode = args.mode,
        direct_s3 = args.direct_s3
        )
    processor.run()

if __name__ == "__main__":
    main()
