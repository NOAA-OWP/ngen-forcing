import os
import geopandas as gpd

class GeoUtils:
    @staticmethod
    def read_geo(gpkg_file):
        """
        Read the 'divides' layer from a geopackage file and ensure geographic CRS.
        
        Parameters
        ----------
        gpkg_file : str
            Path to the geopackage file
            
        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing the basin divides with geographic CRS
        """
        if not os.path.exists(gpkg_file):
            raise FileNotFoundError(f"Geopackage file '{gpkg_file}' not found. Please check the file path.")

        basin_gdf = gpd.read_file(gpkg_file, layer='divides')

        if basin_gdf.crs is None or not basin_gdf.crs.is_geographic:
            basin_gdf = basin_gdf.to_crs('EPSG:4326')
            
        return basin_gdf

    @staticmethod
    def get_basin_geometry(basin_gdf):
        """
        Extract a unified basin geometry and bounds from a GeoDataFrame.
        
        Parameters
        ----------
        basin_gdf : geopandas.GeoDataFrame
            GeoDataFrame containing basin divides
            
        Returns
        -------
        tuple
            (basin_geometry, bounds) where:
                - basin_geometry is a shapely geometry representing the entire basin
                - bounds is a tuple of (minx, miny, maxx, maxy) for the basin extent
        """
        # Combine all polygons for basin outline
        try:
            basin_geometry = basin_gdf.union_all()
        except AttributeError:
            basin_geometry = basin_gdf.unary_union
            
        # Store catchment boundaries
        bounds = basin_geometry.bounds

        return basin_geometry, bounds