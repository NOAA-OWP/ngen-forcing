import matplotlib.pyplot as plt
import cartopy.crs as ccrs

class PlotUtils:
    @staticmethod
    def create_base_plot():
        """
        Create a base map plot with cartopy projection.
        
        Returns
        -------
        tuple
            (fig, ax, proj) where:
                - fig is the matplotlib Figure object
                - ax is the GeoAxes object
                - proj is the cartopy projection (PlateCarree)
        """
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})
        return fig, ax, proj

    @staticmethod
    def set_map_extent(ax, bounds, proj):
        """
        Set the map extent with appropriate buffers around bounds.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to set extent for
        bounds : tuple
            Tuple of (minx, miny, maxx, maxy) for map extent
        proj : cartopy.crs
            Projection to use for extent
            
        Returns
        -------
        list
            List of [minx, maxx, miny, maxy] with added buffers
        """
        # Set the extent using dynamic vertical and horizontal buffers
        buff_v = abs(bounds[2] - bounds[0]) * .01
        buff_h = abs(bounds[3] - bounds[1]) * .01
        ext = [
            bounds[0] - buff_v,
            bounds[2] + buff_v,
            bounds[1] - buff_h,
            bounds[3] + buff_h
        ]
        ax.set_extent(ext, crs=proj)
        return ext

    @staticmethod
    def plot_catchment_boundaries(ax, gdf, proj):
        """
        Add catchment boundaries to a map plot.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to add boundaries to
        gdf : geopandas.GeoDataFrame
            GeoDataFrame containing catchment polygons
        proj : cartopy.crs
            Projection to use for geometries
            
        Returns
        -------
        matplotlib.axes.Axes
            Updated axes with catchment boundaries
        """
        # Iterate over polygons in the dataframe, drawing boundaries
        for _, row in gdf.iterrows():
            ax.add_geometries([row.geometry], crs=proj,
                             facecolor='none',
                             edgecolor='black',
                             linewidth=0.5,
                             alpha=0.5)
        return ax

    @staticmethod
    def add_basin_overlay(ax, basin_geometry, proj):
        """
        Add the basin outline to a map plot.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to add basin outline to
        basin_geometry : shapely.geometry
            Basin geometry to add as outline
        proj : cartopy.crs
            Projection to use for geometry
            
        Returns
        -------
        matplotlib.axes.Axes
            Updated axes with basin outline
        """
        # Overlay basin outline
        ax.add_geometries([basin_geometry], crs=proj,
                          facecolor='none', edgecolor='red',
                          linewidth=1.5)
        return ax

    @staticmethod
    def add_gridlines(ax):
        """
        Add gridlines to a map plot.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to add gridlines to
            
        Returns
        -------
        cartopy.mpl.gridliner.Gridliner
            Gridliner object for the added gridlines
        """
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                          color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        return gl

    @staticmethod
    def add_colorbar(im, ax):
        """
        Add a colorbar to a map plot.
        
        Parameters
        ----------
        im : matplotlib.cm.ScalarMappable
            Image or mappable object to create colorbar for
        ax : matplotlib.axes.Axes
            Axes object to add colorbar to
            
        Returns
        -------
        matplotlib.colorbar.Colorbar
            Colorbar object
        """
        # Plot colorbar based on settings in plot functions
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(f'Snow Water Equivalent (m)', fontsize=10)
        return cbar

    @staticmethod
    def save_figure(fig, output_file):
        """
        Save a figure to a file.
        
        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        output_file : str
            Path where the figure should be saved
            
        Returns
        -------
        str
            Path where the figure was saved
        """
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_file