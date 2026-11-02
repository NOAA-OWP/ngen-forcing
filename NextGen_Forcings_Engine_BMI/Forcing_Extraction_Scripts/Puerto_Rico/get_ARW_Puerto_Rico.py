import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class ARWPuertoRicoDownloader(ForecastDownloader):
    """Downloader for WRF-ARW 2.5 km Puerto Rico forecasts."""

    @property
    def base_url(self):
        return "https://ftp.ncep.noaa.gov/data/nccf/com/hiresw/prod"

    def should_process_hour(self, d_current):
        return d_current.hour in [6, 18]

    def get_download_targets(self, d_current):
        return range(1, 49)
