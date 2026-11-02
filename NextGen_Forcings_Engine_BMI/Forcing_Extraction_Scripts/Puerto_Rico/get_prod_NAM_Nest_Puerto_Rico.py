import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class NAMNestPuertoRicoDownloader(ForecastDownloader):
    """Downloader for NAM Nest Puerto Rico 3-km forecasts."""

    @property
    def base_url(self):
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, _):
        return range(1, 61)
