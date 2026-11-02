import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class HRRRSubhourlyDownloader(ForecastDownloader):
    """Downloader for CONUS HRRR sub-hourly data (15-minute forecasts)."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_current):
        return True

    def get_download_targets(self, d_current):
        return range(0, 19)
