import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class HRRRAnAConusDownloader(ForecastDownloader):
    """Downloader for CONUS HRRR Analysis (AnA) surface data."""
    default_lookback = 30
    default_cleanback = 240
    default_lagback = 1

    @property
    def base_url(self):
        return "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return [1]
