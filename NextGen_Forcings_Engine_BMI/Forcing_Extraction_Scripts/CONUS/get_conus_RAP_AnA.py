import os
from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

class RAPAnADownloader(ForecastDownloader):
    """Downloader for CONUS RAP AnA (Analysis) data."""

    @property
    def base_url(self):
        return "https://noaa-rap-pds.s3.amazonaws.com"

    def get_download_targets(self, _):
        return [1]

    def build_output_dir(self, d_start, _):
        return os.path.join(self.out_dir, f"rap.{d_start.strftime('%Y%m%d')}")
