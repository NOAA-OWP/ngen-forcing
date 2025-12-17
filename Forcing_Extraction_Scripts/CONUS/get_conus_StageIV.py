import os
from abc import ABC

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class StageIVDownloader(ForecastDownloader, ABC):
    """
    Downloader for CONUS Stage IV hourly precipitation analysis.

    - Files are organized by date: pcpanl.YYYYMMDD/
    - File names: st4_conus.YYYYMMDDHH.01h.grb2
    - We download one file per hour.
    - Local output is flattened (no pcpanl subfolder used locally).
    """

    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def should_process_hour(self, _):
        # Process every hour (hourly product)
        return True

    def get_download_targets(self, _):
        # Stage IV has a single file per hour — just return a placeholder
        return [None]

    def build_output_dir(self, _, __):
        # Store all files directly in the output directory (flat structure)
        return self.out_dir

    def build_file_url_and_name(self, d_start, _, __):
        subdir = f"pcpanl.{d_start.strftime('%Y%m%d')}"
        filename = f"st4_conus.{d_start.strftime('%Y%m%d%H')}.01h.grb2"
        url = os.path.join(self.base_url, subdir, filename)
        return url, filename


if __name__ == "__main__":
    downloader = StageIVDownloader.from_cli_args()
    downloader.run()
