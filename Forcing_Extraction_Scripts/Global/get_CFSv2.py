import os
from datetime import timedelta

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class CFSv2Downloader(ForecastDownloader):
    """
    Downloader for CFSv2 forecast data (6-hour interval outputs for 30 days).

    - Data is issued at 00Z, 06Z, 12Z, 18Z.
    - Each cycle produces 6-hour forecasts out to ~7.5 days (up to 60h shown here).
    - Files live in: cfs.YYYYMMDD/HH/6hrly_grib_01/
    - Filenames follow format: flxf<valid_time>.01.<init_time>.grb2
    """

    default_lookback = 24
    default_cleanback = 720
    default_lagback = 6

    @property
    def base_url(self):
        return "https://noaa-cfs-pds.s3.amazonaws.com"

    def get_download_targets(self, d_start):
        return range(0, 721, 6) if d_start.hour in [0, 6, 12, 18] else []

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def build_output_dir(self, d_start, ens_number):
        return os.path.join(
            self.out_dir,
            f"cfs.{d_start.strftime('%Y%m%d')}",
            d_start.strftime('%H'),
            f"6hrly_grib_{ens_number}"
        )

    def build_file_url_and_name(self, d_start, fhr, ens_number):
        # Target file has valid_time (forecast) and init_time (cycle) in name
        valid_time = d_start + timedelta(hours=fhr)
        init_time = d_start.strftime('%Y%m%d%H')
        valid_time_str = valid_time.strftime('%Y%m%d%H')
        filename = f"flxf{valid_time_str}.{ens_number}.{init_time}.grb2"
        url = os.path.join(
            self.base_url,
            f"cfs.{d_start.strftime('%Y%m%d')}",
            d_start.strftime('%H'),
            f"6hrly_grib_{ens_number}",
            filename
        )
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True


if __name__ == "__main__":
    downloader = CFSv2Downloader.from_cli_args()
    downloader.run()
