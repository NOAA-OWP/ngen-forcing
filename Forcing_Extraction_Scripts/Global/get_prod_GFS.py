import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class GFSDownloader(ForecastDownloader):
    """
    Downloader for GFS operational forecast data.

    - Available at 00Z, 06Z, 12Z, 18Z only.
    - Downloads sfluxgrbfNN.grib2 files out to 240h (expandable).
    - Files are organized by: gfs.YYYYMMDD/HH/atmos/
    """

    default_lookback = 8
    default_cleanback = 240
    default_lagback = 4

    @property
    def base_url(self):
        return "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, _):
        hourly = range(1, 121)  # 1 through 120
        every_3h = range(123, 241, 3)  # 123 through 240, step of 3
        target_hours = list(hourly) + list(every_3h)

        # Filter download targets by input_horizon if specific
        if self.input_horizon is not None:
            target_hours = [hour for hour in target_hours if hour <= self.input_horizon]

        return target_hours

    def build_output_dir(self, d_start, _):
        return os.path.join(
            self.out_dir,
            f"gfs.{d_start.strftime('%Y%m%d')}",
            d_start.strftime("%H"),
            "atmos",
        )

    def build_file_url_and_name(self, d_start, forecast_hour, _):
        fhr = str(forecast_hour).zfill(3)
        filename = f"gfs.t{d_start.strftime('%H')}z.sfluxgrbf{fhr}.grib2"
        url = os.path.join(
            self.base_url,
            f"gfs.{d_start.strftime('%Y%m%d')}",
            d_start.strftime("%H"),
            "atmos",
            filename,
        )
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True


if __name__ == "__main__":
    downloader = GFSDownloader.from_cli_args()
    downloader.run()
