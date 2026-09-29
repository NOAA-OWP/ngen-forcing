import os

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class NBMConusDownloader(ForecastDownloader):
    """
    Downloader for CONUS NBM forecast data on the Gaussian grid in GRIB2 format.

    - Files are deterministic: blend.t00z.core.fXXX.co.grib2
    - Located in: blend.YYYYMMDD/HH/core/
    - Forecast hours from f001 to f264.
    """

    @property
    def base_url(self):
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com/"

    def should_process_hour(self, d_start):
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, d_start):
        hourly = range(1, 37)  # 1 through 36
        every_3h = range(36, 193, 3)  # 123 through 240, step of 3
        every_6h = range(198, 265, 6)  # 198 through 264, step of 6
        return list(hourly) + list(every_3h) + list(every_6h) if d_start.hour in [0, 6, 12, 18] else []
        # return range(1, 265) if d_start.hour in [0, 6, 12, 18] else []

    def build_output_dir(self, d_start, _):
        return os.path.join(
            self.out_dir,
            f"blend.{d_start.strftime('%Y%m%d')}",
            d_start.strftime('%H'),
            "core"
        )

    def build_file_url_and_name(self, d_start, target, _):
        fhr_str = f"f{str(target).zfill(3)}"
        filename = f"blend.t{d_start.strftime('%H')}z.core.{fhr_str}.co.grib2"
        url = os.path.join(
            self.base_url,
            f"blend.{d_start.strftime('%Y%m%d')}",
            d_start.strftime('%H'),
            "core",
            filename,
        )
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        return True


if __name__ == "__main__":
    downloader = NBMConusDownloader.from_cli_args()
    downloader.run()
