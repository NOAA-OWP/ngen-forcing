import os
from datetime import datetime

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class NBMHawaiiDownloader(ForecastDownloader):
    """Downloader for NBM forecast data over Hawaii.

    - Files live in: blend.YYYYMMDD/HH/core/
    - File names follow the format: blend.tCCz.core.fXXX.hi.grib2
    - This implementation constructs URLs directly (no scraping).
    """

    @property
    def base_url(self):
        """Base URL for NBM Hawaii data."""
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com/"

    def should_process_hour(self, d_start: datetime) -> bool:
        """Only process hours 0, 6, 12, and 18."""
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, d_start: datetime) -> list:
        """Return the list of forecast hours to download based on the start hour."""
        hourly = range(1, 37)  # 1 through 36
        every_3h = range(36, 193, 3)  # 123 through 240, step of 3
        every_6h = range(198, 265, 6)  # 198 through 264, step of 6
        return (
            list(hourly) + list(every_3h) + list(every_6h)
            if d_start.hour in [0, 6, 12, 18]
            else []
        )

    def build_output_dir(self, d_start: datetime, _) -> str:
        """Construct the output directory path based on the start date."""
        return os.path.join(
            self.out_dir,
            f"blend.{d_start.strftime('%Y%m%d')}",
            d_start.strftime("%H"),
            "core",
        )

    def build_file_url_and_name(
        self, d_start: datetime, target: int, _
    ) -> tuple[str, str]:
        """Construct the download URL and filename for a given forecast hour."""
        fhr_str = f"f{str(target).zfill(3)}"
        filename = f"blend.t{d_start.strftime('%H')}z.core.{fhr_str}.hi.grib2"
        url = os.path.join(
            self.base_url,
            f"blend.{d_start.strftime('%Y%m%d')}",
            d_start.strftime("%H"),
            "core",
            filename,
        )
        return url, filename

    @property
    def recursive_cleanup(self) -> bool:
        """Whether to perform recursive cleanup of the output directory."""
        return True


if __name__ == "__main__":
    downloader = NBMHawaiiDownloader.from_cli_args()
    downloader.run()
