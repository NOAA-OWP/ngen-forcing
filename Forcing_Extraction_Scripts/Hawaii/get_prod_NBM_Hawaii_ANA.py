import os
from datetime import datetime

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class NBMAnAHawaiiDownloader(ForecastDownloader):
    """Downloader for NBM forecast data over Hawaii using predictable filenames.

    - Files are stored in: blend.YYYYMMDD/HH/core/
    - File pattern: blend.t{HH}z.core.f{forecast_hour}.hi.grib2
    - Forecast hours range from f001 to f264.
    """

    @property
    def base_url(self):
        """Sets the base url for NBM Hawaii AnA data."""
        return "https://noaa-nbm-grib2-pds.s3.amazonaws.com"

    def get_download_targets(self, _) -> list[int]:
        """Set the forecast hours to download."""
        return list(range(1, 2))

    def build_output_dir(self, d_start: datetime, _) -> str:
        """Create the output directory path."""
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
        """Indicates whether cleanup should be recursive."""
        return True


if __name__ == "__main__":
    downloader = NBMAnAHawaiiDownloader.from_cli_args()
    downloader.run()
