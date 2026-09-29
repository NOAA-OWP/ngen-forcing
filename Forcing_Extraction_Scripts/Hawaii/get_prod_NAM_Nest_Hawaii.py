import os
from datetime import datetime

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader


class NAMNestHawaiiDownloader(ForecastDownloader):
    """Downloader for NAM Nest Hawaii 3-km forecast data.

    - Forecasts available every 6 hours: 00Z, 06Z, 12Z, 18Z
    - Each forecast has 60 hourly files.
    """

    @property
    def base_url(self) -> str:
        """Base URL for NAM Nest Hawaii data."""
        return "https://noaa-nam-pds.s3.amazonaws.com"

    def should_process_hour(self, d_start: datetime) -> bool:
        """Only process forecasts initialized at 00Z, 06Z, 12Z, and 18Z."""
        return d_start.hour in [0, 6, 12, 18]

    def get_download_targets(self, d_start: datetime) -> range:
        """Get list of forecast hours to download for a given initialization time."""
        return (
            range(1, self.input_horizon + 1) if d_start.hour in [0, 6, 12, 18] else []
        )

    def build_output_dir(self, d_start: datetime, _) -> str:
        """Build output directory path based on initialization date."""
        return os.path.join(self.out_dir, f"nam.{d_start.strftime('%Y%m%d')}")

    def build_file_url_and_name(
        self, d_start: datetime, target: int, _
    ) -> tuple[str, str]:
        """Construct the download URL and filename for a given initialization time and forecast hour."""
        fhr = str(target).zfill(2)
        filename = f"nam.t{d_start.strftime('%H')}z.hawaiinest.hiresf{fhr}.tm00.grib2"
        url = os.path.join(self.base_url, f"nam.{d_start.strftime('%Y%m%d')}", filename)
        return url, filename


if __name__ == "__main__":
    downloader = NAMNestHawaiiDownloader.from_cli_args()
    downloader.run()
