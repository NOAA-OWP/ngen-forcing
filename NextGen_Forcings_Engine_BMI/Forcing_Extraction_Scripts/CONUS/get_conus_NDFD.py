"""
National Digital Forecast Database (NDFD) CONUS operational forcing downloader.
Extracts and concatenates operational GRIB2 bins for temperature, wind, and precipitation.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import uuid

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

logger = logging.getLogger(__name__)

PRIMARY_PERIOD = "VP.001-003"
EXTENDED_PERIOD = "VP.004-007"


class NDFDConusDownloader(ForecastDownloader):
    """Downloader implementation for operational CONUS NDFD forecasts."""

    #: Engine field tag (as read by find_ndfd_neighbors) -> operational bin file.
    FIELD_MAP: dict[str, str] = {
        "tmp": "ds.temp.bin",
        "wdir": "ds.wdir.bin",
        "wspd": "ds.wspd.bin",
        "qpf": "ds.qpf.bin",
    }

    # QPF is only produced operationally through 72 hours (VP.001-003).
    # Instantaneous elements (TMP, WDIR, WSPD) extend through Day 7 (VP.004-007).
    FIELD_VALID_PERIODS: dict[str, list[str]] = {
        "tmp": [PRIMARY_PERIOD, EXTENDED_PERIOD],
        "wdir": [PRIMARY_PERIOD, EXTENDED_PERIOD],
        "wspd": [PRIMARY_PERIOD, EXTENDED_PERIOD],
        "qpf": [PRIMARY_PERIOD],
    }

    @property
    def base_url(self) -> str:
        return "https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus"

    def should_process_hour(self, d_current: datetime.datetime) -> bool:
        """Process 6-hourly operational cycles (00, 06, 12, 18 UTC)."""
        return d_current.hour in (0, 6, 12, 18)

    def get_download_targets(self, d_current: datetime.datetime) -> list[str]:
        """Return the field tags expected by the engine."""
        return list(self.FIELD_MAP)

    def build_output_dir(self, d_current: datetime.datetime, _=None) -> str:
        """Construct <inDir>/NDFD/YYYYMMDD/wgrbbul/ndfd_conus."""
        date_str = d_current.strftime("%Y%m%d")
        return os.path.join(self.out_dir, "NDFD", date_str, "wgrbbul", "ndfd_conus")

    def build_file_url_and_name(
        self,
        d_current: datetime.datetime,
        target: str,
        _=None,
    ) -> tuple[str, str]:
        """Return the primary-period URL (used for availability checks) and the local filename."""
        url = f"{self.base_url}/{PRIMARY_PERIOD}/{self.FIELD_MAP[target]}"
        filename = f"ndfd_conus_{target}.grib2"
        return url, filename

    def iter_remote_files(self):
        """Yield every valid-period part; the local file is their concatenation."""
        for d_current, primary_url, local_path in super().iter_remote_files():
            field = os.path.basename(local_path)[len("ndfd_conus_"):-len(".grib2")]
            for vp in self.FIELD_VALID_PERIODS[field]:
                yield d_current, primary_url.replace(PRIMARY_PERIOD, vp), local_path

    def _download_file(self, url: str, out_path: str) -> bool:
        """Download every valid period of one field and publish their concatenation, or nothing."""
        field = os.path.basename(out_path)[len("ndfd_conus_"):-len(".grib2")]
        bin_file = self.FIELD_MAP[field]
        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)
        token = uuid.uuid4()
        parts: list[str] = []
        temp_out = os.path.join(out_dir, f".{os.path.basename(out_path)}.tmp.{token}")
        try:
            for vp in self.FIELD_VALID_PERIODS[field]:
                vp_url = f"{self.base_url}/{vp}/{bin_file}"
                part = os.path.join(out_dir, f".{os.path.basename(out_path)}.{vp}.{token}")
                parts.append(part)
                self._fetch(vp_url, part)
            with open(temp_out, "wb") as outfile:
                for part in parts:
                    with open(part, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
            self._publish(temp_out, out_path)
        finally:
            for path in [*parts, temp_out]:
                if os.path.exists(path):
                    os.remove(path)
        logger.info("Downloaded %s (%s)", out_path, ", ".join(self.FIELD_VALID_PERIODS[field]))
        return True


def main() -> None:
    """CLI entrypoint for standalone NDFD CONUS forcing extraction."""
    NDFDConusDownloader.cli_main(desc="Download operational NDFD CONUS forecast forcings")


if __name__ == "__main__":
    main()
