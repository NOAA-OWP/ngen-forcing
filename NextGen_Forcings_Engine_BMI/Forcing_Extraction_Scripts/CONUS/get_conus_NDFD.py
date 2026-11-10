"""
National Digital Forecast Database (NDFD) CONUS operational forcing downloader.
Extracts and concatenates operational GRIB2 bins for temperature, wind, and precipitation.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
import typing
import requests

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader

logger = logging.getLogger(__name__)


class NDFDConusDownloader(ForecastDownloader):
    """Downloader implementation for operational CONUS NDFD forecasts."""

    FIELD_MAP: dict[str, str] = {
        "TMP": "ds.temp.bin",
        "WDIR": "ds.wdir.bin",
        "WSPD": "ds.wspd.bin",
        "APCP": "ds.qpf.bin",
    }

    # APCP (QPF) is only produced operationally through 72 hours (VP.001-003).
    # Instantaneous elements (TMP, WDIR, WSPD) extend through Day 7 (VP.004-007).
    FIELD_VALID_PERIODS: dict[str, list[str]] = {
        "TMP": ["VP.001-003", "VP.004-007"],
        "WDIR": ["VP.001-003", "VP.004-007"],
        "WSPD": ["VP.001-003", "VP.004-007"],
        "APCP": ["VP.001-003"],
    }

    @property
    def base_url(self) -> str:
        return "https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus"

    def should_process_hour(self, d_current: datetime.datetime) -> bool:
        """Process 6-hourly operational cycles (00, 06, 12, 18 UTC)."""
        return d_current.hour in (0, 6, 12, 18)

    def get_valid_cycles(self, cycle_time: datetime.datetime) -> list[datetime.datetime]:
        """NDFD operates continuously; cycles align to 6-hour NextGen steps."""
        return [cycle_time]

    def get_download_targets(self, d_current: datetime.datetime) -> list[str]:
        """Return the target variable names expected by NextGen."""
        return list(self.FIELD_MAP.keys())

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
        """Return primary VP.001-003 URL for availability ping and local filename."""
        bin_file = self.FIELD_MAP.get(target, f"ds.{target.lower()}.bin")
        url = f"{self.base_url}/VP.001-003/{bin_file}"
        filename = f"ndfd_conus_{target}.grib2"
        return url, filename

    def _execute_multi_vp_download(self, url: str, out_file: str, attempt_limit: int = 10, interval: int = 10) -> bool:
        """Download and concatenate GRIB2 records across supported valid periods."""
        base_name = os.path.basename(out_file)
        field = base_name.replace("ndfd_conus_", "").replace(".grib2", "")
        bin_file = self.FIELD_MAP.get(field)

        valid_periods = self.FIELD_VALID_PERIODS.get(field, ["VP.001-003"])
        if field == "APCP":
            logger.info("NDFD QPF (APCP) is capped at 72 hours (VP.001-003) by operational design; omitting extended periods.")

        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        temp_out = out_path.with_name(f".{out_path.name}.tmp")

        headers = {"User-Agent": "Mozilla/5.0 (NextGen-Forcing-Engine)"}
        base_success = False

        with open(temp_out, "wb") as outfile:
            for vp in valid_periods:
                vp_url = f"{self.base_url}/{vp}/{bin_file}"
                logger.info(f"Attempting download of {vp}/{bin_file} -> {out_file}")
                try:
                    resp = requests.get(vp_url, headers=headers, stream=True, timeout=30)
                    if resp.status_code == 200:
                        for chunk in resp.iter_content(chunk_size=1048576):
                            if chunk:
                                outfile.write(chunk)
                        logger.info(f"Appended {vp}/{bin_file} to {out_file}")
                        if vp == "VP.001-003":
                            base_success = True
                    else:
                        logger.warning(f"Unexpected HTTP {resp.status_code} for {vp_url}")
                except Exception as exc:
                    logger.warning(f"Failed to retrieve {vp_url}: {exc}")

        if base_success and temp_out.stat().st_size > 0:
            temp_out.replace(out_path)
            logger.info(f"Consolidated download complete: {out_file} ({out_path.stat().st_size} bytes)")
            return True
        else:
            if temp_out.exists():
                temp_out.unlink()
            return False

    def _download_file(self, *args, **kwargs) -> bool:
        return self._execute_multi_vp_download(*args, **kwargs)

    def download_file(self, *args, **kwargs) -> bool:
        return self._execute_multi_vp_download(*args, **kwargs)


def main() -> None:
    """CLI entrypoint for standalone NDFD CONUS forcing extraction."""
    NDFDConusDownloader.cli_main(desc="Download operational NDFD CONUS forecast forcings")


if __name__ == "__main__":
    main()
