"""
National Digital Forecast Database (NDFD) CONUS operational forcing downloader.
Extracts and concatenates operational GRIB2 bins for temperature, wind, and precipitation.
"""

from __future__ import annotations

import datetime
import email.utils
import os

from Forcing_Extraction_Scripts.forecast_download_base import DownloadError, ForecastDownloader

PRIMARY_PERIOD = "VP.001-003"
EXTENDED_PERIOD = "VP.004-007"


class NDFDConusDownloader(ForecastDownloader):
    """
    Downloader implementation for operational CONUS NDFD forecasts.

    The operational feed (``opnl/AR.conus``) only holds the latest issuance under fixed names, so a
    cycle can only be populated while that issuance is close to the cycle time.
    """

    default_lookback = 6
    default_lagback = 0
    mutable_upstream = True

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

    #: Largest accepted difference between the upstream issuance time and the cycle being filled.
    MAX_ISSUANCE_OFFSET = datetime.timedelta(hours=6)

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

    def _part_urls(self, url: str, out_path: str) -> list[str]:
        """A field's valid periods, concatenated into one file."""
        field = os.path.basename(out_path)[len("ndfd_conus_"):-len(".grib2")]
        return [url.replace(PRIMARY_PERIOD, vp) for vp in self.FIELD_VALID_PERIODS[field]]

    def _check_part(self, url: str, headers) -> None:
        if f"/{PRIMARY_PERIOD}/" in url:
            self._check_issuance(headers, url)

    def _check_issuance(self, headers, url: str) -> None:
        """Refuse an upstream issuance that does not belong to the cycle being filled."""
        last_modified = headers.get("Last-Modified")
        if not last_modified:
            return
        issued = email.utils.parsedate_to_datetime(last_modified).astimezone(datetime.timezone.utc)
        cycle = self._cycle_time.replace(tzinfo=datetime.timezone.utc)
        if abs(issued - cycle) > self.MAX_ISSUANCE_OFFSET:
            raise DownloadError(
                f"{url} was issued at {issued:%Y-%m-%d %H:%M} UTC, more than "
                f"{self.MAX_ISSUANCE_OFFSET} from cycle {cycle:%Y-%m-%d %H:%M} UTC. The operational NDFD "
                "feed only holds the latest issuance, so it cannot supply this cycle."
            )


def main() -> None:
    """CLI entrypoint for standalone NDFD CONUS forcing extraction."""
    NDFDConusDownloader.cli_main(desc="Download operational NDFD CONUS forecast forcings")


if __name__ == "__main__":
    main()
