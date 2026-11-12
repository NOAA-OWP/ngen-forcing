"""Stage IV precipitation analysis downloader."""

from __future__ import annotations

from dataclasses import dataclass

from Forcing_Extraction_Scripts.forecast_download_base import FileSpec, FixedFileDownloader, Region


@dataclass(frozen=True)
class StageIVDomain:
    """One Stage IV domain."""

    #: Domain tag in file names.
    tag: str
    #: Accumulation period in hours, which is also the interval between analyses.
    hours: int


class StageIVDownloader(FixedFileDownloader):
    """Downloader for Stage IV precipitation analyses (flat output directory)."""

    REGIONS = {Region.CONUS: StageIVDomain("conus", 1), Region.ALASKA: StageIVDomain("ak", 6)}

    default_lookback = 36
    default_cleanback = 240
    default_lagback = 0
    mutable_upstream = True  # Stage IV analyses are rewritten upstream about 24 h after valid time

    @property
    def base_url(self):
        return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/v4.1"

    def should_process_hour(self, d_start):
        return d_start.hour % self.domain.hours == 0

    def get_file_specs(self, d_start):
        filename = f"st4_{self.domain.tag}.{d_start.strftime('%Y%m%d%H')}.{self.domain.hours:02d}h.grb2"
        return [FileSpec(f"pcpanl.{d_start.strftime('%Y%m%d')}", filename)]
