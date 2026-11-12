"""
Downloaders, one class per source, each parameterized over the regions it covers.

Analysis and Assimilation (AnA) variants are separate classes from the forecast ones.
"""

from __future__ import annotations

from Forcing_Extraction_Scripts.forecast_download_base import ForecastDownloader, Region
from Forcing_Extraction_Scripts.sources.cfs import CFSv2Downloader
from Forcing_Extraction_Scripts.sources.gfs import GFSDownloader
from Forcing_Extraction_Scripts.sources.hiresw import HiResWARWDownloader
from Forcing_Extraction_Scripts.sources.hrrr import (
    HRRRAnADownloader,
    HRRRDownloader,
    HRRRSubhourlyAnADownloader,
    HRRRSubhourlyDownloader,
)
from Forcing_Extraction_Scripts.sources.mrms import (
    MRMSMultiSensorDownloader,
    MRMSPrecipRateDownloader,
    MRMSRadarOnlyDownloader,
)
from Forcing_Extraction_Scripts.sources.nam import NAMNestAnADownloader, NAMNestDownloader
from Forcing_Extraction_Scripts.sources.nbm import NBMAnADownloader, NBMDownloader
from Forcing_Extraction_Scripts.sources.ndfd import NDFDDownloader
from Forcing_Extraction_Scripts.sources.rap import RAPAnADownloader, RAPDownloader
from Forcing_Extraction_Scripts.sources.stage_iv import StageIVDownloader

#: Every concrete downloader class.
DOWNLOADERS: tuple[type[ForecastDownloader], ...] = (
    CFSv2Downloader,
    GFSDownloader,
    HRRRDownloader,
    HRRRAnADownloader,
    HRRRSubhourlyDownloader,
    HRRRSubhourlyAnADownloader,
    HiResWARWDownloader,
    MRMSMultiSensorDownloader,
    MRMSPrecipRateDownloader,
    MRMSRadarOnlyDownloader,
    NAMNestDownloader,
    NAMNestAnADownloader,
    NBMDownloader,
    NBMAnADownloader,
    NDFDDownloader,
    RAPDownloader,
    RAPAnADownloader,
    StageIVDownloader,
)

#: Source name (class name without ``Downloader``) -> class, as used on command lines.
SOURCES: dict[str, type[ForecastDownloader]] = {
    cls.__name__.removesuffix("Downloader"): cls for cls in DOWNLOADERS
}

#: A source in one region.
Source = tuple[type[ForecastDownloader], Region]


def all_sources() -> list[Source]:
    """Every (downloader class, region) combination."""
    return [(cls, region) for cls in DOWNLOADERS for region in cls.REGIONS]


def parse_source(text: str) -> list[Source]:
    """
    Sources named by ``NAME`` (every region of that source) or ``NAME[Region]`` / ``NAME:Region``.

    Names are matched without regard to case.
    """
    name, sep, region = text.replace("[", ":").rstrip("]").partition(":")
    by_lower = {key.lower(): cls for key, cls in SOURCES.items()}
    cls = by_lower.get(name.strip().lower())
    if cls is None:
        raise ValueError(f"Unknown source {name!r}; expected one of {', '.join(SOURCES)}.")
    if sep:
        return [(cls, cls.resolve_region(region))]
    return [(cls, r) for r in cls.REGIONS]


__all__ = ["DOWNLOADERS", "SOURCES", "Source", "all_sources", "parse_source", "Region",
           *(cls.__name__ for cls in DOWNLOADERS)]
