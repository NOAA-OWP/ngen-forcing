"""Multi-Radar Multi-Sensor (MRMS) precipitation downloaders. ``REGIONS`` maps to the MRMS domain directory."""

from __future__ import annotations

import typing
from datetime import datetime, timedelta

from Forcing_Extraction_Scripts.forecast_download_base import FileSpec, FixedFileDownloader, Region


class _MRMSBase(FixedFileDownloader):
    """Server location shared by the MRMS downloaders."""

    @property
    def base_url(self) -> str:
        return f"https://noaa-mrms-pds.s3.amazonaws.com/{self.domain}"


class MRMSMultiSensorDownloader(_MRMSBase):
    """
    Downloader for hourly MRMS MultiSensor QPE (Pass1 and Pass2).

    The server layout is ``<domain>/MultiSensor_QPE_01H_PassN_00.00/YYYYMMDD/<file>``. The local
    layout depends on which engine reader consumes the files and is selected by ``local_layout``.
    """

    REGIONS = {Region.CONUS: "CONUS", Region.ALASKA: "ALASKA", Region.HAWAII: "HAWAII",
               Region.PUERTO_RICO: "CARIB"}

    #: Flat per-pass directories, read for SuppPcp 2, 5, 6 and 10.
    LAYOUT_FLAT: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}_00.00"
    #: Per-pass, per-date directories, read under MRMS_Hourly for SuppPcp 16 bias correction.
    LAYOUT_DATED: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}_00.00/{date}"
    #: Per-date directories without the product version, read for the MRMS half of SuppPcp 11 and 12.
    LAYOUT_STAGE4_BLEND: typing.ClassVar[str] = "MultiSensor_QPE_01H_{pass_}/{date}"

    def __init__(self, *args, local_layout: str = LAYOUT_FLAT, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.local_layout: str = local_layout

    def get_file_specs(self, d_start: datetime) -> list[FileSpec]:
        date = d_start.strftime("%Y%m%d")
        specs: list[FileSpec] = []
        for pass_ in ("Pass1", "Pass2"):
            product = f"MultiSensor_QPE_01H_{pass_}_00.00"
            filename = f"MRMS_{product}_{date}-{d_start.strftime('%H')}0000.grib2.gz"
            specs.append(FileSpec(f"{product}/{date}", filename, self.local_layout.format(pass_=pass_, date=date)))
        return specs


class MRMSRadarOnlyDownloader(_MRMSBase):
    """Downloader for hourly MRMS RadarOnly QPE."""

    REGIONS = {Region.CONUS: "CONUS"}

    def get_file_specs(self, d_start):
        date = d_start.strftime("%Y%m%d")
        product = "RadarOnly_QPE_01H_00.00"
        filename = f"MRMS_{product}_{date}-{d_start.strftime('%H')}0000.grib2.gz"
        # Stored locally under <domain>/<product>/<date>, as on the server.
        return [FileSpec(f"{product}/{date}", filename, f"{self.domain}/{product}/{date}")]


class MRMSPrecipRateDownloader(_MRMSBase):
    """Downloader for MRMS sub-hourly (2-minute) precipitation rate."""

    REGIONS = {Region.CONUS: "CONUS"}

    def get_file_specs(self, d_start):
        date = d_start.strftime("%Y%m%d")
        specs = []
        for minute in range(0, 60, 2):
            step = d_start + timedelta(minutes=minute)
            filename = f"MRMS_PrecipRate_00.00_{date}-{step.strftime('%H%M%S')}.grib2.gz"
            specs.append(FileSpec(f"PrecipRate_00.00/{date}", filename, date))
        return specs
