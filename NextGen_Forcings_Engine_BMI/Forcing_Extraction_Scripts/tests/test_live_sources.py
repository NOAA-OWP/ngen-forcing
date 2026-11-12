"""
Live probes of every downloader, in every region it covers, against its upstream server (run with ``pytest --live``).

Each test picks a recently published cycle, enumerates the files the downloader would fetch, and
checks a sample with HEAD plus a ranged GET that must contain a GRIB2 message. See probe_sources.py.
"""

import pytest

from Forcing_Extraction_Scripts import probe_sources
from Forcing_Extraction_Scripts.sources import all_sources

#: Retired products are probed at their last cycles (see probe_sources.pick_cycle).
#: Not probed at all: the only source is NOMADS, which keeps about two days and rate-limits clients.
NOT_PROBED = {
    "HiResWARW[Hawaii]": "HiResW Hawaii retired 2026-11-03; NOMADS no longer holds pre-retirement cycles",
    "HiResWARW[Puerto_Rico]": "HiResW Puerto Rico retired 2026-11-03; NOMADS no longer holds them",
}


def _param(source):
    label = probe_sources.source_label(source)
    marks = [pytest.mark.skip(reason=NOT_PROBED[label])] if label in NOT_PROBED else []
    return pytest.param(source, marks=marks, id=label)


@pytest.mark.live
@pytest.mark.parametrize("source", [_param(s) for s in all_sources()])
def test_downloader_reaches_grib2_data(source):
    result = probe_sources.probe_source(source)
    assert result.ok, probe_sources.format_report([result])
