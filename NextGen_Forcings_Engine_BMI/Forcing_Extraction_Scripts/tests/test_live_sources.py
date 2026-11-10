"""
Live probes of every downloader script against its upstream server (run with ``pytest --live``).

Each test picks a recently published cycle, enumerates the files the downloader would fetch, and
checks a sample with HEAD plus a ranged GET that must contain a GRIB2 message. See probe_sources.py.
"""

import pytest

from Forcing_Extraction_Scripts import probe_sources

#: Retired products are probed at their last cycles (see probe_sources.pick_cycle).
#: Not probed at all: the only source is NOMADS, which keeps about two days and rate-limits clients.
NOT_PROBED = {
    "Hawaii/get_ARW_Hawaii.py": "HiResW Hawaii retired 2026-11-03; NOMADS no longer holds pre-retirement cycles",
    "Puerto_Rico/get_ARW_Puerto_Rico.py": "HiResW Puerto Rico retired 2026-11-03; NOMADS no longer holds them",
}


def _param(script):
    marks = [pytest.mark.skip(reason=NOT_PROBED[script])] if script in NOT_PROBED else []
    return pytest.param(script, marks=marks, id=script)


@pytest.mark.live
@pytest.mark.parametrize("script", [_param(s) for s in probe_sources.all_scripts()])
def test_downloader_reaches_grib2_data(script):
    result = probe_sources.probe_script(script)
    assert result.ok, probe_sources.format_report([result])
