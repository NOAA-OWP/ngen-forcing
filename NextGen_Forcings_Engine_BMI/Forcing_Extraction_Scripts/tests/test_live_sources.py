"""
Live probes of every downloader script against its upstream server (run with ``pytest --live``).

Each test picks a recently published cycle, enumerates the files the downloader would fetch, and
checks a sample with HEAD plus a ranged GET that must contain a GRIB2 message. See probe_sources.py.
"""

import pytest

from Forcing_Extraction_Scripts import probe_sources

#: Scripts whose upstream data was confirmed unavailable on 2026-11-10. Non-strict, so a recovered
#: source shows up as XPASS rather than failing the run.
KNOWN_UPSTREAM_GAPS = {
    "Alaska/get_prod_NAM_Nest_Alaska.py": "noaa-nam-pds on S3 has no data after 2026-11-03",
    "Hawaii/get_prod_NAM_Nest_Hawaii.py": "NAM retired 2026-11-03; noaa-nam-pds has no later cycles",
    "Hawaii/get_prod_NAM_Nest_Hawaii_AnA.py": "NAM retired 2026-11-03; noaa-nam-pds has no later cycles",
    "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico.py": "NAM retired 2026-11-03; noaa-nam-pds has no later cycles",
    "Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico_AnA.py": "NAM retired 2026-11-03; noaa-nam-pds has no later cycles",
    "Hawaii/get_ARW_Hawaii.py": "NOMADS hiresw/prod only carries the Guam domain",
    "Puerto_Rico/get_ARW_Puerto_Rico.py": "NOMADS hiresw/prod only carries the Guam domain",
}


def _param(script):
    reason = KNOWN_UPSTREAM_GAPS.get(script)
    marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
    return pytest.param(script, marks=marks, id=script)


@pytest.mark.live
@pytest.mark.parametrize("script", [_param(s) for s in probe_sources.all_scripts()])
def test_downloader_reaches_grib2_data(script):
    result = probe_sources.probe_script(script)
    assert result.ok, probe_sources.format_report([result])
