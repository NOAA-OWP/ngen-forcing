"""
Live probes of every downloader script against its upstream server (run with ``pytest --live``).

Each test picks a recently published cycle, enumerates the files the downloader would fetch, and
checks a sample with HEAD plus a ranged GET that must contain a GRIB2 message. See probe_sources.py.
"""

import pytest

from Forcing_Extraction_Scripts import probe_sources

def _param(script):
    return pytest.param(script, id=script)


@pytest.mark.live
@pytest.mark.parametrize("script", [_param(s) for s in probe_sources.all_scripts()])
def test_downloader_reaches_grib2_data(script):
    result = probe_sources.probe_script(script)
    assert result.ok, probe_sources.format_report([result])
