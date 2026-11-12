"""
Offline tests for the forcing downloaders and the extraction orchestrator.

Upstream HTTP is replaced by the ``server`` fixture (see conftest.py). Expected local paths follow
the readers in NextGen_Forcings_Engine/core/time_handling.py on the ensemble_mvp_fy2026_work engine.
"""

import datetime as dt
import email.utils
import gzip
import logging
import os
import sys
from pathlib import Path

import pytest

import forcing_extraction as fe
from config_models import ForcingExtractionConfig
from Forcing_Extraction_Scripts import forecast_download_base as fdb
from Forcing_Extraction_Scripts import download, probe_sources
from Forcing_Extraction_Scripts import sources as src
from Forcing_Extraction_Scripts.forecast_download_base import Region
from Forcing_Extraction_Scripts.sources import nam

#: Yesterday's 12Z cycle: recent enough that a 404 is retried rather than treated as archived.
CYCLE = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).replace(
    hour=12, minute=0, second=0, microsecond=0, tzinfo=None)
DAY = f"{CYCLE:%Y%m%d}"
PREV_DAY = f"{CYCLE - dt.timedelta(days=1):%Y%m%d}"
OLD_DAY = f"{CYCLE - dt.timedelta(days=9):%Y%m%d}"
START_ARG = f"{CYCLE + dt.timedelta(hours=1):%Y-%m-%d %H:%M:%S}"


def http_date(when):
    """``when`` (naive UTC) as an HTTP date."""
    return email.utils.format_datetime(when.replace(tzinfo=dt.timezone.utc), usegmt=True)


CONFIG_DIR = Path(fe.__file__).resolve().parent / "BMI_NextGen_Configs"


def make(source, out_dir, start=CYCLE + dt.timedelta(hours=1), **overrides):
    """Instantiate a (class, region) pair, or a class that covers a single region, for a one-cycle window."""
    if isinstance(source, tuple):
        cls, region = source
    else:
        cls, (region,) = source, source.REGIONS
    params = dict(out_dir=str(out_dir), start_time=start, lookback_hours=1, ens_number=1,
                  max_download_attempts=1, download_attempt_interval=0, region=region)
    params.update(overrides)
    return cls(**params)


def files(downloader):
    return list(downloader.iter_remote_files())


def rel(path, root):
    return os.path.relpath(path, root)


def config(**kw):
    base = dict(b_date_proc=CYCLE, input_forcings=[], supp_precip_forcings=[], input_force_dirs=[],
                supp_precip_dirs=[], fcst_input_horizons=[60], cfsv2EnsMember="", ana_flag=1, look_back=180)
    base.update(kw)
    return ForcingExtractionConfig(**base)


HRRR_ANA = (src.HRRRAnADownloader, Region.CONUS)
MRMS = (src.MRMSMultiSensorDownloader, Region.CONUS)
HRRR_PATH = f"hrrr.{DAY}/conus/hrrr.t12z.wrfsfcf01.grib2"
HRRR_ANA_URL = f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/{HRRR_PATH}"
MRMS_URL = (f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/{DAY}/"
            f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{DAY}-120000.grib2.gz")

# ----------------------------------------------------------------------------- registry and configs

REGISTERED = sorted(set(fe.FORCING_SRC.values()) | set(fe.FORCING_ANA_SRC.values()) | set(fe.STAGE4_MRMS_SRC.values()),
                    key=probe_sources.source_label)


@pytest.mark.parametrize("source", src.all_sources(), ids=probe_sources.source_label)
def test_every_source_constructs_in_every_region(source, tmp_path):
    cls, region = source
    downloader = make(source, tmp_path)
    assert downloader.region is region and downloader.domain is cls.REGIONS[region]
    files(downloader)  # enumerating must not raise


@pytest.mark.parametrize("source", REGISTERED, ids=probe_sources.source_label)
def test_registered_source_covers_its_region(source):
    cls, region = source
    assert cls in src.DOWNLOADERS and region in cls.REGIONS


def test_every_downloader_class_is_registered_somewhere():
    assert {cls for cls, _ in REGISTERED} == set(src.DOWNLOADERS)


# ----------------------------------------------------------------------------- regions

def test_region_is_required(tmp_path):
    with pytest.raises(TypeError, match="region"):
        src.GFSDownloader(out_dir=str(tmp_path), start_time=CYCLE, lookback_hours=1)
    assert make((src.HRRRDownloader, "alaska"), tmp_path).region is Region.ALASKA


def test_region_not_covered_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="RAPDownloader does not cover region Alaska; it covers CONUS"):
        make((src.RAPDownloader, Region.ALASKA), tmp_path)


@pytest.mark.parametrize("text,region", [("conus", Region.CONUS), ("Puerto Rico", Region.PUERTO_RICO),
                                         ("puerto_rico", Region.PUERTO_RICO), ("ALASKA", Region.ALASKA),
                                         (Region.HAWAII, Region.HAWAII)])
def test_region_parse(text, region):
    assert Region.parse(text) is region


def test_region_parse_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown region 'Guam'"):
        Region.parse("Guam")


def test_cli_selects_region_and_its_window_defaults(tmp_path):
    alaska = src.NAMNestDownloader.from_cli_args([str(tmp_path), START_ARG, "--region", "Alaska"])
    assert (alaska.region, alaska.lookback_hours, alaska.lagback_hours) == (Region.ALASKA, 36, 1)
    hawaii = src.NAMNestDownloader.from_cli_args([str(tmp_path), START_ARG, "--region", "hawaii"])
    assert (hawaii.region, hawaii.lookback_hours, hawaii.lagback_hours) == (Region.HAWAII, 24, 6)
    for argv in ([], ["--region", "CONUS"]):  # required, and must be a region the source covers
        with pytest.raises(SystemExit):
            src.NAMNestDownloader.from_cli_args([str(tmp_path), START_ARG, *argv])


def test_parse_source():
    assert src.parse_source("hrrr") == [(src.HRRRDownloader, Region.CONUS), (src.HRRRDownloader, Region.ALASKA)]
    assert src.parse_source("NBM[Alaska]") == [(src.NBMDownloader, Region.ALASKA)]
    assert src.parse_source("StageIV:conus") == [(src.StageIVDownloader, Region.CONUS)]
    with pytest.raises(ValueError, match="Unknown source"):
        src.parse_source("ECMWF")


SHIPPED_CONFIGS = sorted(CONFIG_DIR.glob("**/config.yml"))


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: str(p.relative_to(CONFIG_DIR).parent))
def test_shipped_config_builds_downloaders(path, tmp_path):
    cfg = ForcingExtractionConfig.from_yaml_file(path)
    if cfg.ana_flag == 1:
        fe.validate(cfg)
    for job in fe.build_jobs(cfg):
        job.out_dir = str(tmp_path)
        downloader = fe.make_downloader(cfg, job)
        assert files(downloader) or True  # constructing and enumerating must not raise


# ----------------------------------------------------------------------------- targets

@pytest.mark.parametrize("targets,max_hour,expected", [
    ([1, 2, 3, 6, 9], 4, [1, 2, 3, 6]),
    ([1, 2, 3, 6, 9], 6, [1, 2, 3, 6]),
    ([1, 2, 3], 10, [1, 2, 3]),
    (["tmp", "qpf"], 1, ["tmp", "qpf"]),
])
def test_cap_forecast_hours(targets, max_hour, expected):
    assert fdb.cap_forecast_hours(targets, max_hour) == expected


def test_gfs_horizon_cap(tmp_path):
    capped = make(src.GFSDownloader, tmp_path, max_forecast_hour=240)
    hours = [int(url.rsplit("sfluxgrbf", 1)[1][:3]) for _, url, _ in files(capped)]
    assert max(hours) == 240 and len(hours) == 160
    assert len(files(make(src.GFSDownloader, tmp_path))) == 208


@pytest.mark.parametrize("hour,count", [(0, 49), (6, 49), (12, 49), (18, 49), (3, 19), (21, 19)])
def test_alaska_hrrr_horizon_by_cycle(tmp_path, hour, count):
    cycle = CYCLE.replace(hour=hour)
    assert len(files(make((src.HRRRDownloader, Region.ALASKA), tmp_path, start=cycle + dt.timedelta(hours=1)))) == count


@pytest.mark.parametrize("region,tag", [(Region.CONUS, "co"), (Region.ALASKA, "ak"), (Region.HAWAII, "hi"),
                                        (Region.PUERTO_RICO, "pr")])
def test_nbm_targets_unique_through_f264(tmp_path, region, tag):
    downloader = make((src.NBMDownloader, region), tmp_path)
    assert files(downloader)[0][1].endswith(f"/blend.{DAY}/12/core/blend.t12z.core.f001.{tag}.grib2")
    targets = list(downloader.get_download_targets(CYCLE))
    assert len(targets) == len(set(targets))
    assert targets[0] == 1 and max(targets) == 264


def test_nbm_puerto_rico_ana_skips_missing_f000(tmp_path):
    (_, url, _), = files(make((src.NBMAnADownloader, Region.PUERTO_RICO), tmp_path))
    assert url.endswith("core.f001.pr.grib2")


def test_cfs_requires_valid_member(tmp_path):
    with pytest.raises(ValueError, match="ensemble member"):
        make(src.CFSv2Downloader, tmp_path, ens_number=None)
    assert "/6hrly_grib_01/" in files(make(src.CFSv2Downloader, tmp_path))[0][1]


def test_rap_fetches_both_grib_types_without_forcing_id(tmp_path):
    names = [os.path.basename(p) for _, _, p in files(make(src.RAPAnADownloader, tmp_path))]
    assert names == ["rap.t12z.awp130pgrbf01.grib2", "rap.t12z.awp130bgrbf01.grib2"]


@pytest.mark.parametrize("region,nest", [(Region.HAWAII, "hawaiinest"), (Region.PUERTO_RICO, "priconest")])
def test_nam_ana_uses_covering_six_hour_cycle(tmp_path, region, nest):
    urls = [u for _, u, _ in files(make((src.NAMNestAnADownloader, region), tmp_path, start=CYCLE.replace(hour=16)))]
    assert urls[0].endswith(f"nam.{DAY}/nam.t12z.{nest}.hiresf01.tm00.grib2")
    assert len(urls) == 6


# ----------------------------------------------------------------------------- paths read by the engine

def test_mrms_multisensor_urls_and_layouts(tmp_path):
    d = make(MRMS, tmp_path, start=CYCLE + dt.timedelta(hours=1))
    (_, url1, path1), (_, url2, _) = files(d)
    assert url1 == MRMS_URL
    assert f"Pass2_00.00/{DAY}/" in url2
    assert rel(path1, tmp_path) == f"MultiSensor_QPE_01H_Pass1_00.00/MRMS_MultiSensor_QPE_01H_Pass1_00.00_{DAY}-120000.grib2.gz"

    d = make(MRMS, tmp_path, local_layout=src.MRMSMultiSensorDownloader.LAYOUT_DATED)
    assert rel(files(d)[0][2], tmp_path).startswith(f"MultiSensor_QPE_01H_Pass1_00.00/{DAY}/")
    d = make(MRMS, tmp_path, local_layout=src.MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND)
    assert rel(files(d)[0][2], tmp_path).startswith(f"MultiSensor_QPE_01H_Pass1/{DAY}/")


@pytest.mark.parametrize("region,mrms_domain", [(Region.ALASKA, "ALASKA"), (Region.HAWAII, "HAWAII"),
                                                (Region.PUERTO_RICO, "CARIB")])
def test_regional_mrms_urls(tmp_path, region, mrms_domain):
    url = files(make((src.MRMSMultiSensorDownloader, region), tmp_path))[0][1]
    assert url.startswith(f"https://noaa-mrms-pds.s3.amazonaws.com/{mrms_domain}/MultiSensor_QPE_01H_Pass1_00.00/{DAY}/")


def test_mrms_radar_only_and_precip_rate_paths(tmp_path):
    (_, url, path), = files(make(src.MRMSRadarOnlyDownloader, tmp_path))
    name = f"MRMS_RadarOnly_QPE_01H_00.00_{DAY}-120000.grib2.gz"
    assert url == f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/RadarOnly_QPE_01H_00.00/{DAY}/{name}"
    assert rel(path, tmp_path) == f"CONUS/RadarOnly_QPE_01H_00.00/{DAY}/{name}"

    entries = files(make(src.MRMSPrecipRateDownloader, tmp_path))
    assert len(entries) == 30
    name = f"MRMS_PrecipRate_00.00_{DAY}-120200.grib2.gz"
    assert entries[1][1] == f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/PrecipRate_00.00/{DAY}/{name}"
    assert rel(entries[1][2], tmp_path) == f"{DAY}/{name}"


@pytest.mark.parametrize("source,url,local", [
    ((src.HRRRDownloader, Region.CONUS), "hrrr.{day}/conus/hrrr.t12z.wrfsfcf00.grib2", "hrrr.{day}/conus"),
    ((src.HRRRDownloader, Region.ALASKA), "hrrr.{day}/alaska/hrrr.t12z.wrfsfcf00.ak.grib2", "hrrr.{day}/alaska"),
    ((src.HRRRAnADownloader, Region.ALASKA), "hrrr.{day}/alaska/hrrr.t12z.wrfsfcf01.ak.grib2", "hrrr.{day}/alaska"),
    ((src.HRRRSubhourlyDownloader, Region.CONUS), "hrrr.{day}/conus/hrrr.t12z.wrfsubhf00.grib2", "hrrr.{day}/subhourly"),
    ((src.HRRRSubhourlyAnADownloader, Region.CONUS), "hrrr.{day}/conus/hrrr.t12z.wrfsubhf01.grib2", "hrrr.{day}/subhourly"),
    ((src.NAMNestDownloader, Region.ALASKA), "nam.{day}/nam.t12z.alaskanest.hiresf01.tm00.grib2", "nam.{day}"),
    ((src.NAMNestDownloader, Region.HAWAII), "nam.{day}/nam.t12z.hawaiinest.hiresf01.tm00.grib2", "nam.{day}"),
    ((src.NAMNestDownloader, Region.PUERTO_RICO), "nam.{day}/nam.t12z.priconest.hiresf01.tm00.grib2", "nam.{day}"),
    ((src.HiResWARWDownloader, Region.HAWAII), "hiresw.{day}/hiresw.t12z.arw_2p5km.f01.hi.grib2", "hiresw.{day}"),
    ((src.GFSDownloader, Region.GLOBAL), "gfs.{day}/12/atmos/gfs.t12z.sfluxgrbf001.grib2", "gfs.{day}/12/atmos"),
])
def test_first_file_of_cycle(tmp_path, source, url, local):
    _, first_url, first_path = files(make(source, tmp_path))[0]
    assert first_url.endswith("/" + url.format(day=DAY))
    assert rel(first_path, tmp_path) == os.path.join(local.format(day=DAY), os.path.basename(first_url))


def test_hiresw_cycles_differ_by_region(tmp_path):
    hawaii = make((src.HiResWARWDownloader, Region.HAWAII), tmp_path)
    puerto_rico = make((src.HiResWARWDownloader, Region.PUERTO_RICO), tmp_path)
    assert [h for h in range(24) if hawaii.should_process_hour(CYCLE.replace(hour=h))] == [0, 12]
    assert [h for h in range(24) if puerto_rico.should_process_hour(CYCLE.replace(hour=h))] == [6, 18]
    (_, url, _), *_ = files(make((src.HiResWARWDownloader, Region.PUERTO_RICO), tmp_path,
                                 start=CYCLE.replace(hour=19)))
    assert url.endswith(f"hiresw.{DAY}/hiresw.t18z.arw_2p5km.f01.pr.grib2")


@pytest.mark.parametrize("hour,count", [(12, 4), (13, 0), (15, 4)])
def test_alaska_hrrr_ana_reads_four_hours_of_three_hourly_cycles(tmp_path, hour, count):
    start = CYCLE.replace(hour=hour) + dt.timedelta(hours=1)
    assert len(files(make((src.HRRRAnADownloader, Region.ALASKA), tmp_path, start=start))) == count
    assert len(files(make((src.HRRRAnADownloader, Region.CONUS), tmp_path, start=start))) == 1


def test_ndfd_files_match_engine_names(tmp_path):
    entries = files(make(src.NDFDDownloader, tmp_path))
    local = {rel(p, tmp_path) for _, _, p in entries}
    assert local == {f"NDFD/{DAY}/wgrbbul/ndfd_conus/ndfd_conus_{t}.grib2" for t in ("tmp", "wdir", "wspd", "qpf")}
    qpf_urls = [u for _, u, p in entries if p.endswith("_qpf.grib2")]
    assert qpf_urls == ["https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus/VP.001-003/ds.qpf.bin"]


@pytest.mark.parametrize("region,name", [(Region.CONUS, f"st4_conus.{DAY}12.01h.grb2"),
                                         (Region.ALASKA, f"st4_ak.{DAY}12.06h.grb2")])
def test_stageiv_flat_output(tmp_path, region, name):
    (_, url, path), = files(make((src.StageIVDownloader, region), tmp_path))
    assert url.endswith(f"/pcpanl.{DAY}/{name}")
    assert rel(path, tmp_path) == name


# ----------------------------------------------------------------------------- download behavior



def test_download_publishes_file(server, tmp_path):
    server.add(HRRR_ANA_URL, b"GRIB-data")
    assert make(HRRR_ANA, tmp_path).run() is True
    assert (tmp_path / HRRR_PATH).read_bytes() == b"GRIB-data"
    leftovers = [p.name for p in (tmp_path / HRRR_PATH).parent.iterdir()
                 if p.name.startswith(".") and not p.name.endswith(".meta.json")]
    assert not leftovers


def test_missing_file_raises_instead_of_exiting(server, tmp_path):
    with pytest.raises(fdb.DownloadError, match="HTTP 404"):
        make(HRRR_ANA, tmp_path, max_download_attempts=3).run()
    assert server.gets == [HRRR_ANA_URL] * 3


def test_retries_do_not_sleep_after_last_attempt(server, tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    with pytest.raises(fdb.DownloadError):
        make(HRRR_ANA, tmp_path, max_download_attempts=3, download_attempt_interval=7).run()
    assert sleeps == [7, 7]


def test_missing_last_hour_tolerated_for_subhourly_averaging(server, tmp_path):
    assert make(HRRR_ANA, tmp_path, subhourly_avg=True).run() is True


@pytest.mark.parametrize("declared", [100, 2])
def test_body_length_disagreeing_with_content_length_is_reported(server, tmp_path, caplog, declared):
    server.add(HRRR_ANA_URL, b"GRIB", declared_length=declared)
    with pytest.raises(fdb.DownloadError, match=f"received 4 bytes but Content-Length is {declared}"):
        make(HRRR_ANA, tmp_path).run()
    assert not list((tmp_path / HRRR_PATH).parent.iterdir())
    assert any(r.levelname == "ERROR" and "inconsistent with its metadata" in r.message for r in caplog.records)


def test_connection_dropped_mid_body_is_retried(server, tmp_path):
    server.add(HRRR_ANA_URL, b"GRIB-0123456789", fail_times=1, fail_after=5)
    assert make(HRRR_ANA, tmp_path, max_download_attempts=2).run() is True
    assert (tmp_path / HRRR_PATH).read_bytes() == b"GRIB-0123456789"
    assert server.gets == [HRRR_ANA_URL] * 2


@pytest.mark.parametrize("url,body,match", [
    (HRRR_ANA_URL, b"<html>Service Unavailable</html>", "no GRIB marker"),
    (MRMS_URL, b"GRIB\x00\x00\x00\x02not-compressed", "named .gz"),
])
def test_content_not_matching_its_name_is_reported(server, tmp_path, caplog, url, body, match):
    server.add(url, body)
    source = HRRR_ANA if url == HRRR_ANA_URL else MRMS
    with pytest.raises(fdb.DownloadError, match=match):
        make(source, tmp_path).run()
    assert any(r.levelname == "ERROR" and "content does not match" in r.message for r in caplog.records)


@pytest.mark.parametrize("name,head,problem", [
    ("x.grib2", b"GRIB\x00\x00\x00\x02", ""),
    ("ds.temp.bin", b"****0000000000****\r\r\nGRIB\x00\x00\x00\x02", ""),
    ("x.grib2.gz", gzip.compress(b"GRIB\x00\x00\x00\x02"), ""),
    ("x.grb2", b"<html>", "no GRIB marker"),
    ("x.grib2.gz", b"GRIB", "named .gz"),
    ("https://h/p/x.grib2?sig=1", b"GRIB", ""),
])
def test_content_signature_problem(name, head, problem):
    assert problem in fdb.content_signature_problem(name, head)
    assert bool(fdb.content_signature_problem(name, head)) == bool(problem)


def test_requests_ask_for_identity_encoding(server, tmp_path, monkeypatch):
    seen = []

    def get(url, **kw):
        seen.append(kw.get("headers") or {})
        return server.get(url, **kw)

    monkeypatch.setattr("requests.get", get)
    server.add(HRRR_ANA_URL)
    make(HRRR_ANA, tmp_path).run()
    assert seen and all(h.get("Accept-Encoding") == "identity" for h in seen)


def test_availability_check_reports_without_exiting(server, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    assert make(HRRR_ANA, tmp_path, check_file_availability=1).run() is False
    assert "1 file(s) are currently missing" in caplog.text
    server.add(HRRR_ANA_URL)
    assert make(HRRR_ANA, tmp_path, check_file_availability=1).run() is True
    assert not server.gets


def test_no_matching_cycle_warns(server, tmp_path, caplog):
    make(src.GFSDownloader, tmp_path, start=CYCLE.replace(hour=14)).run()
    assert "nothing was downloaded" in caplog.text


def test_cli_main_exit_status(server, tmp_path, monkeypatch):
    cls = src.HRRRAnADownloader
    argv = ["prog", str(tmp_path), START_ARG, "--region", "CONUS", "--lookBackHours", "1", "--lagBackHours", "0",
            "--max_download_attempts", "1", "--inputforcing", "supp12"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        cls.cli_main()
    assert exc.value.code == 1
    server.add(HRRR_ANA_URL)
    with pytest.raises(SystemExit) as exc:
        cls.cli_main()
    assert exc.value.code == 0


def test_download_cli_runs_named_source(server, tmp_path):
    server.add(HRRR_ANA_URL)
    with pytest.raises(SystemExit) as exc:
        download.main(["HRRRAnA", str(tmp_path), START_ARG, "--region", "CONUS", "--lookBackHours", "1",
                       "--lagBackHours", "0", "--max_download_attempts", "1"])
    assert exc.value.code == 0 and (tmp_path / HRRR_PATH).exists()
    with pytest.raises(SystemExit) as exc:
        download.main(["NoSuchSource", str(tmp_path), START_ARG])
    assert exc.value.code == 2


# ----------------------------------------------------------------------------- cleanup

def test_rerun_with_cli_defaults_keeps_window_data(server, tmp_path):
    server.serve_all = True
    old = tmp_path / f"hrrr.{OLD_DAY}" / "conus"
    old.mkdir(parents=True)
    (old / "stale.grib2").write_bytes(b"x")

    def run():
        make((src.HRRRDownloader, Region.CONUS), tmp_path, start=CYCLE, lookback_hours=30, cleanback_hours=240,
             lagback_hours=1).run()

    run()
    first = len(server.gets)
    assert first > 0 and not old.exists()
    run()
    assert len(server.gets) == first  # nothing in the window was deleted and fetched again


def test_fixed_file_cleanup_removes_only_old_files(server, tmp_path):
    server.serve_all = True
    flat = tmp_path / "MultiSensor_QPE_01H_Pass1_00.00"
    flat.mkdir()
    old = flat / f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{PREV_DAY}-120000.grib2.gz"
    current = flat / f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{DAY}-090000.grib2.gz"
    old.write_bytes(b"old")
    current.write_bytes(b"current")
    make(MRMS, tmp_path, start=CYCLE, lookback_hours=3,
         cleanback_hours=240).run()
    assert not old.exists()
    assert current.read_bytes() == b"current"
    assert not any(url.endswith(f"{DAY}-090000.grib2.gz") and "Pass1" in url for url in server.gets)


def test_alaska_stageiv_does_not_wipe_output_directory(server, tmp_path):
    server.serve_all = True
    other = tmp_path / "unrelated.grib2"
    other.write_bytes(b"keep")
    make((src.StageIVDownloader, Region.ALASKA), tmp_path).run()
    assert other.read_bytes() == b"keep"
    assert (tmp_path / f"st4_ak.{DAY}12.06h.grb2").exists()


# ----------------------------------------------------------------------------- NDFD

NDFD = "https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus"
NDFD_OUT = f"NDFD/{DAY}/wgrbbul/ndfd_conus"
ISSUED = {"Last-Modified": http_date(CYCLE + dt.timedelta(minutes=30))}
LATER_ISSUANCE = http_date(CYCLE + dt.timedelta(days=2))


def serve_ndfd(server, headers=ISSUED, skip=()):
    for bin_file in ("ds.temp.bin", "ds.wdir.bin", "ds.wspd.bin", "ds.qpf.bin"):
        for vp in ("VP.001-003", "VP.004-007"):
            url = f"{NDFD}/{vp}/{bin_file}"
            if url not in skip:
                server.add(url, f"GRIB {vp} {bin_file};".encode(), headers=headers)


def test_ndfd_concatenates_valid_periods(server, tmp_path):
    serve_ndfd(server)
    make(src.NDFDDownloader, tmp_path).run()
    out = tmp_path / NDFD_OUT
    assert (out / "ndfd_conus_tmp.grib2").read_bytes() == b"GRIB VP.001-003 ds.temp.bin;GRIB VP.004-007 ds.temp.bin;"
    assert (out / "ndfd_conus_qpf.grib2").read_bytes() == b"GRIB VP.001-003 ds.qpf.bin;"
    data_files = sorted(p.name for p in out.iterdir() if not p.name.endswith(".meta.json"))
    assert data_files == [f"ndfd_conus_{t}.grib2" for t in ("qpf", "tmp", "wdir", "wspd")]


def test_ndfd_partial_field_is_not_published(server, tmp_path):
    serve_ndfd(server, skip={f"{NDFD}/VP.004-007/ds.temp.bin"})
    with pytest.raises(fdb.DownloadError):
        make(src.NDFDDownloader, tmp_path).run()
    out = tmp_path / NDFD_OUT
    assert not list(out.iterdir())


def test_ndfd_rejects_issuance_from_another_cycle(server, tmp_path):
    serve_ndfd(server, headers={"Last-Modified": LATER_ISSUANCE})
    with pytest.raises(fdb.DownloadError, match="only holds the latest issuance"):
        make(src.NDFDDownloader, tmp_path).run()


# ----------------------------------------------------------------------------- orchestrator

def test_stage4_options_also_fetch_mrms():
    jobs = fe.build_jobs(config(supp_precip_forcings=[12], supp_precip_dirs=["/s4", "/mrms"]))
    assert [(j.downloader, j.region, j.out_dir, j.local_layout) for j in jobs] == [
        (src.StageIVDownloader, Region.CONUS, "/s4", None),
        (src.MRMSMultiSensorDownloader, Region.CONUS, "/mrms", src.MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND),
    ]
    assert [j.label for j in jobs] == ["StageIV[CONUS]", "MRMSMultiSensor[CONUS]"]


def test_job_region_and_layout_reach_the_downloader(tmp_path):
    cfg = config(supp_precip_forcings=[11], supp_precip_dirs=[str(tmp_path / "s4"), str(tmp_path / "mrms")])
    stage4, mrms = (fe.make_downloader(cfg, job) for job in fe.build_jobs(cfg))
    assert isinstance(stage4, src.StageIVDownloader) and stage4.region is Region.ALASKA
    assert isinstance(mrms, src.MRMSMultiSensorDownloader) and mrms.region is Region.ALASKA
    assert mrms.local_layout == src.MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND


def test_same_source_is_registered_for_several_regions():
    assert fe.FORCING_SRC[5] == (src.HRRRDownloader, Region.CONUS)
    assert fe.FORCING_SRC[19] == (src.HRRRDownloader, Region.ALASKA)
    assert fe.FORCING_ANA_SRC["supp6"] == (src.MRMSMultiSensorDownloader, Region.HAWAII)


def test_subhourly_mrms_adds_dated_hourly_mrms():
    jobs = fe.build_jobs(config(supp_precip_forcings=[16], supp_precip_dirs=["/m"]))
    assert [(j.forcing, j.out_dir, j.local_layout) for j in jobs] == [
        ("supp16", "/m/MRMS_SubHourly", None),
        ("supp2", "/m/MRMS_Hourly", src.MRMSMultiSensorDownloader.LAYOUT_DATED),
    ]


def test_unregistered_forcing_is_logged(caplog):
    jobs = fe.build_jobs(config(input_forcings=[22], input_force_dirs=["/x"]))
    assert jobs == []
    assert "No downloader is registered for forcing 22" in caplog.text


def test_forecast_horizon_is_passed_as_cap(tmp_path):
    cfg = config(ana_flag=0, input_forcings=[3], input_force_dirs=[str(tmp_path)], fcst_input_horizons=[14400])
    (job,) = fe.build_jobs(cfg)
    assert fe.make_downloader(cfg, job).max_forecast_hour == 240


@pytest.mark.parametrize("kw,match", [({"ana_flag": 2}, "AnAFlag"), ({"look_back": -9999}, "LookBack")])
def test_validate_rejects_bad_configs(kw, match):
    with pytest.raises(ValueError, match=match):
        fe.validate(config(**kw))


def test_supp_dirs_must_match_supp_ids():
    with pytest.raises(ValueError, match="SuppPcpDirectories"):
        fe.build_jobs(config(supp_precip_forcings=[1, 6], supp_precip_dirs=["/only-one"]))


# ----------------------------------------------------------------------------- retired products

NAM_AK = (src.NAMNestDownloader, Region.ALASKA)


def test_retired_product_fails_after_retirement_without_requests(server, tmp_path):
    with pytest.raises(fdb.DownloadError, match="retired on 2026-11-03"):
        make(NAM_AK, tmp_path, start=dt.datetime(2026, 11, 10, 13)).run()
    assert server.gets == []


def test_retired_product_still_downloads_earlier_cycles(server, tmp_path):
    server.serve_all = True
    make(NAM_AK, tmp_path, start=dt.datetime(2026, 11, 2, 13), max_forecast_hour=1).run()
    assert server.gets == ["https://noaa-nam-pds.s3.amazonaws.com/nam.20261102/nam.t12z.alaskanest.hiresf01.tm00.grib2"]


def test_retired_product_reported_missing_in_check_mode(server, tmp_path, caplog):
    assert make(NAM_AK, tmp_path, start=dt.datetime(2026, 11, 10, 13), check_file_availability=1).run() is False
    assert server.heads == [] and "retired on 2026-11-03" in caplog.text


# ----------------------------------------------------------------------------- probe politeness

def test_probe_samples_one_file_from_rate_limited_hosts():
    nomads = [f"https://nomads.ncep.noaa.gov/pub/x/f{i:02d}.grb2" for i in range(5)]
    s3 = [f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/x/f{i:02d}.grib2" for i in range(5)]
    assert probe_sources.sample(nomads) == [nomads[0]]
    assert probe_sources.sample(s3) == [s3[0], s3[2], s3[4]]


def test_polite_session_spaces_requests_to_rate_limited_hosts(monkeypatch):
    clock, sleeps = [100.0], []
    monkeypatch.setattr(probe_sources.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe_sources.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(probe_sources.requests.Session, "request", lambda self, method, url, *a, **k: url)
    session = probe_sources.PoliteSession()
    for url in ("https://nomads.ncep.noaa.gov/a", "https://noaa-mrms-pds.s3.amazonaws.com/b",
                "https://nomads.ncep.noaa.gov/c"):
        session.request("GET", url)
    assert sleeps == [probe_sources.RATE_LIMITED_HOSTS["nomads.ncep.noaa.gov"][0]]


def test_probe_picks_last_cycles_before_retirement(tmp_path):
    cycle = probe_sources.pick_cycle(NAM_AK, dt.datetime(2026, 11, 10, 12), 26, str(tmp_path))
    assert cycle == nam.RETIRED_AFTER
