"""
Offline tests for the forcing downloaders and the extraction orchestrator.

Upstream HTTP is replaced by the ``server`` fixture (see conftest.py). Expected local paths follow
the readers in NextGen_Forcings_Engine/core/time_handling.py on the ensemble_mvp_fy2026_work engine.
"""

import datetime as dt
import logging
import os
import sys
from pathlib import Path

import pytest

import forcing_extraction as fe
from config_models import ForcingExtractionConfig
from Forcing_Extraction_Scripts import forecast_download_base as fdb
from Forcing_Extraction_Scripts import probe_sources

CYCLE = dt.datetime(2026, 11, 10, 12)
CONFIG_DIR = Path(fe.__file__).resolve().parent / "BMI_NextGen_Configs"


def load(script):
    return fe.load_downloader_class(fe.SCRIPT_DIR / script)


def make(script_or_cls, out_dir, start=CYCLE + dt.timedelta(hours=1), **overrides):
    cls = load(script_or_cls) if isinstance(script_or_cls, str) else script_or_cls
    params = dict(
        out_dir=str(out_dir), start_time=start, lookback_hours=1, cleanback_hours=0, lagback_hours=0,
        ens_number=1, input_forcing=None, max_download_attempts=1, download_attempt_interval=0,
        check_file_availability=0, subhourly_avg=False, gfs_extended=False,
    )
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


# ----------------------------------------------------------------------------- registry and configs

REGISTERED = sorted(set(fe.FORCING_SRC.values()) | set(fe.FORCING_ANA_SRC.values()) | set(fe.STAGE4_MRMS_SRC.values()))


@pytest.mark.parametrize("script", probe_sources.all_scripts())
def test_every_script_defines_one_downloader(script):
    cls = load(script)
    assert issubclass(cls, fdb.ForecastDownloader)


@pytest.mark.parametrize("script", REGISTERED)
def test_registered_script_exists(script):
    assert (fe.SCRIPT_DIR / script).is_file()


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
    capped = make("Global/get_prod_GFS.py", tmp_path, max_forecast_hour=240)
    hours = [int(url.rsplit("sfluxgrbf", 1)[1][:3]) for _, url, _ in files(capped)]
    assert max(hours) == 240 and len(hours) == 160
    assert len(files(make("Global/get_prod_GFS.py", tmp_path))) == 208


@pytest.mark.parametrize("hour,count", [(0, 49), (6, 49), (12, 49), (18, 49), (3, 19), (21, 19)])
def test_alaska_hrrr_horizon_by_cycle(tmp_path, hour, count):
    cycle = CYCLE.replace(hour=hour)
    assert len(files(make("Alaska/get_Alaska_HRRR.py", tmp_path, start=cycle + dt.timedelta(hours=1)))) == count


@pytest.mark.parametrize("script", ["Alaska/get_prod_NBM_Alaska.py", "CONUS/get_prod_NBM_Conus.py",
                                    "Hawaii/get_prod_NBM_Hawaii.py", "Puerto_Rico/get_prod_NBM_Puerto_Rico.py"])
def test_nbm_targets_unique_through_f264(tmp_path, script):
    targets = list(make(script, tmp_path).get_download_targets(CYCLE))
    assert len(targets) == len(set(targets))
    assert targets[0] == 1 and max(targets) == 264


def test_nbm_puerto_rico_ana_skips_missing_f000(tmp_path):
    (_, url, _), = files(make("Puerto_Rico/get_prod_NBM_Puerto_Rico_AnA.py", tmp_path))
    assert url.endswith("core.f001.pr.grib2")


def test_cfs_requires_valid_member(tmp_path):
    with pytest.raises(ValueError, match="ensemble member"):
        make("Global/get_CFSv2.py", tmp_path, ens_number=None)
    assert "/6hrly_grib_01/" in files(make("Global/get_CFSv2.py", tmp_path))[0][1]


def test_rap_fetches_both_grib_types_without_forcing_id(tmp_path):
    names = [os.path.basename(p) for _, _, p in files(make("CONUS/get_conus_RAP_AnA.py", tmp_path))]
    assert names == ["rap.t12z.awp130pgrbf01.grib2", "rap.t12z.awp130bgrbf01.grib2"]


@pytest.mark.parametrize("script,domain", [("Hawaii/get_prod_NAM_Nest_Hawaii_AnA.py", "hawaiinest"),
                                           ("Puerto_Rico/get_prod_NAM_Nest_Puerto_Rico_AnA.py", "priconest")])
def test_nam_ana_uses_covering_six_hour_cycle(tmp_path, script, domain):
    urls = [u for _, u, _ in files(make(script, tmp_path, start=CYCLE.replace(hour=16)))]
    assert urls[0].endswith(f"nam.20261110/nam.t12z.{domain}.hiresf01.tm00.grib2")
    assert len(urls) == 6


# ----------------------------------------------------------------------------- paths read by the engine

def test_mrms_multisensor_urls_and_layouts(tmp_path):
    d = make("CONUS/get_conus_MRMS_MultiSensor.py", tmp_path, start=CYCLE + dt.timedelta(hours=1))
    (_, url1, path1), (_, url2, _) = files(d)
    assert url1 == ("https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MultiSensor_QPE_01H_Pass1_00.00/20261110/"
                    "MRMS_MultiSensor_QPE_01H_Pass1_00.00_20261110-120000.grib2.gz")
    assert "Pass2_00.00/20261110/" in url2
    assert rel(path1, tmp_path) == "MultiSensor_QPE_01H_Pass1_00.00/MRMS_MultiSensor_QPE_01H_Pass1_00.00_20261110-120000.grib2.gz"

    d.local_layout = fdb.MRMSMultiSensorDownloader.LAYOUT_DATED
    assert rel(files(d)[0][2], tmp_path).startswith("MultiSensor_QPE_01H_Pass1_00.00/20261110/")
    d.local_layout = fdb.MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND
    assert rel(files(d)[0][2], tmp_path).startswith("MultiSensor_QPE_01H_Pass1/20261110/")


@pytest.mark.parametrize("script,region", [("Alaska/get_MRMS_MultiSensor_Alaska.py", "ALASKA"),
                                           ("Hawaii/get_MRMS_MultiSensor_Hawaii.py", "HAWAII"),
                                           ("Puerto_Rico/get_MRMS_MultiSensor_Puerto_Rico.py", "CARIB")])
def test_regional_mrms_urls(tmp_path, script, region):
    url = files(make(script, tmp_path))[0][1]
    assert url.startswith(f"https://noaa-mrms-pds.s3.amazonaws.com/{region}/MultiSensor_QPE_01H_Pass1_00.00/20261110/")


def test_ndfd_files_match_engine_names(tmp_path):
    entries = files(make("CONUS/get_conus_NDFD.py", tmp_path))
    local = {rel(p, tmp_path) for _, _, p in entries}
    assert local == {f"NDFD/20261110/wgrbbul/ndfd_conus/ndfd_conus_{t}.grib2" for t in ("tmp", "wdir", "wspd", "qpf")}
    qpf_urls = [u for _, u, p in entries if p.endswith("_qpf.grib2")]
    assert qpf_urls == ["https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus/VP.001-003/ds.qpf.bin"]


@pytest.mark.parametrize("script,name", [("CONUS/get_conus_StageIV.py", "st4_conus.2026111012.01h.grb2"),
                                         ("Alaska/get_Alaska_StageIV.py", "st4_ak.2026111012.06h.grb2")])
def test_stageiv_flat_output(tmp_path, script, name):
    (_, url, path), = files(make(script, tmp_path))
    assert url.endswith(f"/pcpanl.20261110/{name}")
    assert rel(path, tmp_path) == name


# ----------------------------------------------------------------------------- download behavior

HRRR_ANA = "CONUS/get_conus_HRRR_AnA.py"
HRRR_ANA_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20261110/conus/hrrr.t12z.wrfsfcf01.grib2"


def test_download_publishes_file(server, tmp_path):
    server.add(HRRR_ANA_URL, b"GRIB-data")
    assert make(HRRR_ANA, tmp_path).run() is True
    assert (tmp_path / "hrrr.20261110/conus/hrrr.t12z.wrfsfcf01.grib2").read_bytes() == b"GRIB-data"
    assert not [p for p in (tmp_path / "hrrr.20261110/conus").iterdir() if p.name.startswith(".")]


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


def test_incomplete_transfer_is_rejected(server, tmp_path):
    server.add(HRRR_ANA_URL, b"GRIB", declared_length=100)
    with pytest.raises(fdb.DownloadError, match="incomplete transfer"):
        make(HRRR_ANA, tmp_path).run()
    assert not list((tmp_path / "hrrr.20261110/conus").iterdir())


def test_availability_check_reports_without_exiting(server, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    assert make(HRRR_ANA, tmp_path, check_file_availability=1).run() is False
    assert "1 file(s) are currently missing" in caplog.text
    server.add(HRRR_ANA_URL)
    assert make(HRRR_ANA, tmp_path, check_file_availability=1).run() is True
    assert not server.gets


def test_no_matching_cycle_warns(server, tmp_path, caplog):
    make("Global/get_prod_GFS.py", tmp_path, start=CYCLE.replace(hour=14)).run()
    assert "nothing was downloaded" in caplog.text


def test_cli_main_exit_status(server, tmp_path, monkeypatch):
    cls = load(HRRR_ANA)
    argv = ["prog", str(tmp_path), "2026-11-10 13:00:00", "--lookBackHours", "1", "--lagBackHours", "0",
            "--max_download_attempts", "1", "--inputforcing", "supp12"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        cls.cli_main()
    assert exc.value.code == 1
    server.add(HRRR_ANA_URL)
    with pytest.raises(SystemExit) as exc:
        cls.cli_main()
    assert exc.value.code == 0


# ----------------------------------------------------------------------------- cleanup

def test_rerun_with_cli_defaults_keeps_window_data(server, tmp_path):
    server.serve_all = True
    old = tmp_path / "hrrr.20261101" / "conus"
    old.mkdir(parents=True)
    (old / "stale.grib2").write_bytes(b"x")

    def run():
        make("CONUS/get_conus_HRRR.py", tmp_path, start=CYCLE, lookback_hours=30, cleanback_hours=240,
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
    old = flat / "MRMS_MultiSensor_QPE_01H_Pass1_00.00_20261109-120000.grib2.gz"
    current = flat / "MRMS_MultiSensor_QPE_01H_Pass1_00.00_20261110-090000.grib2.gz"
    old.write_bytes(b"old")
    current.write_bytes(b"current")
    make("CONUS/get_conus_MRMS_MultiSensor.py", tmp_path, start=CYCLE, lookback_hours=3,
         cleanback_hours=240).run()
    assert not old.exists()
    assert current.read_bytes() == b"current"
    assert not any(url.endswith("20261110-090000.grib2.gz") and "Pass1" in url for url in server.gets)


def test_alaska_stageiv_does_not_wipe_output_directory(server, tmp_path):
    server.serve_all = True
    other = tmp_path / "unrelated.grib2"
    other.write_bytes(b"keep")
    make("Alaska/get_Alaska_StageIV.py", tmp_path).run()
    assert other.read_bytes() == b"keep"
    assert (tmp_path / "st4_ak.2026111012.06h.grb2").exists()


# ----------------------------------------------------------------------------- NDFD

NDFD = "https://noaa-ndfd-pds.s3.amazonaws.com/opnl/AR.conus"
ISSUED = {"Last-Modified": "Tue, 10 Nov 2026 12:30:00 GMT"}


def serve_ndfd(server, headers=ISSUED, skip=()):
    for bin_file in ("ds.temp.bin", "ds.wdir.bin", "ds.wspd.bin", "ds.qpf.bin"):
        for vp in ("VP.001-003", "VP.004-007"):
            url = f"{NDFD}/{vp}/{bin_file}"
            if url not in skip:
                server.add(url, f"GRIB {vp} {bin_file};".encode(), headers=headers)


def test_ndfd_concatenates_valid_periods(server, tmp_path):
    serve_ndfd(server)
    make("CONUS/get_conus_NDFD.py", tmp_path).run()
    out = tmp_path / "NDFD/20261110/wgrbbul/ndfd_conus"
    assert (out / "ndfd_conus_tmp.grib2").read_bytes() == b"GRIB VP.001-003 ds.temp.bin;GRIB VP.004-007 ds.temp.bin;"
    assert (out / "ndfd_conus_qpf.grib2").read_bytes() == b"GRIB VP.001-003 ds.qpf.bin;"
    assert sorted(p.name for p in out.iterdir()) == [f"ndfd_conus_{t}.grib2" for t in ("qpf", "tmp", "wdir", "wspd")]


def test_ndfd_partial_field_is_not_published(server, tmp_path):
    serve_ndfd(server, skip={f"{NDFD}/VP.004-007/ds.temp.bin"})
    with pytest.raises(fdb.DownloadError):
        make("CONUS/get_conus_NDFD.py", tmp_path).run()
    out = tmp_path / "NDFD/20261110/wgrbbul/ndfd_conus"
    assert not list(out.iterdir())


def test_ndfd_rejects_issuance_from_another_cycle(server, tmp_path):
    serve_ndfd(server, headers={"Last-Modified": "Thu, 12 Nov 2026 12:00:00 GMT"})
    with pytest.raises(fdb.DownloadError, match="only holds the latest issuance"):
        make("CONUS/get_conus_NDFD.py", tmp_path).run()


# ----------------------------------------------------------------------------- orchestrator

def test_stage4_options_also_fetch_mrms():
    jobs = fe.build_jobs(config(supp_precip_forcings=[12], supp_precip_dirs=["/s4", "/mrms"]))
    assert [(j.script, j.out_dir, j.local_layout) for j in jobs] == [
        ("CONUS/get_conus_StageIV.py", "/s4", None),
        ("CONUS/get_conus_MRMS_MultiSensor.py", "/mrms", fdb.MRMSMultiSensorDownloader.LAYOUT_STAGE4_BLEND),
    ]


def test_subhourly_mrms_adds_dated_hourly_mrms():
    jobs = fe.build_jobs(config(supp_precip_forcings=[16], supp_precip_dirs=["/m"]))
    assert [(j.forcing, j.out_dir, j.local_layout) for j in jobs] == [
        ("supp16", "/m/MRMS_SubHourly", None),
        ("supp2", "/m/MRMS_Hourly", fdb.MRMSMultiSensorDownloader.LAYOUT_DATED),
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
