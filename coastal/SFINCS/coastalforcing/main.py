import os
import sys
import shutil
import re
import yaml
import traceback
from datetime import datetime, timezone
from pathlib import Path
from download_data.data_downloader import DataDownloader
from process_data.data_processor import DataProcessor
from download_data.download_noaa_obv_wl_sfincs import get_waterlevel_station_ids_from_nc, get_waterlevel_station_ids_from_obs, run_download_noaa_obv_wl_from_params
from process_data.compare_sfincs_map_vs_noaa import compare_sfincs_his_vs_noaa
import subprocess
import os

def run_sfincs_cmd(forcing_output_dir: str) -> None:
    """
    Run the SFINCS docker container with the given forcing_output_dir mounted to /data.
    Waits for completion and reports success or errors.
    """
    cmd = [
        "sudo", "docker", "run", "-ti",
        "-v", f"{os.path.abspath(forcing_output_dir)}:/data",
        "deltares/sfincs-cpu"
    ]

    try:
        print(f"[INFO] Running SFINCS with directory: {forcing_output_dir}")
        result = subprocess.run(
            cmd,
            check=True,            # raises CalledProcessError if non-zero exit
            capture_output=True,   # capture both stdout and stderr
            text=True
        )
        print("[SUCCESS] SFINCS completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("[ERROR] SFINCS run failed.")
        print("Exit code:", e.returncode)
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        raise

def _parse_utc(s: str) -> str:
    """
    Accept common ISO-8601 strings:
      - 'YYYY-MM-DDTHH:MM:SSZ'
      - 'YYYY-MM-DDTHH:MM:SS+00:00'
      - 'YYYY-MM-DD HH:MM:SSZ'
      - 'YYYY-MM-DDTHH-MM-SSZ'  (your legacy format)
    Return normalized string 'YYYY-MM-DDTHH-MM-SSZ' to preserve your downstream expectations.
    """
    raw = s.strip()

    # If already in your legacy 'T%H-%M-%SZ' format, accept it
    try:
        dt = datetime.strptime(raw, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    except Exception:
        pass

    # Normalize common ISO variants to aware UTC
    s2 = raw.replace(" ", "T")
    if s2.endswith(("Z", "z")):
        s2 = s2[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s2).astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%SZ")
    except Exception as e:
        raise ValueError(f"Unrecognized UTC time format '{s}': {e}")

def _normpath(*parts) -> str:
    return str(Path(*parts).resolve())

def _must_exist(path: str, label: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path

def _replace_param_line(lines, key, new_value):
    """
    Replace lines like:
        'tstart               = 20250101 000000'
    robustly (preserve indentation & trailing comments).
    """
    out = []
    pat = re.compile(rf'^(\s*{re.escape(key)}\s*=\s*)(.*?)(\s*(#.*)?)$', re.IGNORECASE)
    replaced = False
    for ln in lines:
        m = pat.match(ln)
        if m and not replaced:
            out.append(f"{m.group(1)}{new_value}{m.group(3)}")
            replaced = True
        else:
            out.append(ln)
    return out, replaced

# ---- Validation ----

def validate_config(cfg: dict):
    req = [
        "coastal_model",
        "meteo_source",
        "hydrology_source",
        "coastal_water_level_source",
        "domain_file",
        "start_time",
        "end_time",
        "raw_download_dir",
        "sim_dir",
    ]
    missing = [k for k in req if k not in cfg]
    empty   = [k for k in req if not cfg.get(k)]

    if missing:
        raise KeyError(f"Missing config keys: {', '.join(missing)}")
    if empty:
        raise ValueError(f"Empty values for config keys: {', '.join(empty)}")

    print("Configuration validated")

# ---- SFINCS prep ----

def prepare_sfincs_base_simulation_folder(cfg, domain_info):
    # Normalize & ensure dirs
    sim_root = _normpath(cfg['sim_dir'])
    start_iso = _parse_utc(cfg['start_time'])
    run_folder_name = f"{cfg['coastal_model']}_{start_iso}"
    sim_dir = _normpath(sim_root, run_folder_name)
    Path(sim_dir).mkdir(parents=True, exist_ok=True)

    source_path = _normpath(domain_info['domain'][0]['path'])
    _must_exist(source_path, "Base domain path")

    # Copy files; choose srcfile variant
    want_nwm = cfg["hydrology_source"].lower() == "nwm"
    for entry in Path(source_path).iterdir():
        dst = Path(sim_dir, entry.name)
        if entry.name in ("sfincs_nwm.src", "sfincs_ngen.src"):
            if want_nwm and entry.name == "sfincs_nwm.src":
                shutil.copy2(entry, dst)
                print(f"Copied {entry.name} -> {dst}")
            elif (not want_nwm) and entry.name == "sfincs_ngen.src":
                shutil.copy2(entry, dst)
                print(f"Copied {entry.name} -> {dst}")
        else:
            if entry.is_file():
                print(entry)
                print(dst)
                # shutil.copy2(entry, dst)
                print(f"Copied {entry.name} -> {dst}")
                try:
                    shutil.copy(entry, dst)  # best effort (includes metadata)
                    print(f"Copied {entry.name} -> {dst}")
                except PermissionError:
                    # Fallback: copy file contents only
                    shutil.copy(entry, dst)
                except OSError as e:
                    raise

    # Edit sfincs.inp
    # convert normalized 'YYYY-%m-%dT%H-%M-%SZ' into SFINCS 'YYYYMMDD HHMMSS'
    dts = datetime.strptime(_parse_utc(cfg['start_time']), "%Y-%m-%dT%H-%M-%SZ")
    dte = datetime.strptime(_parse_utc(cfg['end_time']), "%Y-%m-%dT%H-%M-%SZ")
    output_timestep = float(cfg['output_timestep'])
    
    sfincs_start = dts.strftime("%Y%m%d %H%M%S")
    sfincs_end   = dte.strftime("%Y%m%d %H%M%S")

    inp_path = _normpath(sim_dir, "sfincs.inp")
    if Path(inp_path).exists():
        with open(inp_path, "r") as f:
            lines = f.readlines()

        # do targeted replacements
        lines, ok1 = _replace_param_line(lines, "tref",   sfincs_start)
        lines, ok2 = _replace_param_line(lines, "tstart", sfincs_start)
        lines, ok3 = _replace_param_line(lines, "tstop",  sfincs_end)
        lines, ok4 = _replace_param_line(lines, "dtmapout",  output_timestep)

        srcfile_value = "sfincs_nwm.src" if want_nwm else "sfincs_ngen.src"
        lines, ok5 = _replace_param_line(lines, "srcfile", srcfile_value)

        with open(inp_path, "w") as f:
            f.writelines(lines)

        print(f"Updated tref/tstart/tstop/srcfile in {inp_path} "
              f"(replaced: {ok1},{ok2},{ok3},{ok4})")
    else:
        print(f"WARNING: sfincs.inp not found in {sim_dir}")

def prepare_schism_base_simulation_folder(cfg, domain_info):
    # Normalize & ensure dirs
    sim_root = _normpath(cfg['sim_dir'])
    start_iso = _parse_utc(cfg['start_time'])
    run_folder_name = f"{cfg['coastal_model']}_{start_iso}"
    sim_dir = _normpath(sim_root, run_folder_name)
    Path(sim_dir).mkdir(parents=True, exist_ok=True)

    dts = datetime.strptime(_parse_utc(cfg['start_time']), "%Y-%m-%dT%H-%M-%SZ")
    dte = datetime.strptime(_parse_utc(cfg['end_time']), "%Y-%m-%dT%H-%M-%SZ")

    startpdy = dts.strftime( "%Y%m%d")
    startcyc = dts.strftime("%H")

    forecastlength_in_hrs = int( (dte - dts).total_seconds()/3600 )

    here = Path(__file__).resolve().parent
    raw_download_dir=_normpath(here, cfg['raw_download_dir'])

    with open( f"{sim_dir}/schism_calib.cfg", "w") as schcfg:
      schcfg.write(f"export STARTPDY={startpdy}\n")
      schcfg.write(f"export STARTCYC={startcyc}\n")
      schcfg.write(f"export FCST_LENGTH_HRS={forecastlength_in_hrs}\n")
      schcfg.write("export HOT_START_FILE=\'\'\n")
      if cfg['coastal_water_level_source'] == 'tpxo':
        schcfg.write('export USE_TPXO="YES"\n')
      else:
        schcfg.write('export USE_TPXO="NO"\n')
      schcfg.write(f"export COASTAL_DOMAIN={cfg['domain_file']}\n")
      schcfg.write(f"export METEO_SOURCE={cfg['meteo_source'].upper()}\n")
      schcfg.write(f"export COASTAL_WORK_DIR={sim_dir}\n")
      schcfg.write(f"export RAW_DOWNLOAD_DIR={raw_download_dir}\n")


# ---- Main ----

def main():
    here = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.yaml"

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    print(f"Using config file: {config_file}")

    # Load config (no global chdir)
    cfg_path = _normpath(here, config_file)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)

    nwm_domain = cfg['domain_file'] if cfg['domain_file'] == 'hawaii' or cfg['domain_file'] == 'prvi' else \
                 'conus' 

    # Load domain info and normalize base path relative to the domain YAML
    domain_file = f"domain_lists/{cfg['coastal_model']}/{cfg['domain_file']}.yaml"
    with open(domain_file) as f:
        domain_info = yaml.safe_load(f)

    # Resolve the domain path relative to the YAML’s folder
    domain_yaml_dir = os.path.dirname(os.path.abspath(domain_file))
    raw_domain_path = domain_info['domain'][0]['path']

    # 🔧 Sanitize Windows-style separators for POSIX
    raw_domain_path_sanitized = raw_domain_path.replace('\\', '/')

    abs_domain_path = os.path.abspath(
        os.path.normpath(os.path.join(domain_yaml_dir, raw_domain_path_sanitized))
    )

    # Save the normalized absolute path back
    domain_info['domain'][0]['path'] = abs_domain_path

    # Ensure it exists
    if not os.path.isdir(abs_domain_path):
        raise FileNotFoundError(f"Resolved domain path not found: {abs_domain_path}")

    # Download data
    downloader = DataDownloader(
        coastal_model=cfg['coastal_model'],
        start_time=_parse_utc(cfg['start_time']),
        end_time=_parse_utc(cfg['end_time']),
        meteo_source=cfg['meteo_source'],
        hydrology_source=cfg['hydrology_source'],
        coastal_water_level_source=cfg['coastal_water_level_source'],
        raw_download_dir=_normpath(here, cfg['raw_download_dir']),
        nwm_domain=nwm_domain,
        domain_info=domain_info
    )
    downloader.download_all()

    # REORDERED: prepare first, then process
    if cfg["coastal_model"].lower() == "sfincs":
        prepare_sfincs_base_simulation_folder(cfg, domain_info)
    elif cfg["coastal_model"].lower() == "schism":
        prepare_schism_base_simulation_folder(cfg, domain_info)
    else:
        print(f"WARNING: No preparation routine defined for model '{cfg['coastal_model']}'.")

    # Process data
    sim_dir = _normpath(here, cfg['sim_dir'], f"{cfg['coastal_model']}_{_parse_utc(cfg['start_time'])}")

    if  cfg["coastal_model"].lower() == "sfincs": 
      tpxo_env = None
      ld_library_path = cfg.get('ld_library_path', None)
      if ld_library_path:
        tpxo_env={
            "LD_LIBRARY_PATH": ld_library_path
        }

      processor = DataProcessor(
        coastal_model=cfg['coastal_model'],
        domain_info=domain_info,
        sim_dir=sim_dir,
        start_time=_parse_utc(cfg['start_time']),
        end_time=_parse_utc(cfg['end_time']),
        meteo_source=cfg['meteo_source'],
        hydrology_source=cfg['hydrology_source'],
        coastal_water_level_source = cfg['coastal_water_level_source'],
        raw_download_dir=_normpath(here, cfg['raw_download_dir']),
        tpxo_relative_path=cfg.get('tpxo_relative_path', None),
        tpxo_model_control=cfg.get('tpxo_model_control', None),
        tpxo_env=tpxo_env
      )
      processor.process_all()

    print("Forcing file generation COMPLETE")

if __name__ == "__main__":
    main()

