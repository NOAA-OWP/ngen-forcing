import os
from datetime import datetime, timedelta
from typing import Optional, Iterable

import requests


def _glofs_cycle_and_suffix(dt: datetime) -> tuple[str, str]:
    """
    Given a datetime (UTC), return (cycle, suffix) like ('t00z', 'n003').
    """
    cycle_hour = (dt.hour // 6) * 6
    cycle = f"t{cycle_hour:02d}z"
    suffix = f"n{dt.hour % 6:03d}"
    return cycle, suffix


def _glofs_filename(short_model: str, dt: datetime) -> str:
    """
    Build the filename for a given datetime using your breakpoint rule:
      - If (YYYY, mm) >= (2024, 9): {short}.{cycle}.{datestr}.fields.{suffix}.nc
      - Else:                        nos.{short}.fields.{suffix}.{datestr}.{cycle}.nc
    """
    datestr = dt.strftime("%Y%m%d")
    cycle, suffix = _glofs_cycle_and_suffix(dt)
    ym = (dt.year, dt.month)

    if ym >= (2024, 9):
        # New pattern
        # e.g., leofs.t00z.20241201.fields.n000.nc
        return f"{short_model}.{cycle}.{datestr}.fields.{suffix}.nc"
    else:
        # Old NOS pattern
        # e.g., nos.leofs.fields.n000.20231201.t00z.nc
        return f"nos.{short_model}.fields.{suffix}.{datestr}.{cycle}.nc"


def _glofs_url(
    access_area: str,
    dt: datetime,
    base_url: str = "https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
) -> str:
    """
    Compose the full URL:
      {base_url}/{access_area}/{YYYY}/{mm}/{filename}
    where filename follows the breakpoint rule above.
    """
    short = access_area.split("-")[-1]  # '...-leofs' -> 'leofs'
    yyyy = f"{dt.year:04d}"
    mm = dt.strftime("%m")
    fname = _glofs_filename(short, dt)
    return f"{base_url}/{access_area}/{yyyy}/{mm}/{fname}"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _download_once(url: str, dest: str, timeout: int = 90) -> str:
    """
    Single-attempt streaming download. Raises on HTTP errors.
    """
    _ensure_dir(os.path.dirname(dest))
    print(f"[GLOFS] GET {url} -> {dest}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dest


def ensure_glofs_local(
    dt: datetime,
    outdir: str,
    *,
    access_area: str = "lake-erie-operational-forecast-system-leofs",
    base_url: str = "https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
    allow_cached: bool = True,
) -> Optional[str]:
    """
    Ensure the GLOFS file for a single datetime exists locally in `outdir`.
    - Checks for an existing file by basename first.
    - If not present, builds the URL (with your pattern rules) and downloads once.
    - Returns local path on success, or None if download fails.
    """
    url = _glofs_url(access_area, dt, base_url)
    dest = os.path.join(_ensure_dir(outdir), os.path.basename(url))

    if allow_cached and os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[GLOFS] exists: {dest}")
        return dest

    try:
        return _download_once(url, dest)
    except Exception as e:
        print(f"[GLOFS] download failed: {url} → {e}")
        return None


def download_glofs_range(
    start: datetime,
    end: datetime,
    step: timedelta,
    outdir: str,
    *,
    access_area: str = "lake-erie-operational-forecast-system-leofs",
    base_url: str = "https://www.ncei.noaa.gov/data/operational-nowcast-and-forecast-hydrodynamic-model-systems-co-ops/access",
    allow_cached: bool = True,
) -> list[str]:
    """
    Convenience helper: iterate [start, end] by `step` and ensure each file is local.
    Returns the list of successfully materialized local file paths (order by time).
    """
    paths: list[str] = []
    t = start
    while t <= end:
        p = ensure_glofs_local(
            t,
            outdir,
            access_area=access_area,
            base_url=base_url,
            allow_cached=allow_cached,
        )
        if p:
            paths.append(p)
        t += step
    return paths

