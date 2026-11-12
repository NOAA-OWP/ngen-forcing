"""
Run one downloader standalone.

Run from ``NextGen_Forcings_Engine_BMI/``::

    python -m Forcing_Extraction_Scripts.download HRRR /data/hrrr "2026-11-10 13:00:00" --region Alaska
    python -m Forcing_Extraction_Scripts.download GFS /data/gfs "2026-11-10 13:00:00" --region Global
    python -m Forcing_Extraction_Scripts.download HRRR --help      # options and regions of one source
"""

from __future__ import annotations

import argparse
import sys
import typing

from Forcing_Extraction_Scripts.sources import SOURCES


def main(argv: typing.Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="python -m Forcing_Extraction_Scripts.download",
        description=__doc__.split("\n\n")[0],
        epilog="Sources: " + ", ".join(SOURCES) + ". Give a source followed by --help for its options.",
    )
    parser.add_argument("source", choices=list(SOURCES), metavar="source", help="Source to download.")
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv or ["--help"])
    source = parser.parse_args(argv[:1]).source
    SOURCES[source].cli_main(argv[1:], prog=f"{parser.prog} {source}")


if __name__ == "__main__":
    main()
