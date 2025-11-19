"""
XYN Decimate script.

Clip a XYN to a desired polygon.
"""

import argparse
import pathlib
import numpy as np
from itertools import islice, compress
from common.io import read_xyn, write_xyn, read_polygon
from common.geometry import clip_point_to_roi

def process_stations(d, polygon=None, n=1):
    # Remove duplicate station labels
    uniq_index, uniq_idxs = np.unique(d['index'], return_index=True)
    d['index'] = uniq_index
    d['values'] = d['values'][uniq_idxs]

    if polygon is not None:
        idxs = clip_point_to_roi(polygon, d['values'])
        d['values'] = d['values'][idxs]
        d['index'] = d['index'][idxs]

    new_index, new_values = zip(*islice(zip(d['index'], d['values']), 0, None, n))
    d['index'] = list(new_index)
    d['values'] = np.asarray(new_values, dtype='float64')
    return d
        

def main(args):
    xyn = read_xyn(args.xyn)
    
    if args.polygon:
        P = read_polygon(args.polygon)
    else:
        P = None
    
    xyn = process_stations(xyn, polygon=P, n=args.n)
    write_xyn(args.output, xyn)


def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('xyn', type=pathlib.Path, help="Path to PLI boundary file")
    parser.add_argument('-n', type=int, default=1, help="Decimation factor")
    parser.add_argument('-o', '--output', type=pathlib.Path, help="Output file path")
    parser.add_argument("-c", "--clip", dest="polygon", default=None, type=pathlib.Path, help="Use a polygon to select sites for output")
    args = parser.parse_args()

    if args.output is None:
        out = pathlib.Path()/args.pli.name
        if out.exists():
            raise RuntimeError("-o must be defined")
        else:
            args.output = pathlib.Path()/args.pli.name
    return args


if __name__ == "__main__":
    args = get_options()
    main(args)
