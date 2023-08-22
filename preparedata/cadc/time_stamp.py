""" Extract the time stamp information about each frame
"""
import os
import math
import numpy as np
import itertools
import argparse
import json
from multiprocessing import Process, Pool
import yaml
from easydict import EasyDict
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, default='./data/cadc/',
    help='the location of output information')
parser.add_argument('--data_folder', type=str, help='location of CADC data')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

def get_timestamps(timestamp_path):
    timestamps = []
    with open(timestamp_path) as f:
        for line in f.readlines():
            ts = (pd.to_datetime(line) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1us')
            timestamps.append(ts)
    return timestamps

def main(data_folder, out_folder, split):
    print(f"Creating timestamp info files in {out_folder}")
    for seq in split:
        FILE_NAME = data_folder / seq / 'raw' / 'lidar_points_corrected' / 'timestamps.txt'
        seq_ts = get_timestamps(FILE_NAME)

        file_name = seq.replace("/", "-")
        f = open(os.path.join(out_folder, '{}.json'.format(file_name)), 'w')
        json.dump(seq_ts, f)
        f.close()

# Create timestamps in microseconds (JSON) in `ts_info` folder
if __name__ == '__main__':
    cadc_splits = EasyDict(yaml.safe_load(open(Path(__file__).resolve().parent / "cadc_splits.yaml")))

    # Build path to `output folder`
    if args.test:
        print(args.test)
        args.output_folder=Path(args.output_folder ) / 'validation'
        split = cadc_splits.val
    else:
        args.output_folder=Path(args.output_folder) / 'training'
        split = cadc_splits.train

    args.output_folder = args.output_folder / 'ts_info'

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(Path(args.data_folder), args.output_folder, split)