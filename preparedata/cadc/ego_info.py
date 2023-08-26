""" Extract the ego location information
    Output file format: dict compressed in .npz files
    {
        st(frame_num): ego_info (4 * 4 matrix)
    }
"""

import argparse
import math
import numpy as np
import json
import os
import multiprocessing
import yaml
from easydict import EasyDict
from pathlib import Path
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, help='location of tfrecords')
parser.add_argument('--output_folder', type=str, default='./data/cadc/',
    help='output folder')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

def get_seq_infos(infos):
    ''' Reformat the sequence infos to be easily accessible by the sequence name.
    '''
    seq_infos_dict = {}
    for info in infos:
        seq_name = f"{info['date']}-{info['drive']}"
        seq_infos_dict.setdefault(seq_name, []).append(info)

    seq_names = list(seq_infos_dict.keys())

    return seq_names, seq_infos_dict

def main(output_folder, infos):
    print(f"Creating timestamp info files in {output_folder}")

    seq_names, seq_infos_dict = get_seq_infos(infos)

    # Iterate over sequences
    for seq in seq_names:
        seq_infos = seq_infos_dict[seq]

        # Iterate over labeled frames
        ego_infos = {}
        first_frame = seq_infos[0]
        starts_with_sweeps = (first_frame['sweeps'][0]['pose'] is not None) and (first_frame['sweeps'][1]['pose'] is not None)
        start_idx = 2 if starts_with_sweeps else 0

        for i, info in enumerate(seq_infos):
            frame_num = start_idx + 3 * i

            if starts_with_sweeps or (i > 0):
                assert info['sweeps'][0]['time_lag'] < info['sweeps'][1]['time_lag']

                # Iterate over unlabeled frames (2 prior sweeps)
                for j, sweep_info in reversed(list(enumerate(info['sweeps']))):
                    sweep_frame_num = str(frame_num - (j + 1))
                    ego_infos[sweep_frame_num] = sweep_info['pose']

            ego_infos[str(frame_num)] = info['pose']

        assert list(ego_infos.keys())[0] == '0', 'not all sequences start from frame 0'

        fname = output_folder / f"{seq}.npz"
        np.savez_compressed(fname, **ego_infos)


if __name__ == '__main__':
    cadc_splits = EasyDict(yaml.safe_load(open(Path(__file__).resolve().parent / "cadc_splits.yaml")))

    args.data_folder = Path(args.data_folder)
    args.output_folder = Path(args.output_folder)

    # Build path to `output folder`
    if args.test:
        args.output_folder = args.output_folder / 'validation'
        info_path = args.data_folder / 'cadc_lisnownet_infos_3sweeps_val.normalized.pkl'
        split = cadc_splits.val
    else:
        args.output_folder = args.output_folder / 'training'
        info_path = args.data_folder / 'cadc_lisnownet_infos_3sweeps_train.normalized.pkl'
        split = cadc_splits.train

    args.output_folder = args.output_folder / 'ego_info'

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    main(args.output_folder, infos)
