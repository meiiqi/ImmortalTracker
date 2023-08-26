import os, numpy as np, argparse, json, sys, numba, yaml, multiprocessing, shutil
import mot_3d.visualization as visualization, mot_3d.utils as utils
from mot_3d.data_protos import BBox, Validity
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData
from data_loader import CADCLoader
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
# running configurations
parser.add_argument('--name', type=str, default='immortal')
parser.add_argument('--det_name', type=str, default='cadc_lisnownet_5f_pretrained')
parser.add_argument('--obj_type', type=str, default='Car', choices=['Car'])
parser.add_argument('--start_frame', type=int, default=0, help='start at a middle frame for debug')
# paths
parser.add_argument('--config_path', type=str, default='configs/cadc_configs/immortal.yaml')
parser.add_argument('--data_folder', type=str, default='./data/cadc')
parser.add_argument('--det_folder', type=str, default='./data/cadc')
parser.add_argument('--result_folder', type=str, default='./mot_results/cadc')
parser.add_argument('--summary_folder', type=str, default='./mot_results/cadc')
args = parser.parse_args()


def sequence_mot(configs, data_loader: CADCLoader, sequence_id, gt_bboxes=None, gt_ids=None):
    tracker = MOTModel(configs)
    frame_num = len(data_loader)
    IDs, bboxes, states, types = list(), list(), list(), list()
    for frame_index in range(data_loader.cur_frame, frame_num):
        frame_data = next(data_loader)
        frame_data = FrameData(dets=frame_data['dets'], ego=frame_data['ego'], pc=frame_data['pc'],
            det_types=frame_data['det_types'], aux_info=frame_data['aux_info'], time_stamp=frame_data['time_stamp'])

        # mot
        results = tracker.frame_mot(frame_data)
        result_pred_bboxes = [trk[0] for trk in results]
        result_pred_ids = [trk[1] for trk in results]
        result_pred_states = [trk[2] for trk in results]
        result_types = [trk[3] for trk in results]

        # wrap for output
        IDs.append(result_pred_ids)
        result_pred_bboxes = [BBox.bbox2array(bbox) for bbox in result_pred_bboxes]
        bboxes.append(result_pred_bboxes)
        states.append(result_pred_states)
        types.append(result_types)

    return IDs, bboxes, states, types


def main(data_folder, det_folder, result_folder, summary_folder, config_path, obj_type, start_frame):
    configs = yaml.load(open(config_path, 'r'))

    # Only supporting 'Car' class for now
    if obj_type == 'Car':
        type_token = 1

    # Get sequence names
    file_names = sorted(os.listdir(os.path.join(data_folder, 'ego_info')))

    # Iterate over sequences
    for file_index, file_name in enumerate(file_names):
        seq_name = file_name.replace(".npz", "")

        out_file = summary_folder / file_name
        print(f"Saving MOT output to {out_file}")

        # import pdb
        # pdb.set_trace()

        data_loader = CADCLoader(configs, [type_token], seq_name, data_folder, det_folder, start_frame)

        # import pdb
        # pdb.set_trace()

        ids, bboxes, states, types = sequence_mot(configs, data_loader, file_index)
        # pdb.set_trace()

        np.savez_compressed(out_file,
            ids=ids, bboxes=bboxes, states=states)


if __name__ == '__main__':
    args.data_folder = Path(args.data_folder)
    args.det_folder = Path(args.det_folder)
    args.result_folder = Path(args.result_folder)
    args.summary_folder = Path(args.summary_folder)

    args.data_folder = args.data_folder / 'validation'
    args.det_folder = args.det_folder / 'validation' / 'detection' / args.det_name
    args.result_folder = args.result_folder / 'validation' / args.name
    args.summary_folder = args.result_folder / 'summary' / args.obj_type

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    if not os.path.exists(args.summary_folder):
        os.makedirs(args.summary_folder)

    main(args.data_folder, args.det_folder, args.result_folder, args.summary_folder, args.config_path,  args.obj_type, args.start_frame)