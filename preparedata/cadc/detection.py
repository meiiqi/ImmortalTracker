""" Extract the detections from results.pkl file
"""
import os, math, numpy as np, itertools, argparse, json
import yaml
from easydict import EasyDict
from pathlib import Path
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='cadc_lisnownet_5f_pretrained')
parser.add_argument('--data_folder', type=str)
parser.add_argument('--detection_file', type=str, default='cadc_lisnownet_5f_pretrained_result.pkl', help= 'Detection results file name')
parser.add_argument('--det_folder', type=str, default='./data/cadc/')
parser.add_argument('--output_folder', type=str, default='./data/cadc/',
    help='output folder to store converted detections')
args = parser.parse_args()

def get_seq_results(results):
    ''' Reformat the sequence results to be easily accessible by the sequence name.
    '''
    frame_indices = {}
    seq_results_dict = {}
    for res in results:
        date, drive, frame_idx = res['frame_id'].split("/")
        seq_name = f"{date}-{drive}"
        seq_results_dict.setdefault(seq_name, []).append(res)
        frame_indices.setdefault(seq_name, []).append(int(frame_idx))

    seq_names = list(seq_results_dict.keys())

    # Ensure frames are in order
    for seq in seq_names:
        indices = np.asarray(frame_indices[seq])
        sorted_indices = np.sort(indices)
        assert (sorted_indices == indices).all()

    return seq_names, seq_results_dict

def create_bboxes(frame_result):
    bbox_list = []
    for i, box in enumerate(frame_result['boxes_lidar']):
        x,y,z,l,w,h,yaw = box
        score = frame_result['score'][i]
        bbox_list.append(np.array([x, y, z, yaw, l, w, h, score]))

    return bbox_list


def main(detection_file, out_folder):
    with open(detection_file, 'rb') as f:
        results = pickle.load(f)

    seq_names, seq_results_dict = get_seq_results(results)

    # Iterate over sequences
    for seq in seq_names:
        frame_results = seq_results_dict[seq]

        bboxes = []
        obj_types = []

        # Iterate over frames
        for frame_result in frame_results:
            bboxes.append(create_bboxes(frame_result))
            obj_types.append(frame_result['pred_labels'])

        bboxes = np.asarray(bboxes, dtype=object)
        obj_types = np.asarray(obj_types, dtype=object)

        # Save newly formatted results
        result = {'bboxes': bboxes, 'types': obj_types}
        out_file = out_folder / f"{seq}.npz"
        np.savez_compressed(out_file, **result)


if __name__ == '__main__':
    cadc_splits = EasyDict(yaml.safe_load(open(Path(__file__).resolve().parent / "cadc_splits.yaml")))

    args.data_folder = Path(args.data_folder)
    args.output_folder = Path(args.output_folder)
    args.det_folder = Path(args.det_folder)
    args.detection_file = args.data_folder / args.detection_file

    # Build path to `output folder`
    args.det_folder = args.det_folder / 'validation' / 'detection'
    args.output_folder = args.det_folder / args.name / 'dets'

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    main(args.detection_file, args.output_folder)