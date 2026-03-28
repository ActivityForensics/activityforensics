import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from IPython import embed

@register_dataset("vidf")
class VidF(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        **kwargs,
    ):
        # file path
        # embed()
        # assert os.path.exists(feat_folder) and os.path.exists(json_file)
        # assert isinstance(split, tuple) or isinstance(split, list)
        # assert crop_ratio == None or len(crop_ratio) == 2
        # self.feat_folder = feat_folder
        # if file_prefix is not None:
        #     self.file_prefix = file_prefix
        # else:
        #     self.file_prefix = ''
        # self.file_ext = file_ext
        # self.json_file = json_file
        #
        # # split / training mode
        # self.split = split
        # self.is_training = is_training
        #
        # # features meta info
        # self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        # self.default_fps = default_fps
        # self.downsample_rate = downsample_rate
        # self.max_seq_len = max_seq_len
        # self.trunc_thresh = trunc_thresh
        # self.num_classes = num_classes
        # self.label_dict = None
        # self.crop_ratio = crop_ratio
        #
        # # load database and select the subset
        # dict_db, label_dict = self._load_json_db(self.json_file)
        # assert len(label_dict) == num_classes
        # self.data_list = dict_db
        # self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'vidf',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            'empty_label_ids': [],
        }


        self.version = kwargs['version']

        self.data_dir = f'/home/users/xxx/scratch/dataset/vidf/{self.version}'
        assert os.path.exists(self.data_dir), 'Please specify data_dir'
        self.split = split
        if isinstance(self.split, str):
            self.split = [self.split]

        annotations = []
        self.split = [s for s in self.split if "real" not in s] + [s for s in self.split if "real" in s]
        for split_itm in self.split:
            anno_file = open(
                os.path.join(
                    self.data_dir,
                    "{}.txt".format(split_itm)
                ), 'r'
            )
            line_cnt = -1
            tmp_annotations = []
            for line in anno_file:
                line_cnt += 1
                # anno = line.split("##")[0]
                # sent = sent.split('.\n')[0]
                anno = line
                if 'real' in split_itm:
                    vid, duration = anno.split(" ")
                    duration = float(duration)
                    pairs = []
                else:
                    vid, duration, time_str = anno.split(" ")
                    duration = float(duration)
                    time_str = time_str.replace('\n', '')
                    pairs = [x.split('=') for x in time_str.split('+')]
                time_list = []
                start_list = []
                end_list = []
                for p in pairs:
                    # Check format
                    assert len(p) == 2, f"Invalid format: '{'='.join(p)}' is not in start=end format"

                    start_str, end_str = p
                    # Convert to float and assert valid
                    start = float(start_str)
                    end = min(float(end_str), duration)

                    time_list.append([start, end])
                    start_list.append(start)
                    end_list.append(end)
                # if line_cnt % 2 == 1:
                #     time_list = []
                #     annotations.append(
                #         {'video': vid, 'times': time_list, 'duration': duration})
                #     continue

                tmp_annotations.append(
                    {'video': vid, 'times': time_list, 'duration': duration})
            anno_file.close()
            assert 'real' not in split_itm
            if 'real' in split_itm:
                tmp_annotations_num_1 = int(len(tmp_annotations) * kwargs['real_ratio'])
                tmp_annotations_num_2 = int(len(annotations))
                tmp_annotations_num = min(tmp_annotations_num_1, tmp_annotations_num_2)
                tmp_annotations = tmp_annotations[:tmp_annotations_num]
            annotations += tmp_annotations

        if 'train' in split[0]:
            annot_num = kwargs['train_annot_num']
        else:
            annot_num = kwargs['test_annot_num']
        if annot_num > 0:
            indices = np.linspace(0, len(annotations) - 1, annot_num, dtype=int)
            annotations = [annotations[i] for i in indices]

        self.annotations = annotations

        self.feature_type = 'clipL14'

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        # Example dimensions
        C = self.input_dim  # feature channels

        # Generate feats: C x T
        video_id = self.annotations[idx]['video'].split('.mp4')[0]
        visual_input = self.get_video_features(video_id)

        def average_to_fixed_length(visual_input, num_sample_clips):
            num_clips = visual_input.shape[0]
            idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips
            idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
            new_visual_input = []
            for i in range(num_sample_clips):
                s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
                if s_idx < e_idx:
                    new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
                else:
                    new_visual_input.append(visual_input[s_idx])
            new_visual_input = torch.stack(new_visual_input, dim=0)
            return new_visual_input

        visual_input = average_to_fixed_length(visual_input, self.num_frames)
        feats = visual_input.permute(1, 0)

        times = torch.tensor(self.annotations[idx]['times'])  # (N, 2)
        N = times.shape[0]

        starts = times[:, 0] / self.annotations[idx]['duration'] * self.num_frames
        ends = times[:, 1] / self.annotations[idx]['duration'] * self.num_frames

        segments = torch.stack([starts, ends], dim=1)

        labels = torch.zeros((N,)).long()

        data_dict = {'video_id'        : str(idx),
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'feat_num_frames' : self.num_frames,
                     'duration' : self.annotations[idx]['duration'],
                     'gt_time' : self.annotations[idx]['times'],
                     }

        return data_dict


    def get_video_features(self, vid):
        if 'clipL14' in self.feature_type:
            features = np.load(os.path.join(self.data_dir, f'../feat/01a.2a_L14/{vid}.npy'))
            features = torch.from_numpy(features).float()
        return features
