"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data
import cv2
from coviar import get_num_frames
from coviar import load
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from transforms import color_aug
GOP_SIZE = 12


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg + 1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frmae.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self, data_root,
                 dataset,
                 video_list,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._dataset = dataset
        self._num_segments = num_segments
        self._transform = transform

        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    def _load_list(self, video_list):
        self._video_list = []
        if self._dataset == 'ucf101': # for ucf
            with open(video_list, 'r') as f:
                for line in f:
                    video, _, label = line.strip().split()
                    video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                    self._video_list.append((
                        video_path,
                        int(label),
                        get_num_frames(video_path)))

        if self._dataset == 'kinetics400':  # for kinetics
            with open(video_list, 'r') as f:
                for line in f:
                    video, label,num_frames = line.strip().split(',')
                    video_path = os.path.join(self._data_root, video)
                    self._video_list.append((
                        video_path,
                        int(label),
                        int(num_frames)))

        print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg ,representation):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, representation)

    def _get_test_frame_index(self, num_frames, seg, representation):
        if representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, representation)

    def __getitem__(self, index):
        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
        else:
            video_path, label, num_frames = self._video_list[index]

        iframes = []
        mvs = []
        for seg in range(self._num_segments):

            #For I-frame
            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg,'iframe')
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,'iframe')

            img = load(video_path, gop_index, gop_pos, 0, self._accumulate)
            if img is None:
                print('Error: loading video iframe %s failed.' % video_path)
                img = np.zeros((256, 256, 3))
            img = color_aug(img)
            # BGR to RGB. (PyTorch uses RGB according to doc.)
            img = img[..., ::-1]

            # For MV
            # if self._is_train:
            #     gop_index, gop_pos = self._get_train_frame_index(num_frames, seg,'mv')
            # else:
            #     gop_index, gop_pos = self._get_test_frame_index(num_frames, seg,'mv')
            mv = load(video_path, gop_index, 11, 1, self._accumulate) #use the last pos of every gop
            if mv is None:
                print('Error: loading video mv %s failed.' % video_path)
                mv = np.zeros((256, 256, 2))

            mv = clip_and_scale(mv, 20)
            mv += 128
            mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)

            iframes.append(img)
            mvs.append(mv)

        iframes,mvs = self._transform((iframes,mvs))

        iframes = np.array(iframes)
        iframes = np.transpose(iframes, (0, 3, 1, 2))
        iframes = torch.from_numpy(iframes).float() / 255.0
        iframes = (iframes-self._input_mean) / self._input_std

        mvs = np.array(mvs)
        mvs = np.transpose(mvs,(0, 3, 1, 2))
        mvs = torch.from_numpy(mvs).float() / 255.0
        mvs = mvs-0.5

        # (iframes,mvs),label (depth, channel,width,height)
        return (iframes, mvs), label

    def __len__(self):
        return len(self._video_list)


def visualize_mv(mat):
    # Use Hue, Saturation, Value colour model
    mat = mat.astype(np.float32)
    w = mat.shape[0]
    h = mat.shape[1]
    hsv = np.zeros((w, h, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(mat[..., 0], mat[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt.imshow(bgr_frame)
    plt.show()