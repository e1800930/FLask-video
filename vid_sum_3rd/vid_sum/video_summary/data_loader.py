import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import numpy as np
from video_summary.fragments import compute_fragments

class VideoData(Dataset):
    def __init__(self, mode, filepath, action_state_size):
        self.mode = mode
        self.filename = filepath

        hdf = h5py.File(self.filename, 'r')
        self.keys = list(hdf)

        self.action_fragments = {}
        self.list_features = []

        for key in self.keys:
            features = torch.Tensor(np.array(hdf[key + '/features']))
            self.list_features.append(features)
            self.action_fragments[key] = compute_fragments(features.shape[0], action_state_size)

        hdf.close()

    def __len__(self):
        self.len = len(self.keys)
        return self.len

    # In "train" mode it returns the features and the action_fragments
    # In "test" mode it also returns the video_name
    def __getitem__(self, index):
        video_name = self.keys[index]  # gets the current video name
        frame_features = self.list_features[index]

        if self.mode == 'test':
            return frame_features, video_name, self.action_fragments[video_name]
        else:
            return frame_features, self.action_fragments[video_name]

def get_loader(mode, save_path, action_state_size):
    vd = VideoData(mode, save_path, action_state_size)
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return vd