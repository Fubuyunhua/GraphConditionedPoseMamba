import torch
import numpy as np
import glob
import os
import io
import random
import pickle
from torch.utils.data import Dataset, DataLoader
from lib.data.augmentation import Augmenter3D
from lib.utils.tools import read_pkl
from lib.utils.utils_data import flip_data
    
class MotionDataset(Dataset):
    def __init__(self, args, subset_list, data_split): # data_split: train/test
        np.random.seed(0)
        self.data_root = args.data_root
        self.subset_list = subset_list
        self.data_split = data_split
        file_list_all = []
        for subset in self.subset_list:
            data_path = os.path.join(self.data_root, subset, self.data_split)
            motion_list = sorted(os.listdir(data_path))
            for i in motion_list:
                file_list_all.append(os.path.join(data_path, i))
        self.file_list = file_list_all
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_list)

    def __getitem__(self, index):
        raise NotImplementedError 

class MotionDataset3D(MotionDataset):
    def __init__(self, args, subset_list, data_split):
        super(MotionDataset3D, self).__init__(args, subset_list, data_split)
        self.flip = args.flip
        self.synthetic = args.synthetic
        self.aug = Augmenter3D(args)
        self.gt_2d = args.gt_2d
        # Evaluation-only robustness hook.
        # Randomly drop a ratio of joints per frame from the 2D input during test time.
        self.eval_joint_dropout_ratio = float(getattr(args, 'eval_joint_dropout_ratio', 0.0))
        self.eval_joint_dropout_mode = getattr(args, 'eval_joint_dropout_mode', 'zero_conf')
        self.eval_joint_dropout_seed = int(getattr(args, 'eval_joint_dropout_seed', 1234))
        self.eval_joint_dropout_block_len = int(getattr(args, 'eval_joint_dropout_block_len', 1))

    def _build_eval_dropout_mask(self, motion_2d, index):
        """Construct a deterministic evaluation-time joint-drop mask.

        The default mode is i.i.d. frame-joint dropout. When block_len > 1, the
        mask becomes temporally contiguous, which better matches short occlusion
        intervals produced by real 2D detectors.
        """
        T, J = motion_2d.shape[:2]
        rng = np.random.RandomState(self.eval_joint_dropout_seed + index)
        block_len = max(1, self.eval_joint_dropout_block_len)
        if block_len == 1:
            return rng.rand(T, J) < self.eval_joint_dropout_ratio

        mask = np.zeros((T, J), dtype=bool)
        target = int(round(T * self.eval_joint_dropout_ratio))
        if target <= 0:
            return mask
        for joint_idx in range(J):
            dropped = 0
            max_trials = max(8, 4 * target)
            trials = 0
            while dropped < target and trials < max_trials:
                start = rng.randint(0, max(1, T - block_len + 1))
                end = min(T, start + block_len)
                before = mask[:, joint_idx].sum()
                mask[start:end, joint_idx] = True
                dropped = int(mask[:, joint_idx].sum())
                if int(before) == dropped:
                    trials += 1
                else:
                    trials = 0
            if dropped < target:
                remaining = np.where(~mask[:, joint_idx])[0]
                if len(remaining) > 0:
                    extra = remaining[: max(0, target - dropped)]
                    mask[extra, joint_idx] = True
        return mask

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file_path = self.file_list[index]
        motion_file = read_pkl(file_path)
        motion_3d = motion_file["data_label"]  
        if self.data_split=="train":
            if self.synthetic or self.gt_2d:
                motion_3d = self.aug.augment3D(motion_3d)
                motion_2d = np.zeros(motion_3d.shape, dtype=np.float32)
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1                        # No 2D detection, use GT xy and c=1.
            elif motion_file["data_input"] is not None:     # Have 2D detection 
                motion_2d = motion_file["data_input"]
                if self.flip and random.random() > 0.5:                        # Training augmentation - random flipping
                    motion_2d = flip_data(motion_2d)
                    motion_3d = flip_data(motion_3d)
            else:
                raise ValueError('Training illegal.') 
        elif self.data_split=="test":                                           
            motion_2d = motion_file["data_input"]
            if self.gt_2d:
                motion_2d[:,:,:2] = motion_3d[:,:,:2]
                motion_2d[:,:,2] = 1
            if self.eval_joint_dropout_ratio > 0:
                motion_2d = motion_2d.copy()
                drop_mask = self._build_eval_dropout_mask(motion_2d, index)
                if self.eval_joint_dropout_mode == "zero_conf":
                    motion_2d[drop_mask, :2] = 0.0
                    if motion_2d.shape[-1] > 2:
                        motion_2d[drop_mask, 2] = 0.0
                elif self.eval_joint_dropout_mode == "conf_only":
                    if motion_2d.shape[-1] > 2:
                        motion_2d[drop_mask, 2] = 0.0
                else:
                    raise ValueError(f"Unknown eval_joint_dropout_mode: {self.eval_joint_dropout_mode}")
        else:
            raise ValueError('Data split unknown.')    
        return torch.FloatTensor(motion_2d), torch.FloatTensor(motion_3d)