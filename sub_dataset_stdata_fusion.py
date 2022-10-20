import os
import cv2
import torch
import pickle

import numpy as np

from scipy.interpolate import interp1d
from torch.utils.data import Dataset



# 해야할 것
# Spectrogram cut, Flip, Normalize, Resize, Dynamic range adjust

class prepro_Normalize(object):
    """Frame normalize"""
    def __init__(self, params):
        self.mean_radar = params['mean_radar']
        self.std_radar = params['std_radar']
        self.mean_vidA = params['mean_vidA']
        self.std_vidA = params['std_vidA']
        self.mean_vidM = params['mean_vidM']
        self.std_vidM = params['std_vidM']
    def __call__(self, sample):
        dat_radar, dat_vidA, dat_vidM, gt_resp_diff, gt_label = sample['dat_radar'], sample['dat_vidA'], sample['dat_vidM'], sample['gt_resp_diff'], sample['gt_label']
        # Normalize for Radar data
        for i_chl in range(dat_radar.shape[0]):
            dat_radar[i_chl,:,:] = (dat_radar[i_chl,:,:]-self.mean_radar[i_chl])/self.std_radar[i_chl]
        # Normalize for Video data
        for i_chl in range(dat_vidA.shape[1]):
            dat_vidA[:,i_chl,:,:] = (dat_vidA[:,i_chl,:,:]-self.mean_vidA[i_chl])/self.std_vidA[i_chl]
            dat_vidM[:,i_chl,:,:] = (dat_vidM[:,i_chl,:,:]-self.mean_vidM[i_chl])/self.std_vidM[i_chl]


        return {'dat_radar': dat_radar, 'dat_vidA': dat_vidA, 'dat_vidM': dat_vidM,
                    'gt_resp_diff': gt_resp_diff, 'gt_label': gt_label}

class augment_randFlip(object):
    """Flip the frame/resp randomly in the V/H directions"""
    def __call__(self, sample, flip_T=0.5):
        dat_radar, dat_vidA, dat_vidM, gt_resp_diff, gt_label = sample['dat_radar'], sample['dat_vidA'], sample['dat_vidM'], sample['gt_resp_diff'], sample['gt_label']
        rand_prob = np.random.rand()

        if rand_prob < flip_T:      # flip in temporal domain
            dat_radar = np.flip(dat_radar, axis=2).copy()
            dat_vidA = np.flip(dat_vidA, axis=0).copy()
            dat_vidM = np.flip(dat_vidM, axis=0).copy()
            gt_resp = np.flip(gt_resp).copy()
            gt_resp_diff = np.flip(gt_resp_diff).copy()

        return {'dat_radar': dat_radar, 'dat_vidA': dat_vidA, 'dat_vidM': dat_vidM,
                    'gt_resp_diff': gt_resp_diff, 'gt_label': gt_label}

class Custom_Dataset(Dataset):
    def __init__(self, root_dir, opt, params, set='train', transform=None):     
        self.root_dir = root_dir
        self.train_dir = []
        self.test_dir = []
        self.set_name = set
        self.transform = transform

        for person_name in os.listdir(self.root_dir):
            if opt.test_file in person_name:
                for data_ID in np.arange(1,len(os.listdir(os.path.join(self.root_dir, person_name)))+1):
                    self.test_dir.append(os.path.join(self.root_dir, person_name, f'{str(data_ID).zfill(4)}.pickle'))
            else:
                for data_ID in np.arange(1,len(os.listdir(os.path.join(self.root_dir, person_name)))+1):
                    self.train_dir.append(os.path.join(self.root_dir, person_name, f'{str(data_ID).zfill(4)}.pickle'))
            
    def __len__(self):
        if self.set_name=='train':
            return len(self.train_dir)
        elif self.set_name=='test':
            return len(self.test_dir)

    def __getitem__(self, idx):
        if self.set_name=='train':
            with open(self.train_dir[idx], 'rb') as f:
                data = pickle.load(f)
        elif self.set_name=='test':
            with open(self.test_dir[idx], 'rb') as f:
                data = pickle.load(f)
        dat_radar = data['dat_radar']
        dat_vidA = data['dat_vidA']
        dat_vidM = data['dat_vidM']
        gt_resp_diff = data['GT_valDiff']
        gt_label = data['GT_label']
        
        # Transform
        sample = {'dat_radar': dat_radar, 'dat_vidA': dat_vidA, 'dat_vidM': dat_vidM,
                    'gt_resp_diff': gt_resp_diff, 'gt_label': gt_label}
        if self.transform:
            sample = self.transform(sample)

        dat_radar, dat_vidA, dat_vidM, gt_resp_diff, gt_label = sample['dat_radar'], sample['dat_vidA'], sample['dat_vidM'], sample['gt_resp_diff'], sample['gt_label']
        dat_radar = torch.from_numpy(dat_radar)   # numpy to torch tensor
        dat_vidA = torch.from_numpy(dat_vidA)
        dat_vidM = torch.from_numpy(dat_vidM)
        

        return dat_radar, dat_vidA, dat_vidM, gt_resp_diff, gt_label
