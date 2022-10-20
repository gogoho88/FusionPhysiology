"""
- Training/Inference Stationary target respiration based on FMCW radar signal
- v1 (210624)
"""
import argparse
import os
import copy
import time
import pickle   
import shutil   
import yaml
import h5py
import shutil
# import librosa
import datetime

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from sub_dataload_stdata_fusion import data_formation_Fusion
from sub_dataset_stdata_fusion import Custom_Dataset, prepro_Normalize, augment_randFlip
from sub_utils import AverageMeter, ProgressResult
from model import model_form



def get_args():
    parser = argparse.ArgumentParser(description = 'Training/Inference multi-human respiration from radar signals')
    parser.add_argument('-p', '--project', type=str, default='Fusion_st', help='project file that contains parameters')
    # deep learning hyper-parameter
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images per batch among all devices')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers of dataloader')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    # test parameter
    parser.add_argument('--frame_length', type=float, default=1, help='Length of Processing Frame [s]')
    parser.add_argument('--test_file', type=str, default='JJH_st_1', help='Name of test file')
    # model parameter
    parser.add_argument('--loss_type_resp', type=str, default='l1')
    parser.add_argument('--fold_div', type=int, default=8, help='division coefficient for TSM module')
    parser.add_argument('--fusion_type', type=str, default='fusion')
    parser.add_argument('--fusion_mode', type=str, default='transformer')

    # other parameter (e.g., validation interval, save, ...)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between validation phases')
    parser.add_argument('--flag_saveData', type=int, default=False, help='True: save the preprocessed data')
    parser.add_argument('--flag_saveResult', type=int, default=False, help='True: save the output results')
    
    args = parser.parse_args()
    return args

def nPCC_loss(target, output, eps=1e-6):
    target_mean = target.mean(axis=1).view(-1,1)
    output_mean = output.mean(axis=1).view(-1,1)
    Pcc = torch.sum((target-target_mean)*(output-output_mean),dim=1) / \
        ((torch.sqrt(torch.sum((target-target_mean)**2,dim=1)+eps))*(torch.sqrt(torch.sum((output-output_mean)**2,dim=1)+eps)))
    ccc = 1-Pcc
    return torch.mean(ccc)

def PCC(target, output, eps=1e-6):
    target_mean = np.mean(target)
    output_mean = np.mean(output)
    Pcc = np.sum((target-target_mean)*(output-output_mean)) / \
        ((np.sqrt(np.sum((target-target_mean)**2)+eps))*(np.sqrt(np.sum((output-output_mean)**2)+eps)))
    return Pcc

def estimate_freq_psd(data_resp, label, vec_s):
    # estimate frequency from sinosoidal signal using periodogram
    vec_len = len(data_resp)
    sos = signal.butter(2, [0.08, 0.6], 'bandpass', fs=vec_len/vec_s, output='sos') # band pass filter
    label_temp = label
    data_resp = np.cumsum(data_resp)
    data_temp = data_resp - np.mean(data_resp)
    if label_temp==0:
        if np.mean(np.abs(data_temp))<=0.3:
            resp_freq = 0
        else:
            data_filter = signal.sosfilt(sos,data_temp)
            [f,Sf] = signal.periodogram(data_filter,fs=vec_len/vec_s,nfft=2048)
            resp_freq = f[np.argmax(Sf)]
    else:            
        data_filter = signal.sosfilt(sos,data_temp)
        [f,Sf] = signal.periodogram(data_filter,fs=vec_len/vec_s,nfft=2048)
        resp_freq = f[np.argmax(Sf)]
    return resp_freq


def train(opt, params):
    dir_dataset = os.path.join(params['data_save_root'],opt.project,'data')
    opt.test_file = 'None'
    train_set = Custom_Dataset(dir_dataset, opt, params, set='train', 
                                    transform=transforms.Compose([
                                                                    prepro_Normalize(params),
                                                                    # augment_randFlip()
                                                                    ]))
    test_set = Custom_Dataset(dir_dataset, opt, params, set='test', 
                                    transform=transforms.Compose([
                                                                    prepro_Normalize(params),
                                                                    ]))
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,  num_workers=opt.num_workers)

    freq_vec = []

    progress_bar = tqdm(train_loader)
    for iter, data in enumerate(progress_bar):
        X_radar, X_vidA, X_vidM, Y_respDiff, Y_label = data
        X_radar = Variable(X_radar.float().cuda())
        X_vidA = Variable(X_vidA.float().cuda())
        X_vidM = Variable(X_vidM.float().cuda())
        Y_respDiff = Variable(Y_respDiff.float().cuda())
        Y_label = Variable(Y_label.long().cuda())

        GT_label = np.array(torch.squeeze(Y_label).to('cpu'))
        GT_respDiff = np.array(torch.squeeze(Y_respDiff).to('cpu'))
        freq_GT = estimate_freq_psd(GT_respDiff, np.round(np.mean(GT_label)), vec_s=params['Frame_s'])*60

        freq_vec.append(freq_GT)

    mean_BPM = np.array(freq_vec).mean()
    std_BPM = np.array(freq_vec).std()

    a = 1

if __name__ == '__main__':
    current_dir = '/'.join(os.path.realpath(__file__).split('/')[:-1])       
    opt = get_args()
    # Data load
    if opt.project=='Fusion_st':
        params = yaml.safe_load(open(os.path.join(current_dir, f'projects/{opt.project}.yml')))
        data_list = data_formation_Fusion(opt, params, flag_save=opt.flag_saveData)
    else:
        params = yaml.safe_load(open(os.path.join(current_dir, f'dataset/{opt.project}/code/Metalog_data.yml')))

    train(opt, params)






