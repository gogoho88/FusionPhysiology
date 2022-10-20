import os
import cv2
import h5py
import time
import pickle
import yaml
import shutil
import datetime

import numpy as np

from scipy import signal
from scipy import fft
from scipy.interpolate import interp1d

def form_spectrogram(signal_temp, PRF, stft_wseg, nfft_num, flag_DB, crop_ratio, resize_dim_T, resize_dim_F):
    win_size = int(PRF*stft_wseg)
    over_size = int(win_size*0.9)
    nfft_stft = nfft_num
    # form spectrogram
    _, _, Sxx = signal.spectrogram(signal_temp, PRF, window='hamming', nperseg=win_size, noverlap=over_size, nfft=nfft_stft, 
                                    return_onesided=False, mode='psd')
    Sxx = fft.fftshift(Sxx, axes=0)
    # log scale
    if flag_DB==1:
        Sxx = np.log10(Sxx)
    # spectrum cut
    cut_len = int((crop_ratio/2)*Sxx.shape[0])
    Sxx = Sxx[cut_len:-cut_len,:]
    # Reject outliers
    Sxx_sort = np.sort(Sxx.flatten())
    dat_min = Sxx_sort[int(len(Sxx_sort)*0.005)]
    dat_max = Sxx_sort[int(len(Sxx_sort)*0.995)]
    Sxx[np.where(Sxx<=dat_min)] = dat_min
    Sxx[np.where(Sxx>=dat_max)] = dat_max
    # 2D resize
    if resize_dim_T==0:
        Sxx_intr = Sxx
    else:
        Sxx_intr = cv2.resize(Sxx, (resize_dim_T, resize_dim_F), interpolation=cv2.INTER_LINEAR)
    return Sxx_intr

def data_formation_Fusion(opt, params, flag_save=False):
    # Load File
    data_path = params['data_root']
    data_list_all = os.listdir(data_path)
    data_list_radar = [file for file in data_list_all if "fusion_Radar" in file]
    data_list_video = [f"{('_').join(file.split('_')[:-2])}_Video_v1.mat" for file in data_list_radar]
    len_data_list = len(data_list_radar)
    
    for ind_data in range(len_data_list):   
        # ind_data = 7
        data_path_radar = os.path.join(data_path, data_list_radar[ind_data])
        data_path_video = os.path.join(data_path, data_list_video[ind_data])
        # Load Radar Data
        f = h5py.File(data_path_radar, 'r')
        dat_radar_raw = []
        for ind_chl in range(params['N_Rx']):
            data = np.array(f[f['data_FRAME'][ind_chl][0]])
            dat_radar_raw.append(data)
        # Load GT Data 
        GT_val_raw = np.array(f[f['data_FRAME'][0][1]])[0]
        GT_label_raw = np.array(f[f['data_FRAME'][0][2]])[0]
        # Load Video Data
        f = h5py.File(data_path_video, 'r')
        dat_video_raw = np.array(f[f['data_FRAME'][0][0]])
        if np.sum(np.isnan(dat_video_raw)):
            raise Exception('Nan values exist')

        len_radar = dat_radar_raw[0].shape[1]
        len_video = dat_video_raw.shape[0]

        # Pre-processing
        dat_video_M = np.zeros((len_video-1,3,params['img_resize'],params['img_resize']),dtype='float32')
        dat_video_A = np.zeros((len_video-1,3,params['img_resize'],params['img_resize']),dtype='float32')
        # Video pre-processing
        for ind_img in range(len_video-1):
            img_t1 = dat_video_raw[ind_img,:,:,:].transpose(1,2,0).astype(np.float)
            img_t2 = dat_video_raw[ind_img+1,:,:,:].transpose(1,2,0).astype(np.float)
            img_t1_resize = cv2.resize(img_t1, (params['img_resize'],params['img_resize']), interpolation=cv2.INTER_CUBIC)
            img_t2_resize = cv2.resize(img_t2, (params['img_resize'],params['img_resize']), interpolation=cv2.INTER_CUBIC)
            img_t1_resize = cv2.rotate(img_t1_resize, cv2.ROTATE_90_CLOCKWISE).transpose(2,0,1)
            img_t2_resize = cv2.rotate(img_t2_resize, cv2.ROTATE_90_CLOCKWISE).transpose(2,0,1)
            img_M = (img_t2_resize-img_t1_resize)/(img_t2_resize+img_t1_resize+1e-6)
            img_M = np.where(img_M<=3, img_M, 3.)
            img_M = np.where(img_M>=-3, img_M, -3.)
            dat_video_M[ind_img,:,:,:] = img_M
            dat_video_A[ind_img,:,:,:] = img_t2_resize/255.
        # GT pre-processing
        f_val = interp1d(np.linspace(0,len_radar-1,len_radar),GT_val_raw, kind='linear')
        f_label = interp1d(np.linspace(0,len_radar-1,len_radar),GT_label_raw, kind='linear')
        GT_val_resize = f_val(np.linspace(0,len_radar-1,len_video))
        GT_label_resize = f_label(np.linspace(0,len_radar-1,len_video))
        GT_val_norm = (GT_val_resize-np.mean(GT_val_resize))/np.std(GT_val_resize)
        GT_val_tanh = np.tanh(GT_val_norm)
        GT_val_diff = np.diff(GT_val_tanh)
        GT_val_diff = (GT_val_diff-np.mean(GT_val_diff))/np.std(GT_val_diff)
        GT_label_diff = GT_label_resize[1:]
        
        # make data save folder
        if opt.flag_saveData:
            if ind_data==0:
                time_now = datetime.datetime.now()
                folder_name = time_now.strftime('[%Y-%m-%d]%H%M%S')
                save_path_folder = os.path.join(params['data_save_root'],folder_name)
                os.makedirs(save_path_folder, exist_ok=True)
                os.makedirs(os.path.join(save_path_folder,'data'), exist_ok=True)

        ind_frame_PRF = 0
        ind_frame_FPS = 0
        ind_save = 0
        while True:
            if (ind_frame_PRF+params['PRF']*params['Frame_s']>len_radar) or (ind_frame_FPS+params['FPS']*params['Frame_s']>len_video):
                break
            # Radar windowing & pre-processing
            frame_radar_preprocess = np.zeros((2*len(dat_radar_raw),params['STFT_resize_F'],params['STFT_resize_T']), dtype='float32')
            for ind_chl in range(len(dat_radar_raw)):
                frame_radar = dat_radar_raw[ind_chl][:,ind_frame_PRF:ind_frame_PRF+params['PRF']*params['Frame_s']]
                frame_radar = frame_radar['real']+1j*frame_radar['imag']
                if params['Clut_mode']==1:      # Clutter Removal
                    frame_radar_clut = frame_radar-np.repeat(np.expand_dims(frame_radar.mean(axis=1),axis=1),frame_radar.shape[1],axis=1)
                else:
                    frame_radar_clut = frame_radar
                frame_radar_proj = np.sum(frame_radar_clut, axis=0)     # Projection in Range
                frame_radar_abs = np.abs(frame_radar_proj)                  # Abs
                frame_radar_phase = np.exp(1j*np.angle(frame_radar_proj))   # Phase
                STFT_abs = form_spectrogram(frame_radar_abs, params['PRF'], params['STFT_W'], params['STFT_nfft'], 
                                            params['STFT_DB'], params['STFT_Fcut'], params['STFT_resize_T'], params['STFT_resize_F'])   # abs STFT
                STFT_phase = form_spectrogram(frame_radar_phase, params['PRF'], params['STFT_W'], params['STFT_nfft'], 
                                            params['STFT_DB'], params['STFT_Fcut'], params['STFT_resize_T'], params['STFT_resize_F'])   # phase STFT
                frame_radar_preprocess[2*ind_chl,:,:] = STFT_abs
                frame_radar_preprocess[2*ind_chl+1,:,:] = STFT_phase
            # Video windowing
            frame_video_A = dat_video_A[ind_frame_FPS:ind_frame_FPS+params['FPS']*params['Frame_s'],:,:,:]
            frame_video_M = dat_video_M[ind_frame_FPS:ind_frame_FPS+params['FPS']*params['Frame_s'],:,:,:]
            # GT windowing
            frame_GT_val_diff = GT_val_diff[ind_frame_FPS:ind_frame_FPS+params['FPS']*params['Frame_s']]
            frame_GT_label = np.mean(GT_label_diff[ind_frame_FPS:ind_frame_FPS+params['FPS']*params['Frame_s']])
            # stride index
            ind_frame_PRF = int(ind_frame_PRF+params['Frame_stride']*params['PRF'])
            ind_frame_FPS = int(ind_frame_FPS+params['Frame_stride']*params['FPS'])
            # exception cases (mixed label case)
            if (frame_GT_label)>0.04 and (frame_GT_label)<0.96:
                continue
            # data save
            ind_save = ind_save +1
            sample = {'dat_radar': frame_radar_preprocess, 'dat_vidA': frame_video_A, 'dat_vidM': frame_video_M,
                        'GT_valDiff': frame_GT_val_diff, 'GT_label': frame_GT_label}
            if opt.flag_saveData:
                if ind_save==1:
                    person_name = ('_').join(data_path_radar.split('_')[6:9])
                    save_path_folder_person = os.path.join(save_path_folder,'data',person_name)
                    os.makedirs(save_path_folder_person, exist_ok=True)
                    with open(f'{save_path_folder_person}/{str(ind_save).zfill(4)}.pickle', 'wb') as f:
                        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(f'{save_path_folder_person}/{str(ind_save).zfill(4)}.pickle', 'wb') as f:
                        pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(str(ind_save).zfill(4))

    if opt.flag_saveData:
        data_dir = os.path.join(save_path_folder,'data')
        dat_total = []
        for person_name in os.listdir(data_dir):
            path_data = os.path.join(data_dir,person_name)
            for ind_data in np.arange(1,len(os.listdir(path_data))+1):
                with open(f'{path_data}/{str(ind_data).zfill(4)}.pickle', 'rb') as f:
                    dat_total.append(pickle.load(f))
        # Calculate normalization coefficient
        # Radar
        mean_radar = []
        std_radar = []
        for ind_chl in range(dat_total[0]['dat_radar'].shape[0]):
            dat_stack_radar = np.zeros((len(dat_total), dat_total[0]['dat_radar'].shape[1], dat_total[0]['dat_radar'].shape[2]), dtype=float)
            for ind_sample in range(len(dat_total)):
                dat_stack_radar[ind_sample,:,:] = dat_total[ind_sample]['dat_radar'][ind_chl,:,:]
            mean_radar.append(float(round(np.mean(dat_stack_radar),4)))
            std_radar.append(float(round(np.std(dat_stack_radar),4)))
        # Video
        mean_vidA = []
        std_vidA = []
        mean_vidM = []
        std_vidM = []
        for ind_chl in range(dat_total[0]['dat_vidM'].shape[1]):
            dat_stack_vidA = np.zeros((len(dat_total),dat_total[0]['dat_vidA'].shape[0],dat_total[0]['dat_vidA'].shape[2],dat_total[0]['dat_vidA'].shape[3]), dtype=float)
            dat_stack_vidM = np.zeros((len(dat_total),dat_total[0]['dat_vidM'].shape[0],dat_total[0]['dat_vidM'].shape[2],dat_total[0]['dat_vidM'].shape[3]), dtype=float)
            for ind_sample in range(len(dat_total)):
                dat_stack_vidA[ind_sample,:,:,:] = dat_total[ind_sample]['dat_vidA'][:,ind_chl,:,:]
                dat_stack_vidM[ind_sample,:,:,:] = dat_total[ind_sample]['dat_vidM'][:,ind_chl,:,:]
            mean_vidA.append(float(round(np.mean(dat_stack_vidA),4)))
            std_vidA.append(float(round(np.std(dat_stack_vidA),4)))
            mean_vidM.append(float(round(np.mean(dat_stack_vidM),4)))
            std_vidM.append(float(round(np.std(dat_stack_vidM),4)))
        params['mean_radar'] = mean_radar
        params['std_radar'] = std_radar
        params['mean_vidA'] = mean_vidA
        params['std_vidA'] = std_vidA
        params['mean_vidM'] = mean_vidM
        params['std_vidM'] = std_vidM

        # params save
        code_dir = os.path.join(save_path_folder, 'code')
        os.makedirs(code_dir, exist_ok=True)
        with open(os.path.join(code_dir,'Metalog_data.yml'), 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
        # Save Code
        current_file = os.path.realpath(__file__)
        shutil.copy(current_file, code_dir)

    return
