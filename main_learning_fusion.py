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

import visdom
# vis = visdom.Visdom(port='8098')
vis = visdom.Visdom()

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

def visualize_resp(data_resp, vec_s):
    vec_len = data_resp.shape[0]
    sos = signal.butter(2, [0.08, 0.6], 'bandpass', fs=vec_len/vec_s, output='sos') # band pass filter
    if np.mean(np.abs(data_resp))<=0.3:
        data_temp = data_resp
        data_filter = signal.sosfilt(sos,data_temp)
        data_out = data_filter
    else:
        data_resp = np.cumsum(data_resp)
        data_temp = data_resp - np.mean(data_resp)
        data_filter = signal.sosfilt(sos,data_temp)
        # data_filter = (data_filter-np.min(data_filter))/(np.max(data_filter)-np.min(data_filter))
        # data_filter = 2*data_filter-1
        data_out = data_filter
    return data_out

def visualize_resp_nointeg(data_resp, vec_s):
    vec_len = data_resp.shape[0]
    sos = signal.butter(2, [0.08, 0.6], 'bandpass', fs=vec_len/vec_s, output='sos') # band pass filter
    if np.mean(np.abs(data_resp))<=0.3:
        data_temp = data_resp
        data_filter = signal.sosfilt(sos,data_temp)
        data_out = data_filter
    else:
        data_temp = data_resp - np.mean(data_resp)
        data_filter = signal.sosfilt(sos,data_temp)
        # data_filter = (data_filter-np.min(data_filter))/(np.max(data_filter)-np.min(data_filter))
        # data_filter = 2*data_filter-1
        data_out = data_filter
    return data_out

def train(opt, params):
    dir_dataset = os.path.join(params['data_save_root'],opt.project,'data')
    train_set = Custom_Dataset(dir_dataset, opt, params, set='train', 
                                    transform=transforms.Compose([
                                                                    prepro_Normalize(params),
                                                                    # augment_randFlip()
                                                                    ]))
    test_set = Custom_Dataset(dir_dataset, opt, params, set='test', 
                                    transform=transforms.Compose([
                                                                    prepro_Normalize(params),
                                                                    ]))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,  num_workers=opt.num_workers)

    len_frame_video = int((params['Frame_s']*params['FPS']*opt.frame_length)//params['Frame_s'])
    len_frame_radar = int(params['STFT_resize_T']*opt.frame_length//params['Frame_s'])
    num_frame = int(params['Frame_s']//opt.frame_length)
    dim_frame_video = [len_frame_video, params['img_resize'], params['img_resize']]
    dim_frame_radar = [params['STFT_resize_F'], len_frame_radar]

    if opt.fusion_type=='radar':
        model = model_form.main_Net_Radar(dim_frame=dim_frame_radar, activation='relu').cuda()
    elif opt.fusion_type=='video':
        model = model_form.main_Net_Video(dim_frame=dim_frame_video, fold_div=opt.fold_div).cuda()
    elif opt.fusion_type=='fusion':
        if opt.fusion_mode=='transformer':
            model = model_form.main_Net_Fusion(dim_video=dim_frame_video, dim_radar=dim_frame_radar, 
                                                activation='relu', fold_div_video=opt.fold_div).cuda()
        elif opt.fusion_mode=='late':
            from model import model_LateFusion
            model = model_LateFusion.main_Net_Fusion_Late(dim_video=dim_frame_video, dim_radar=dim_frame_radar, fold_div_video=opt.fold_div).cuda()
    # from torchinfo import summary
    # summary(model, input_size=((32, 8, 128, 60), (32, 30, 3, 36, 36), (32, 30, 3, 36, 36)))
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    criterion_L1 = nn.L1Loss().cuda()


    best_loss = 100
    step = 0
    # Learning
    print("=============Learning Started=============")
    train_loss = AverageMeter('Train_Loss', ':3.2f')
    test_loss = AverageMeter('Test_Loss', ':3.2f')
    test_mae_freq = AverageMeter('MAE_total', ':3.2f')
    test_mae_freq_label0 = AverageMeter('MAE_0', ':3.2f')
    test_mae_freq_label1 = AverageMeter('MAE_1', ':3.2f')
    test_rmse_freq = AverageMeter('RMSE_total', ':3.2f')
    test_rmse_freq_label0 = AverageMeter('RMSE_0', ':3.2f')
    test_rmse_freq_label1 = AverageMeter('RMSE_1', ':3.2f')
    test_std_freq = AverageMeter('Std_total', ':3.2f')
    test_std_freq_label0 = AverageMeter('Std_0', ':3.2f')
    test_std_freq_label1 = AverageMeter('Std_1', ':3.2f')
    test_p_freq = AverageMeter('p_total', ':3.2f')
    test_p_freq_label0 = AverageMeter('p_0', ':3.2f')
    test_p_freq_label1 = AverageMeter('p_1', ':3.2f')
    result_all = [train_loss,test_loss,
                    test_mae_freq,test_mae_freq_label0,test_mae_freq_label1,
                    test_rmse_freq,test_rmse_freq_label0,test_rmse_freq_label1,
                    test_std_freq,test_std_freq_label0,test_std_freq_label1,
                    test_p_freq,test_p_freq_label0,test_p_freq_label1]
    num_iter_per_epoch = len(train_loader)
    for epoch in range(opt.num_epochs):
        [meter.reset() for meter in result_all]  # reset all meters
        test_label_0_len = 0
        test_label_1_len = 0
        ts = time.time()

        model.train() 
        progress_bar = tqdm(train_loader)
        for iter, data in enumerate(progress_bar):
            X_radar, X_vidA, X_vidM, Y_respDiff, Y_label = data
            X_radar = Variable(X_radar.float().cuda())
            X_vidA = Variable(X_vidA.float().cuda())
            X_vidM = Variable(X_vidM.float().cuda())
            Y_respDiff = Variable(Y_respDiff.float().cuda())
            Y_label = Variable(Y_label.long().cuda())

            for window_ind in range(num_frame):
            ## Train model
                # Estimation
                if opt.fusion_type=='radar':
                    resp_out = model(X_radar[:,:,:,window_ind*len_frame_radar:window_ind*len_frame_radar+len_frame_radar])
                elif opt.fusion_type=='video':
                    resp_out = model(X_vidA[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:], 
                                        X_vidM[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:])
                elif opt.fusion_type=='fusion':
                    resp_out = model(X_radar[:,:,:,window_ind*len_frame_radar:window_ind*len_frame_radar+len_frame_radar],
                                        X_vidA[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:], 
                                        X_vidM[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:])
                # Backpropagate model
                resp_out = torch.squeeze(resp_out)                      # estimation of resp
                loss_resp = criterion_L1(Y_respDiff[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video], resp_out)
                loss = loss_resp
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Result summary
                step += 1          
                train_loss.update(loss.item(), 1)

            # progress_bar print
            progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Total loss: {:.5f}.'.format(
                    step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, loss.item()))
        scheduler.step(train_loss.avg)


        # Validation
        if epoch % opt.val_interval == 0:
            model.eval()
            freq_vec = []
            freq_est_vec = []
            label_vec = []
            
            for iter, data in enumerate(test_loader):
                with torch.no_grad():
                    X_radar, X_vidA, X_vidM, Y_respDiff, Y_label = data
                    X_radar = Variable(X_radar.float().cuda())
                    X_vidA = Variable(X_vidA.float().cuda())
                    X_vidM = Variable(X_vidM.float().cuda())
                    Y_respDiff = Variable(Y_respDiff.float().cuda())
                    Y_label = Variable(Y_label.long().cuda())

                    # postprocess                                    
                    GT_label = np.array(torch.squeeze(Y_label).to('cpu'))
                    GT_respDiff = np.array(torch.squeeze(Y_respDiff).to('cpu'))
                    est_respDiff = np.zeros(Y_respDiff.shape[1])
                    Y_respDiff = torch.squeeze(Y_respDiff)
                    for window_ind in range(num_frame):
                        if opt.fusion_type=='radar':
                            resp_out = model(X_radar[:,:,:,window_ind*len_frame_radar:window_ind*len_frame_radar+len_frame_radar])
                        elif opt.fusion_type=='video':
                            resp_out = model(X_vidA[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:], 
                                                X_vidM[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:]) 
                        elif opt.fusion_type=='fusion':
                            resp_out = model(X_radar[:,:,:,window_ind*len_frame_radar:window_ind*len_frame_radar+len_frame_radar],
                                                X_vidA[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:], 
                                                X_vidM[:,window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video,:,:,:]) 
                        resp_out = torch.squeeze(resp_out)                      # estimation of resp                        
                        loss_resp = criterion_L1(Y_respDiff[window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video], resp_out)
                        loss = loss_resp
                        test_loss.update(loss.item(), 1)   
                        est_respDiff[window_ind*len_frame_video:window_ind*len_frame_video+len_frame_video] = np.array(resp_out.data.to('cpu'))
                    
                    freq_gt = estimate_freq_psd(GT_respDiff, np.round(np.mean(GT_label)), vec_s=params['Frame_s'])*60
                    freq_est = estimate_freq_psd(est_respDiff, np.round(np.mean(GT_label)), vec_s=params['Frame_s'])*60
                    
                    label_vec.append(np.round(np.mean(GT_label)))
                    freq_vec.append(freq_gt)
                    freq_est_vec.append(freq_est)

                    # visdom plot
                    if epoch==0:
                        if iter==0:
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=GT_respDiff, opts=dict(title=f'G.T. [RR], iter: {iter}'))
                            # vis.line(X=np.linspace(0,10,len(resp_frame)),Y=resp_est_frame, opts=dict(title=f'Epoch: {epoch+1} [RR]'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=np.cumsum(GT_respDiff), opts=dict(title=f'G.T. [RR_sum], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=visualize_resp(GT_respDiff,params['Frame_s']), opts=dict(title=f'G.T. [RR_sum], iter: {iter}'))
                            # vis.line(X=np.linspace(0,10,len(resp_frame)),Y=np.cumsum(resp_est_frame), opts=dict(title=f'Epoch: {epoch+1} [RR]'))
                        if iter==40:
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=GT_respDiff, opts=dict(title=f'G.T. [RR], iter: {iter}'))
                            # vis.line(X=np.linspace(0,10,len(resp_frame)),Y=resp_est_frame, opts=dict(title=f'Epoch: {epoch+1} [RR]'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=np.cumsum(GT_respDiff), opts=dict(title=f'G.T. [RR_sum], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=visualize_resp(GT_respDiff,params['Frame_s']), opts=dict(title=f'G.T. [RR_sum], iter: {iter}'))
                            # vis.line(X=np.linspace(0,10,len(resp_frame)),Y=np.cumsum(resp_est_frame), opts=dict(title=f'Epoch: {epoch+1} [RR]'))
                    elif epoch>0 and epoch % 5==0:
                        if iter==0:
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)),Y=est_respDiff, opts=dict(title=f'Epoch: {epoch+1} [RR], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)),Y=np.cumsum(est_respDiff), opts=dict(title=f'Epoch: {epoch+1} [RR_sum], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=visualize_resp(est_respDiff,params['Frame_s']), opts=dict(title=f'Epoch: {epoch+1} [RR_sum], iter: {iter}'))
                        if iter==40:
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)),Y=est_respDiff, opts=dict(title=f'Epoch: {epoch+1} [RR], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)),Y=np.cumsum(est_respDiff), opts=dict(title=f'Epoch: {epoch+1} [RR_sum], iter: {iter}'))
                            vis.line(X=np.linspace(0,10,len(GT_respDiff)), Y=visualize_resp(est_respDiff,params['Frame_s']), opts=dict(title=f'Epoch: {epoch+1} [RR_sum], iter: {iter}'))

            freq_vec = np.array(freq_vec)
            freq_est_vec = np.array(freq_est_vec)
            label_vec = np.array(label_vec)
            test_mae_freq.update((np.sum(np.abs(freq_vec-freq_est_vec)))/len(freq_vec), 1)
            test_rmse_freq.update(np.sqrt(np.sum((freq_vec-freq_est_vec)**2)/len(freq_vec)), 1)
            test_std_freq.update(np.std(freq_vec-freq_est_vec), 1)
            test_p_freq.update(PCC(freq_vec,freq_est_vec), 1)

            ind_label0 = np.where(label_vec==0)[0]
            ind_label1 = np.where(label_vec==1)[0]
            test_label_0_len = len(ind_label0)+1e-6 
            test_label_1_len = len(ind_label1)+1e-6 
            test_mae_freq_label0.update((np.sum(np.abs(freq_vec[ind_label0]-freq_est_vec[ind_label0])))/test_label_0_len, 1)
            test_mae_freq_label1.update((np.sum(np.abs(freq_vec[ind_label1]-freq_est_vec[ind_label1])))/test_label_1_len, 1)
            test_rmse_freq_label0.update(np.sqrt(np.sum((freq_vec[ind_label0]-freq_est_vec[ind_label0])**2)/test_label_0_len), 1)
            test_rmse_freq_label1.update(np.sqrt(np.sum((freq_vec[ind_label1]-freq_est_vec[ind_label1])**2)/test_label_1_len), 1)
            test_std_freq_label0.update(np.std(np.abs(freq_vec[ind_label0]-freq_est_vec[ind_label0])), 1)
            test_std_freq_label1.update(np.std(np.abs(freq_vec[ind_label1]-freq_est_vec[ind_label1])), 1)
            test_p_freq_label0.update(PCC(freq_vec[ind_label0],freq_est_vec[ind_label0]), 1)
            test_p_freq_label1.update(PCC(freq_vec[ind_label1],freq_est_vec[ind_label1]), 1)

        te = time.time()    # end time

        [meter.update_list() for meter in result_all]   # update result list

        print('Epoch {}, Total Loss(train/test) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch+1, train_loss.avg, test_loss.avg, te-ts))
        print('MAE Loss: Total {:2.2f}[{:2.2f},{:2.2f}].  RMSE Loss: Total {:2.2f}[{:2.2f},{:2.2f}].' 
                '  Std: Total {:2.2f}[{:2.2f},{:2.2f}].   p: Total {:2.2f}[{:2.2f},{:2.2f}].'
                .format(test_mae_freq.avg, test_mae_freq_label0.avg, test_mae_freq_label1.avg, 
                        test_rmse_freq.avg, test_rmse_freq_label0.avg, test_rmse_freq_label1.avg,
                        test_std_freq.avg, test_std_freq_label0.avg, test_std_freq_label1.avg,
                        test_p_freq.avg, test_p_freq_label0.avg, test_p_freq_label1.avg))

        if opt.flag_saveResult and epoch>=20:
            # Make save folder
            if epoch==20:
                save_dir = os.path.join(current_dir, f'dataset/{opt.project}/result')
                os.makedirs(save_dir, exist_ok=True)
                folder_list = [file for file in os.listdir(save_dir) if ("_try") in file]
                folder_name = f'{len(folder_list)+1}_try'
                save_dir = os.path.join(save_dir, folder_name)
                os.makedirs(save_dir, exist_ok=True)
                # Save hyperparams
                with open(os.path.join(save_dir, f'[{folder_name}]opt.yml'), 'w') as f:
                    yaml.dump(vars(opt), f, sort_keys=False, default_flow_style=False)
                with open(os.path.join(save_dir, f'[{folder_name}]params.yml'), 'w') as f:
                    yaml.dump(params, f, sort_keys=False, default_flow_style=False)
                # Save Code
                code_dir = os.path.join(save_dir, 'code')
                os.makedirs(code_dir, exist_ok=True)
                current_file = os.path.realpath(__file__)
                current_path = '/'.join(current_file.split('/')[:-1])
                shutil.copy(current_file, code_dir)
                shutil.copy(os.path.join(current_path,'sub_dataload_stdata_fusion.py'), code_dir)
                shutil.copy(os.path.join(current_path,'sub_dataset_stdata_fusion.py'), code_dir)
                shutil.copy(os.path.join(current_path,'model','model_form.py'), code_dir)
                shutil.copy(os.path.join(current_path,'model','model_LateFusion.py'), code_dir)
                shutil.copy(os.path.join(current_path,'model','sub_model_TransFusor.py'), code_dir)
                os.rename(os.path.join(code_dir,'main_learning_fusion.py'), os.path.join(code_dir,f'[{folder_name}]main_learning_fusion.py'))
                os.rename(os.path.join(code_dir,'sub_dataload_stdata_fusion.py'), os.path.join(code_dir,f'[{folder_name}]sub_dataload_stdata_fusion.py'))
                os.rename(os.path.join(code_dir,'sub_dataset_stdata_fusion.py'), os.path.join(code_dir,f'[{folder_name}]sub_dataset_stdata_fusion.py'))
                os.rename(os.path.join(code_dir,'model_form.py'), os.path.join(code_dir,f'[{folder_name}]model_form.py'))
                os.rename(os.path.join(code_dir,'model_LateFusion.py'), os.path.join(code_dir,f'[{folder_name}]model_LateFusion.py'))
                os.rename(os.path.join(code_dir,'sub_model_TransFusor.py'), os.path.join(code_dir,f'[{folder_name}]sub_model_TransFusor.py'))

            if test_loss.avg < best_loss:
                best_loss = test_loss.avg
                best_state={
                'Epoch': epoch+1,
                'State_dict': copy.deepcopy(model.state_dict()),
                'Optimizer': copy.deepcopy(optimizer.state_dict()),
                }
                result_summary = {}
                for meter in result_all:
                    result_summary[meter.name] =  meter.val_list
                torch.save(best_state, os.path.join(save_dir, 'Best.pth'))          # model save
                with open(os.path.join(save_dir, 'Result_summary.pkl'), 'wb') as f: # result save
                    pickle.dump(result_summary,f)


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


