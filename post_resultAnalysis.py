import os
import yaml
import pickle
import argparse

import numpy as np

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

DATA_SAVE_ROOT = '/workspace/Res_VitalMonitoring_Fusion/dataset/'
NAME_DATA = '[2022-04-03]054304'    # name of reference data
TEST_IND = 30                        # index for test result

if __name__ == '__main__':
    opt_ref = get_args()
    opt_ref = vars(opt_ref)
    params_ref = yaml.safe_load(open(os.path.join(DATA_SAVE_ROOT, f'{NAME_DATA}/code/Metalog_data.yml')))
    opt_test = yaml.safe_load(open(os.path.join(DATA_SAVE_ROOT, f'{NAME_DATA}/result/{str(TEST_IND)}_try/[{str(TEST_IND)}_try]opt.yml')))
    params_test = yaml.safe_load(open(os.path.join(DATA_SAVE_ROOT, f'{NAME_DATA}/result/{str(TEST_IND)}_try/[{str(TEST_IND)}_try]params.yml')))

    with open(os.path.join(DATA_SAVE_ROOT, f'{NAME_DATA}/result/{str(TEST_IND)}_try/Result_summary.pkl'), 'rb') as f:
        result_summary = pickle.load(f)

    # Find hyper parameters different from the reference parameters
    opt_diff = [key for key in opt_ref.keys() if opt_ref[key]!=opt_test[key]]
    params_diff = [key for key in params_ref.keys() if params_ref[key]!=params_test[key]]

    min_idx = np.array(result_summary['Test_Loss']).argmin()
    result_min = {}
    for key in result_summary.keys():
        result_min[key] = result_summary[key][min_idx]

    a = 1