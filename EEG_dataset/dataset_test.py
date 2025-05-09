

from __future__ import print_function
import torch
from torch.utils.data import Dataset
import numpy as np
from EEG_utils.eeg_utils import GetDataPath, GetDataType


class testDataset(Dataset):
    def __init__(self, dataset_name="XUANWU", i=0, using_ictal=1, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5,
                 model_name="STAN", doing_Noise=False):
        data, label = [], []
        self.data_path = GetDataPath(dataset_name)  # data path
        # self.data_path = "/share/home/zhanghan/data/CHBMIT60/"
        self.data_path = "/data1/zhanghan/data/CHBMIT60/"

        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval
        self.step_preictal = step_preictal
        self.doing_STFT = True
        self.doing_Noise = False
        self.model_name = model_name  # model name
        if model_name.startswith("STAN"):
            # data loading
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/preictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/interictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/ictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
        elif model_name.startswith("MONST"):
            print("MONST test")
            print("data path:", "%s/%s/%dmin_%dstep_%dch_STFT_%s_T" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT))
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_T/preictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_T/interictal%d.npy" % (
                self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            if dataset_name.startswith("KAGGLE"):
                pass
            else:
                Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_T/ictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
        else:
            # data loading
            preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            interIctal = np.load(("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            Ictal = np.load(("%s/%s/%dmin_%dstep_%dch/ictal%d.npy" % (
            self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
        # shape transpose
        self.preictal_length = preIctal.shape[0]
        self.interictal_length = interIctal.shape[0]
        if dataset_name.startswith("KAGGLE"):
            pass
        else:
            self.ictal_length = Ictal.shape[0]
        if (len(preIctal.shape) == 3):
            preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
            Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
            interIctal = interIctal.transpose(0, 2, 1)

        # whether to use ictal data for testing
        if dataset_name.startswith("KAGGLE"):
            pass
        else:
            if using_ictal == 1 and Ictal.shape[0] != 0:
                preIctal = np.concatenate((preIctal, Ictal), 0)

        data_zero = len(interIctal)
        data_one = len(preIctal)
        # data concat
        data.append(interIctal)
        data.append(preIctal)
        label.append(np.zeros((interIctal.shape[0], 1)))
        label.append(np.ones((preIctal.shape[0], 1)))

        # numpy to torch
        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        print(data.shape, label.shape)
        # 计算0的数量
        num_zeros = label.size - np.count_nonzero(label)
        # 计算1的数量
        num_ones = np.count_nonzero(label)
        print("label 1 :", num_ones)
        print("label 0 :", num_zeros)
        print("data 1 :", data_one)
        print("data 0 :", data_zero)

        if (len(preIctal.shape) == 3):
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif (len(preIctal.shape)==4):#spectralCNN
        #     data = data[:, np.newaxis, :, :, :].astype('float32')
        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
