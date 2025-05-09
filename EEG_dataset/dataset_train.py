
import time

import torch
from torch.utils.data import Dataset
import numpy as np
from EEG_utils.eeg_utils import GetDataPath, GetDataType


# from imblearn.over_sampling import SMOTE, ADASYN


class trainDataset(Dataset):
    def __init__(self, dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5, model_name="STAN", doing_Noise=True):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        # self.data_path = "/share/home/data/CHBMIT60/"


        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval
        self.step_preictal = step_preictal  # step of sliding window
        self.doing_STFT = True
        # self.doing_Noise = False
        self.doing_Noise = True
        self.model_name = model_name  # model name
        self.scale = 0.25

        start_time = time.time()
        # LOOCV for n times for each patient. n is the number of seizures
        for i in n:
            if model_name.startswith("STAN"):
                # data loading from npy files
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/preictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
                # 175 18 59 114
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/interictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
                # 342 18 59 114
                if dataset_name.startswith("KAGGLE"):
                    pass
                else:
                    Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/ictal%d.npy" % (
                        self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            elif model_name.startswith("MONST"):
                # data loading from npy files
                print("data train")

                print("data path:", "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    str(self.scale)))

                # 临时使用 66-73行注释掉的代码
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/preictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    i)))
                # 175 18 59 114
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/interictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    i)))

                if dataset_name.startswith("KAGGLE"):
                    pass
                else:

                    Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/ictal%d.npy" % (
                        self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                        i)))

            else:

                print("data path:", "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    str(self.scale)))

                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/preictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    str(self.scale), i)))
                # 175 18 59 114
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/interictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    str(self.scale), i)))
                # 342 18 59 114
                Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/ictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, self.doing_Noise,
                    str(self.scale), i)))
                #############################

            # shape transpose
            if (len(preIctal.shape) == 3):
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)

            # whether to use ictal data for training
            if self.using_ictal == 1:
                print("using ictal data")
                if dataset_name.startswith("KAGGLE"):
                    pass
                else:
                    if Ictal.shape[0] > 0:
                        # preIctal = np.concatenate((preIctal, Ictal), 0)
                        preIctal = np.vstack((preIctal, Ictal))


            # data augmentation
            if self.augmentation == 1:
                print("doing augmentation")
                temp = []
                indT = np.arange(len(preIctal) * 2)  # (428,)
                for _ in range(ite):
                    tmp = np.concatenate(np.split(preIctal, 2, -1), 0)  # (428, 19, 640)
                    np.random.shuffle(indT)
                    tmp = tmp[indT]
                    tmp = np.concatenate(np.split(tmp, 2, 0), -1)  # (214, 19, 1280)
                    temp.append(tmp)
                temp.append(preIctal)
                temp = np.concatenate(temp, 0)
                preIctal = temp  # (428, 19, 1280)

            # whether to balance interictal and preictal data
            ind = np.arange(0, len(interIctal))
            np.random.shuffle(ind)
            # 获取preIctal和interIctal的形状信息
            preIctal_shape = preIctal.shape
            interIctal_shape = interIctal.shape
            # 计算preIctal和interIctal之间的数量差异
            diff = abs(preIctal_shape[0] - interIctal_shape[0])


            ########### 7.0 ############

            data.append(preIctal)
            data.append(interIctal)
            label.append(np.ones((preIctal.shape[0], 1)))
            label.append(np.zeros((interIctal.shape[0], 1)))

            ########### 7.0 ############

        # numpy to torch

        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        # preIctal_data_label_1 = data[label[:, 0] == 1]
        # interIctal_data_label_0 = data[label[:, 0] == 0]

        ########### 7.0 ############
        # 获取preIctal和interIctal的形状信息
        ones_count = np.sum(label == 1)
        zeros_count = np.sum(label == 0)
        diff = abs(ones_count - zeros_count)
        # # 添加标签，1 表示 preIctal，0 表示 interIctal
        print("最初")
        print(f"label 中包含 interIctal{zeros_count} 个 0 和 preIctal{ones_count} 个 1")

        # 计算标签数量之间的倍数
        ratio = max(ones_count, zeros_count) // min(ones_count, zeros_count)
        if ratio > 1:
            diff = (diff // ratio) - 1

        # 记录每个 epoch 的结束时间
        end_time = time.time()
        # 计算每个 epoch 的时间差
        epoch_time = end_time - start_time
        print("PRE: {:.2f} seconds".format(epoch_time))
        print("########################### shift ###########################")

        # label = np.vstack((label, label[noise_indices]))

        print("data shape:", data.shape)
        print("label shape:", label.shape)

        # 记录每个 epoch 的结束时间
        end_time = time.time()
        # 计算每个 epoch 的时间差
        epoch_time = end_time - start_time
        print("PRE: {:.2f} seconds".format(epoch_time))

        print("data shape:", data.shape)
        print("label shape:", label.shape)

        ########### 7.0 ############

        if len(preIctal.shape) == 3:
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif (len(preIctal.shape)==4):#spectralCNN
        # data = data[:, np.newaxis, :, :, :].astype('float32')

        label = label.astype('int64')
        self.x_data = torch.from_numpy(data.astype('float32'))  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]
        print("Target {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))
        print("########################### FIN ###########################")
        ones_count = np.sum(label == 1)
        zeros_count = np.sum(label == 0)
        # # 添加标签，1 表示 preIctal，0 表示 interIctal
        print(f"FIN label 中包含 interIctal {zeros_count} 个 0 和 preIctal {ones_count} 个 1")
        # 记录每个 epoch 的结束时间
        end_time = time.time()
        # 计算每个 epoch 的时间差
        epoch_time = end_time - start_time
        # 打印每个 epoch 的时间信息
        print("PRE: {:.2f} seconds".format(epoch_time))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
