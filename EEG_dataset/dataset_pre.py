
import torch
from torch.utils.data import Dataset
import numpy as np
from EEG_utils.eeg_utils import GetDataPath, GetDataType
from imblearn.over_sampling import SMOTE


class preDataset(Dataset):
    def __init__(self, dataset_name="CHB", n=[], ite=1, augmentation=1, using_ictal=1,
                 scaler=0, patient_id=None, patient_name=None, ch_num=1, target_preictal_interval=15, step_preictal=5, model_name="STAN"):
        data, label = [], []
        self.scaler = scaler  # whether to balance preictal data and interictal data
        self.augmentation = augmentation  # whether using data augmentation
        self.using_ictal = using_ictal  # whether using ictal data
        self.data_path = GetDataPath(dataset_name)  # data path
        self.data_type = GetDataType(dataset_name)  # EEG/IEEG
        self.patient_id = patient_id  # patient id
        self.patient_name = patient_name  # patient name
        self.ch_num = ch_num  # number of channels
        self.target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
        self.step_preictal = step_preictal  # step of sliding window
        self.doing_STFT = True
        self.model_name = model_name  # model name

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
                Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/ictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            elif model_name.startswith("MONST"):
                # data loading from npy files
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/preictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
                # 175 18 59 114
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/interictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
                # 342 18 59 114
                Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_%s/ictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, self.doing_STFT, i)))
            else:
                # data loading from npy files
                preIctal = np.load(("%s/%s/%dmin_%dstep_%dch/preictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
                # 175 18 59 114
                interIctal = np.load(("%s/%s/%dmin_%dstep_%dch/interictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
                # 342 18 59 114
                Ictal = np.load(("%s/%s/%dmin_%dstep_%dch/ictal%d.npy" % (
                    self.data_path, self.patient_name, self.target_preictal_interval, self.step_preictal, self.ch_num, i)))
            # shape transpose
            if (len(preIctal.shape) == 3):
                preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
                Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
                interIctal = interIctal.transpose(0, 2, 1)

            # whether to use ictal data for training
            if self.using_ictal == 1:
                print("using ictal data")
                if Ictal.shape[0] > 0:
                    preIctal = np.concatenate((preIctal, Ictal), 0)
                # print("concatenated preictal {}".format(preIctal.shape))#(38, 19, 1280)

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
            smote = SMOTE()
            data2, data2label = [], []

            ########### 7.0 ############
            data2 = np.concatenate((preIctal, interIctal), axis=0)
            data2_shape = data2.shape
            data2 = data2.reshape(data2.shape[0], data2.shape[1] * data2.shape[2] * data2.shape[3])
            data2label = np.concatenate((np.ones((preIctal_shape[0], 1)), np.zeros((interIctal_shape[0], 1))), axis=0)
            # 使用SMOTE算法对preIctal进行过采样，生成与interIctal数量相同的样本
            X_resampled, y_resampled = smote.fit_resample(data2, data2label)
            zeros_count = np.bincount(y_resampled == 0)[1]
            ones_count = np.bincount(y_resampled == 1)[1]
            print(f"y_resampled 中包含 {zeros_count} 个 0 和 {ones_count} 个 1")
            # 将过采样后的preIctal和随机选择的interIctal样本添加到data列表中
            X_resampled = X_resampled.reshape(X_resampled.shape[0], data2_shape[1], data2_shape[2], data2_shape[3])
            data.append(X_resampled)
            # 更新标签
            label.append(y_resampled)
            print('seizure {} : preictal {} | Ictal {} | interIctal {}'.format(i, preIctal.shape, Ictal.shape, interIctal.shape))
            print('seizure {} : X_resampled {} | y_resampled {} '.format(i, X_resampled.shape, y_resampled.shape))
            ########### 7.0 ############


        # numpy to torch
        data, label = np.array(data), np.array(label)
        data, label = np.concatenate(data, 0), np.concatenate(label, 0)
        label = np.expand_dims(label, axis=1)
        if (len(preIctal.shape) == 3):
            data = data[:, np.newaxis, :, :].astype('float32')
        # elif (len(preIctal.shape)==4):#spectralCNN
        # data = data[:, np.newaxis, :, :, :].astype('float32')

        label = label.astype('int64')
        self.x_data = torch.from_numpy(data)  # ([2592, 1, 19, 1280])
        self.y_data = torch.from_numpy(label)  # ([2592, 1])
        self.len = data.shape[0]
        print("Target {} Dataset : {} {}".format(str(augmentation), self.x_data.shape, self.y_data.shape))
        print("preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
