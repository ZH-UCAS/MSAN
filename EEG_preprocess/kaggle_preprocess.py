import numpy
import pandas as pd
import numpy as np
import os
import scipy.io
import parser as p
import csv
import os
import glob
from datetime import datetime

import numpy as np
import random

from scipy.signal import resample

from chb_edf_file import CHBEdfFile
from chb_stft import getSpectral_STFT, getSpectral_STFT_T
import matplotlib.pyplot as plt
import argparse
import sys
import mne

sys.path.append('')
from EEG_utils.eeg_utils import GetInputChannel, mkdir, GetPatientList, GetDataPath

import matplotlib.pyplot as plt
import cupy as cp


def setup_seed(seed):  # 为CPU设置随机种子用于生成随机数，以使得结果是确定的
    '''
    set up random seed for numpy
    '''
    np.random.seed(seed)  # 为numpy设置随机种子
    random.seed(seed)  # 为python设置随机种子


def wgn(x, snr):
    ## x: input vibration signal shape (a,b); a:samples number; b samples length
    ## snr: noise intensity,like -8,4,2,0,2,4,8
    ### snr=0 means noise equal to vibration signal
    ### snr>0 means vibration signal stronger than noise, →∞ means no noise
    ### snr<0 means noise stronger than vibration signal  →-∞ means no signal
    x = x.T
    Ps = np.sum(abs(x) ** 2, axis=1) / len(x)
    Pn = Ps / (10 ** ((snr / 10)))
    row, columns = x.shape
    Pn = np.repeat(Pn.reshape(-1, 1), columns, axis=1)

    noise = np.random.randn(row, columns) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def shiftdata(data):
    # 生成随机的(18,7680)数组
    data = data.copy()
    datashape = data.shape
    data = data.T
    # print(data.shape)
    # Generate a random integer in the range (1280, data.shape[1]-1280)
    random_int = random.randint(1280, data.shape[1] - 2560)
    # 随机选择一个起始点和一个长度为1280的区间
    start1 = np.random.randint(0, random_int)
    end1 = start1 + 1280

    # 选择另一个不重叠的起始点
    start2 = np.random.randint(end1 + 1, data.shape[1] - 1280)
    end2 = start2 + 1280
    # 进行对换
    data[:, start1:end1], data[:, start2:end2] = data[:, start2:end2], data[:, start1:end1].copy()
    return data


def plot_time_intensity(rawdata, save_dir, index, seizure, classification="interictal", colors=None, figsize=(20, 12), dpi=300):
    """
    绘制时序-强度图像。

    参数：
        - data：numpy数组，18行X列，表示时序-强度数据。
        - colors：列表，每个元素是一个颜色字符串，用于绘制每行的线条。默认为None，使用默认颜色列表。
        - figsize：元组，表示图像的尺寸，默认为(20, 12)。
        - dpi：整数，表示图像的分辨率，默认为300。

    返回值：
        无返回值，直接显示图像。
    """
    rawdata = cp.asnumpy(rawdata)
    # 如果未指定颜色列表，则使用默认颜色列表
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'navy', 'teal', 'maroon',
                  'coral']

    with cp.cuda.Device(0):
        # 创建图像和坐标轴
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        data = rawdata.T * 5000

        # 绘制每行的连续线
        for i in range(18):
            ax.plot(data[i, :] + i, color=colors[i % len(colors)], linewidth=0.5, zorder=18 - i)

        # 设置图像属性
        ax.set_xlim([0, data.shape[1]])
        ax.set_ylim([0, 18])
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_title('Time-Intensity Image')

        # 生成文件名
        # t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'seizure{seizure}_{index}_{classification}.png'
        # 保存图像
        save_dir = save_dir + "/img"
        file_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存图像
        plt.savefig(file_path)
        # plt.show()
        plt.close()

        # 创建图像和坐标轴
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        data = rawdata.T
        data = (data - cp.min(data)) / (cp.max(data) - cp.min(data))
        # 绘制每行的连续线
        for i in range(18):
            ax.plot(data[i, :] + i, color=colors[i % len(colors)], linewidth=0.5, zorder=18 - i)

        # 设置图像属性
        ax.set_xlim([0, data.shape[1]])
        ax.set_ylim([0, 18])
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_title('Time-Intensity Image')

        # 生成文件名
        # t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'seizure{seizure}_{index}_{classification}_normalized.png'
        # 保存图像
        # save_dir = save_dir + "/img"
        file_path = os.path.join(save_dir, filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存图像
        plt.savefig(file_path)
        # plt.show()
        plt.close()


class CHBPatient:  # CHB病人类
    def __init__(self, patient_id, data_path, ch_num, doing_lowpass_filter, preictal_interval):
        self.interictal_interval = 90  # 90min or longer before a seizure, decide as interictal data  #  90min或更长时间之前的癫痫，被认为是非癫痫状态
        self.preictal_interval = preictal_interval  # how long we decide as preictal. Default set to 15 min #  我们认为多长时间是癫痫前期。默认设置为15分钟
        self.postictal_interval = 120  # within 120min after a seizure, decide as postictal data #  在癫痫后120min内，被认为是癫痫后期
        self.patient_id = patient_id  # 病人ID
        self.data_path = data_path  # 数据路径
        self.ch_num = ch_num  # 通道数
        self.doing_lowpass_filter = doing_lowpass_filter  # 是否进行低通滤波
        self.patient_name = self.get_patient_name()

        # load edf files with seizure
        self._edf_files_seizure = list(map(
            lambda filename: CHBEdfFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("%s/%s/seizure/*.edf" % (self.data_path, self.patient_name)))
        ))

        # load edf files without seizures
        self._edf_files_unseizure = list(map(
            lambda filename: CHBEdfFile(filename, self.patient_id, self.ch_num, self.doing_lowpass_filter),
            sorted(glob.glob("%s/%s/unseizure/*.edf" % (self.data_path, self.patient_name)))
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDF preprocessing on CHB Dataset')
    # parser.add_argument('--data_path', type=str, default="/data1/zhanghan/data/KAGGLE/", metavar='data path')
    parser.add_argument('--data_path', type=str, default="/share/home/zhanghan/data/KAGGLE/", metavar='data path')

    parser.add_argument('--patient_id', type=int, default=1, metavar='patient id')
    parser.add_argument('--target_preictal_interval', type=int, default=60, metavar='how long we decide as preictal. Default set to 15 min')  # in minute
    parser.add_argument('--seed', type=int, default=1997, metavar='random seed')
    parser.add_argument('--ch_num', type=int, default=18, metavar='number of channel')
    parser.add_argument('--sfreq', type=int, default=384, metavar='sample frequency')
    parser.add_argument('--window_length', type=int, default=30, metavar='sliding window length')  # if stft : 30              else 5
    parser.add_argument('--preictal_step', type=int, default=6, metavar='step of sliding window (second) for preictal data')  # if stft : 5               else 5
    parser.add_argument('--interictal_step', type=int, default=30,
                        metavar='step of sliding window (second) for interictal data')  # if stft : 30             else 5
    parser.add_argument('--doing_STFT', type=bool, default=False, metavar='whether to do STFT')  # if stft : True           else False
    parser.add_argument('--doing_noise', type=bool, default=False, metavar='whether to do noise')
    parser.add_argument('--doing_lowpass_filter', type=bool, default=True, metavar='whether to do low pass filter')  # if stft : False else True

    args = parser.parse_args()
    setup_seed(args.seed)
    patient_list = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
    seizure_list = [4, 7, 12, 14, 5, 3, 3]
    ch_num_list = [16, 16, 16, 16, 15, 15, 24]
    hz_num_list = [384, 384, 384, 384, 384, 512, 512]

    sfreq = args.sfreq
    window_length = args.window_length
    preictal_step = args.preictal_step
    interictal_step = args.interictal_step
    doing_STFT = args.doing_STFT
    doing_noise = args.doing_noise
    doing_lowpass_filter = args.doing_lowpass_filter
    target_preictal_interval = args.target_preictal_interval  # 15min
    preictal_interval = args.target_preictal_interval * 60  # 900s
    data_path = args.data_path
    print("data path : {}".format(data_path))
    # ch_num = 15
    i = 0
    scale = 0.5

    for patient in patient_list:
        # if patient == 'Dog_1':
        #     i += 1
        #     print("c")
        #     continue
        # if patient == 'Dog_2':
        #     i += 1
        #     print("c")
        #     continue
        # if patient == 'Dog_3':
        #     i += 1
        #     print("c")
        #     continue
        # if patient == 'Dog_4':
        #     i += 1
        #     print("c")
        #     continue
        # if patient == 'Dog_5':
        #     i += 1
        #     print("c")
        #     continue
        #
        # if patient == 'Patient_1':
        #     i += 1
        #     print("c")
        #     continue
        #
        # if patient == 'Patient_2':
        #     i += 1
        #     print("c")
        #

        if patient == 'Patient_1' or patient == 'Patient_2':
            sfreq = 512
        print("sfreq", sfreq)

        ch_num = ch_num_list[i]
        # create dir to save results
        if doing_STFT:
            mkdir("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s" % (data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise))
        else:
            mkdir("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s" % (data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise))
        stft_shape = ()
        # preprocessing ictal and preictal data  # 预处理癫痫发作和癫痫前期数据 # sfreq CHB 256 # window_length 5 # preictal_step 5 # interictal_step 5
        # for each edf file with seizure, a sliding window is used to transpose data into clips with shape (ch_num, window_length*sfreq)
        # 对于每个带有癫痫发作的edf文件，都使用滑动窗口将数据转置为具有形状的剪辑（ch_num，window_length*sfreq）
        # i is the i-th seizure of each patient # i是每个患者的第i次癫痫发作
        print("clipping ictal and preictal data")  # 剪辑癫痫和癫痫前期数据

        data_type = ['interictal', 'preictal']

        # targets = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
        import os

        # 指定目录路径
        dir_path = "%s/%s/%s/" % (data_path, patient, patient)

        # 创建保存序号的列表
        inter_list = []
        pre_list = []

        # 遍历目录中的所有文件并保存序号
        for filename in os.listdir(dir_path):
            # 提取 inter 和 pre 序号
            if 'interictal_segment' in filename:
                inter_list.append(int(filename.split('_')[-1].split('.')[0]))
            elif 'preictal_segment' in filename:
                pre_list.append(int(filename.split('_')[-1].split('.')[0]))

        # 查找 inter 序号最大的值
        inter_max = max(inter_list)
        print('Interictal Max:', inter_max)

        # 查找 pre 序号最大的值
        pre_max = max(pre_list)
        print('Preictal Max:', pre_max)

        seizure = seizure_list[i]
        x = pre_max // seizure
        print("seizure num count: ", x)
        # assert seizure == seizure_list[i]
        counter = 0

        # for target in targets:
        edges = []
        datatype = 'preictal'
        data_list = np.array([])
        for segment in range(1, pre_max + 1):
            # fname = '/data1/zhanghan/data/KAGGLE/%s/%s/%s_%s_segment_%04d.mat' % (patient, patient, patient, datatype, segment)
            fname = '/share/home/zhanghan/data/KAGGLE/%s/%s/%s_%s_segment_%04d.mat' % (patient, patient, patient, datatype, segment)
            mat = scipy.io.loadmat(fname)
            k = '%s_segment_%d' % (datatype, segment)
            d = mat[k]['data'][0, 0]
            edges.append((d[:, :1].astype(float), d[:, -1:].astype(float)))

            rawdata = np.array(d)
            if not data_list.any():
                data_list = rawdata
            else:
                data_list = np.hstack((data_list, rawdata))

            # data = self._raw_data.get_data().transpose(1, 0)  # (sample,channel)

            if segment % 6 == 0:

                # data_list = data_list.resample(256, n_jobs=16)
                data_list = resample(data_list, sfreq * 60 * 60, axis=1)  # (16 1438596) -> (16 921600) 256 (16 1438596) ->(16 1382400) 384
                data_list = data_list.transpose(1, 0)  # (sample,channel) (1382400 16 )
                print("segment %d" % segment)

                # process preictal data
                preictal_list = []
                preictal_count = 0
                while preictal_step * preictal_count + window_length <= 60 * 60:
                    preictal_start = preictal_step * preictal_count
                    preictal_end = preictal_step * preictal_count + window_length
                    # if preictal_start < 0 & preictal_end >= 0:
                    #     preictal_count += 1
                    #     continue
                    # if preictal_end == 0:
                    #     preictal_data = data_list[preictal_start * sfreq:]
                    # else:
                    preictal_data = data_list[preictal_start * sfreq: preictal_end * sfreq].copy()
                    # print(preictal_start, preictal_end)
                    if doing_STFT:
                        save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/" % (
                            data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)
                    else:
                        save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/" % (
                            data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)
                    # plot_time_intensity(preictal_data, save_path, len(preictal_list), i, classification="preictal")  # 画出interictal数据的时间强度图

                    # whether  doing_noise
                    x = random.random()
                    # print(x)
                    scale = 0.2
                    if doing_noise and (x < scale):
                        noise_data = wgn(preictal_data, 1)  # (22, 59, 114)
                        noise_data = getSpectral_STFT_T(noise_data, sfreq)  # (22, 59, 114)
                        preictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中

                        noise_data = wgn(preictal_data, 2)  # (22, 59, 114)
                        noise_data = getSpectral_STFT_T(noise_data, sfreq)  # (22, 59, 114)
                        preictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中

                        noise_data = wgn(preictal_data, 8)  # (22, 59, 114)
                        noise_data = getSpectral_STFT_T(noise_data, sfreq)  # (22, 59, 114)
                        preictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中
                        # preictal_count += 1  # 癫痫发作计数器加1
                    if not doing_noise and (x < scale):
                        _ = wgn(preictal_data, 1)  # (22, 59, 114)
                        _ = wgn(preictal_data, 2)  # (22, 59, 114)
                        _ = wgn(preictal_data, 8)  # (22, 59, 114)

                    x = random.random()
                    # print(x)
                    if doing_noise and (x < scale * 2):
                        shift_data = shiftdata(preictal_data)
                        shift_data = getSpectral_STFT_T(shift_data, sfreq)  # (22, 59, 114)
                        preictal_list.append(shift_data)  # 将癫痫发作数据添加到列表中
                    if not doing_noise and (x < scale * 2):
                        _ = shiftdata(preictal_data)

                    # whether doing stft
                    if doing_STFT:
                        preictal_data = getSpectral_STFT(preictal_data, sfreq)  # (22, 59, 114)
                        # print("doing STFT {}".format(preictal_data.shape))
                    preictal_list.append(preictal_data)
                    preictal_count += 1
                preictal_list = np.array(preictal_list)
                print("preictal count {} window_length {} preictal_step {}".format(preictal_count, window_length, preictal_step))

                # save preictal data to npy file
                if doing_STFT:
                    np.save(
                        "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/preictal%d.npy" % (
                            data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, counter),
                        preictal_list)
                else:
                    save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/preictal%d.npy" % (
                        data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, counter)
                    print("save to {}".format(save_path))
                    np.save(save_path, preictal_list)
                print("preictal shape {}\n".format(preictal_list.shape))
                stft_shape = preictal_list.shape

                counter += 1
                data_list = np.array([])

        # # 如果最后一批数据不足6个，则进行一次 numpy 存储
        # if len(data_list) > 0:
        #     np.save('data_batch_{}.npy'.format(counter + 1), np.array(data_list))

        # except Exception as e:
        #     print(e)
        #     print( )
        #     break

        inter_seq = [i + 1 for i in range(inter_max)]
        print("i :", i)
        random.shuffle(inter_seq)
        x = seizure_list[i]
        num_per_group = (inter_max + 1) // x  # 每份的元素数量
        print("seizure num :", x)

        # 等分inter_seq序列
        inter_seq_split = [inter_seq[i:i + num_per_group] for i in range(0, inter_max + 1, num_per_group)]

        datatype = 'interictal'
        interictalnum = inter_max // seizure_list[i]
        print("i :", i)
        counter = 0
        # for target in targets:
        edges = []
        data_list = np.array([])
        drop = -1
        print("interictalnum :", interictalnum)
        if interictalnum >= 3 * 6 * 6:  # 6是 60分钟 5是前期间期比例
            scale = 0.1
            # drop = 0.05
        elif interictalnum >= 2 * 6 * 6:  # 6是 60分钟 5是前期间期比例
            scale = 0.1
            # drop = 0.05
        elif interictalnum >= 6 * 6:
            scale = 0.4
        elif interictalnum < 3 * 6:
            scale = 0.6
        else:
            scale = 0.5
        print("scale :", scale)
        print("drop :", drop)

        for n in range(x):
            print("第{}份：{}".format(n + 1, inter_seq_split[n]))

            for segment in inter_seq_split[n]:

                # fname = '/data1/zhanghan/data/KAGGLE/%s/%s/%s_%s_segment_%04d.mat' % (patient, patient, patient, datatype, segment)
                fname = '/share/home/zhanghan/data/KAGGLE/%s/%s/%s_%s_segment_%04d.mat' % (patient, patient, patient, datatype, segment)
                mat = scipy.io.loadmat(fname)
                k = '%s_segment_%d' % (datatype, segment)
                d = mat[k]['data'][0, 0]
                edges.append((d[:, :1].astype(float), d[:, -1:].astype(float)))

                rawdata = np.array(d)
                if not data_list.any():
                    data_list = rawdata
                else:
                    # print(data_list.shape, rawdata.shape)
                    data_list = np.hstack((data_list, rawdata))

                # data = self._raw_data.get_data().transpose(1, 0)  # (sample,channel)

                # if segment % interictalnum == 0:
                # data_list = data_list.resample(256, n_jobs=16)
            print("data_list shape {}".format(data_list.shape))
            data_list = resample(data_list, sfreq * interictalnum * 10 * 60, axis=1)  # (16 1438596) -> (16 921600) 256 (16 1438596) ->(16 1382400) 384
            data_list = data_list.transpose(1, 0)  # (sample,channel) (1382400 16 )
            print("data_list resample transpose shape {}".format(data_list.shape))
            # process preictal data

            print("clipping interictal data")

            interictal_list_all = []  # interictal数据列表

            # process interictal数据
            interictal_list = []
            interictal_count = 0
            if doing_STFT:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/" % (
                    data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)
            else:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/" % (
                    data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)

            while interictal_step * interictal_count + window_length <= data_list.shape[0] // sfreq:  # 如果滑动窗口的长度小于edf文件的时长
                interictal_start = interictal_step * interictal_count
                interictal_end = interictal_step * interictal_count + window_length
                if interictal_end == 0:
                    interictal_data = data_list[interictal_start * sfreq:].copy()
                else:
                    interictal_data = data_list[interictal_start * sfreq:interictal_end * sfreq].copy()

                # plot_time_intensity(interictal_data, save_path, len(interictal_list), i)  # 画出interictal数据的时间强度图

                # whether  doing_noise
                dropx = random.random()
                x = random.random()
                # print(x)
                if doing_noise and (x < scale) and (dropx > drop):
                    noise_data = wgn(interictal_data, 8)  # (22, 59, 114)
                    noise_data = getSpectral_STFT_T(noise_data, sfreq)  # (22, 59, 114)
                    interictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中
                    # interictal_count += 1  # 癫痫发作计数器加1
                if not doing_noise and (x < scale) and (dropx > drop):
                    _ = wgn(interictal_data, 8)  # (22, 59, 114)
                    No_noise_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)
                    interictal_list.append(No_noise_data)  # 将癫痫发作数据添加到列表中

                x = random.random()
                # print(x)
                if doing_noise and (x < scale) and (dropx > drop):
                    shift_data = shiftdata(interictal_data)
                    shift_data = getSpectral_STFT_T(shift_data, sfreq)  # (22, 59, 114)
                    interictal_list.append(shift_data)  # 将癫痫发作数据添加到列表中
                if not doing_noise and (x < scale) and (dropx > drop):
                    _ = shiftdata(interictal_data)
                    No_noise_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)
                    interictal_list.append(No_noise_data)  # 将癫痫发作数据添加到列表中

                # whether doing stft
                if doing_STFT and (dropx > drop):
                    interictal_data = getSpectral_STFT(interictal_data, sfreq)  # (22, 59, 114)
                elif doing_STFT and (dropx <= drop):
                    interictal_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)

                interictal_list.append(interictal_data)
                interictal_count += 1
            data_list = np.array([])

            interictal_list = np.array(interictal_list)
            print("interictal shape {}".format(interictal_list.shape))
            print("interictal count {} window_length {} interictal_step {}".format(interictal_count, window_length, interictal_step))
            print("interictal {} ".format(i))

            # # concatenate interictal data
            # if len(interictal_list_all) == 0:
            #     interictal_list_all = interictal_list
            # else:
            #     interictal_list_all = np.vstack((interictal_list_all, interictal_list))  # 将interictal数据添加到列表中
            # print("all interictal shape: {}".format(interictal_list.shape))

            # #shuffle interictal data and divide into n gourps. n is the number of seizures of each patient #打乱interictal数据并将其分成n组。 n是每个患者的癫痫发作次数

            # # 创建索引列表
            # index_list = list(range(len(interictal_list_all)))
            # # 使用 zip() 函数将原始列表与索引列表组合
            # zipped_list = list(zip(interictal_list_all, index_list))
            # # 打乱组合后的列表
            # random.shuffle(zipped_list)
            # # 解压缩新列表，以获得打乱后的列表和相应的原始索引
            # shuffled_list, original_indices = zip(*zipped_list)
            # # 将结果转换为list类型
            # interictal_list_all = np.array(shuffled_list)
            # original_indices = np.array(original_indices)
            # # print("Shuffled list:", interictal_list_all)
            # # print("Original indices:", original_indices)
            # # CSV 文件路径
            # if doing_STFT:
            #     csv_file_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/interictal_list_indices.csv" % (
            #         data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)
            # else:
            #     csv_file_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/interictal_list_indices.csv" % (
            #         data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise)
            # # 检查文件是否存在，如果不存在，则创建文件并写入标题行
            # if not os.path.isfile(csv_file_path):
            #     with open(csv_file_path, "w", newline="") as csv_file:
            #         writer = csv.writer(csv_file, delimiter=",")
            #         writer.writerow(["shuffled_index", "original_indices"])
            # # 将数据写入 CSV 文件
            # with open(csv_file_path, "a", newline="") as csv_file:
            #     writer = csv.writer(csv_file, delimiter=",")
            #     # 如果文件非空，则在数据上面隔一行创建新数据
            #     if os.path.getsize(csv_file_path) > 0:
            #         writer.writerow([])
            #     # 将列表索引和值并列写入文件
            #     for j, value in enumerate(original_indices):
            #         writer.writerow([j, value])
            #     # # 将两个列表并列写入文件
            #     # for i in range(max(len(list1), len(list2))):
            #     #     value1 = list1[i] if i < len(list1) else ""
            #     #     value2 = list2[i] if i < len(list2) else ""
            #     #     writer.writerow([value1, value2])
            #
            # # #shuffle interictal data and divide into n gourps. n is the number of seizures of each patient #打乱interictal数据并将其分成n组。 n是每个患者的癫痫发作次数
            # # np.random.shuffle(interictal_list_all) #打乱interictal数据
            #
            # count = 0
            #
            # interictal_length = len(interictal_list_all) // seizure_list[i]  # 每个癫痫发作的数据长度
            # while (count + 1) * interictal_length <= len(interictal_list_all):  # 如果数据长度小于interictal数据的长度
            #     interictal_data = interictal_list_all[count * interictal_length: (count + 1) * interictal_length]  # 取出interictal数据
            interictal_data = interictal_list
            # print(interictal_data.shape)
            all_zero = np.any(interictal_data, axis=(1, 2, 3)) == False
            interictal_data = interictal_data[all_zero == False]
            # print(interictal_data.shape)
            print("FIN interictal shape {}".format(interictal_data.shape))

            # save interictal data to npy file
            if doing_STFT:
                np.save("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/interictal%d.npy" % (
                    data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, n), interictal_data)
            else:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s/interictal%d.npy" % (
                    data_path, patient, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, n)
                print("save to {}".format(save_path))
                np.save(save_path, interictal_data)
            print("interictal count {} : {}".format(n, interictal_data.shape))
            # count += 1
            print("interictal_length {}".format(n, interictalnum))

        i += 1
