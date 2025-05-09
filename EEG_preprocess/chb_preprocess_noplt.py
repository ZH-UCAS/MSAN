
import csv
import os
import glob
from datetime import datetime

import numpy as np
import random
from chb_edf_file import CHBEdfFile

from chb_stft import getSpectral_STFT, getSpectral_STFT_T
import matplotlib.pyplot as plt
import argparse
import sys

sys.path.append('')
from EEG_utils.eeg_utils import GetInputChannel, mkdir, GetPatientList, GetDataPath, GetSeizureList

import matplotlib.pyplot as plt
# import cupy as cp


def setup_seed(seed):  # 为CPU设置随机种子用于生成随机数，以使得结果是确定的
    '''
    set up random seed for numpy
    '''
    np.random.seed(seed)  # 为numpy设置随机种子
    random.seed(seed)  # 为python设置随机种子


def wgn(x, snr):

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

    def get_patient_name(self):
        """
        Get patient name
        """
        patient_list = GetPatientList("CHB")  # 获取病人列表
        return patient_list[str(self.patient_id)]  # 返回病人名字chbxx

    def get_seizure_time_list(self):
        """
        Get seizure time (second) in each EDF file
        for each patient, seizure times are stored in a list. [(start, end), (start, end),...]
        """
        seizure_time_list = {'1': [(2996, 3036), (1467, 1494), (1732, 1772), (1015, 1066),
                                   (1720, 1810), (327, 420), (1862, 1963)],
                             '2': [(130, 212), (2972, 3053), (3369, 3378)],
                             # '3' : [(432, 501),(2162, 2214),(1982, 2029),(2592, 2656),
                             #        (1725, 1778)], #
                             '3': [(731, 796), (432, 501), (2162, 2214), (1982, 2029),
                                   (2592, 2656), (1725, 1778)],  #
                             # '3' : [(362, 414),(731, 796),(432, 501),(2162, 2214),
                             #        (1982, 2029),(2592, 2656),(1725, 1778)], #
                             '5': [(417, 532), (1086, 1196), (2317, 2413), (2451, 2571),
                                   (2348, 2465)],
                             '6': [(1724, 1738), (7461, 7476), (13525, 13540), (6211, 6231),
                                   (12500, 12516), (7799, 7811), (9387, 9403)],  # 1
                             # '6' : [(327, 347),(6211, 6231),(12500, 12516),(10833, 10845),
                             #       (506,519),(7799, 7811),(9387, 9403)],#4(1) 4(2) 9
                             '7': [(4920, 5006), (3285, 3381), (13688, 13831)],
                             '8': [(2670, 2841), (2856, 3046), (2988, 3122), (2417, 2577),
                                   (2083, 2347)],
                             '9': [(12231, 12295), (2951, 3030), (9196, 9267), (5299, 5361)],
                             # '10': [(6313, 6348),(6888, 6958),(2382, 2447),(3021, 3079),
                             #        (3801, 3877),(4618, 4707),(1383, 1437)],#
                             '10': [(6888, 6958), (2382, 2447), (3021, 3079), (3801, 3877),
                                    (4618, 4707), (1383, 1437)],  # 20 27
                             '11': [(298, 320), (2695, 2727), (1454, 2206)],
                             '13': [(2077, 2121), (934, 1004), (2474, 2491), (3339, 3401),
                                    (851, 916)],  # 19 21 58 59 62(1)
                             # 						   '13': [(2077, 2121),(934, 1004),(142, 173),(458, 478),
                             # 				                  (2474, 2491)],# 19 21 40(1) 55(1) 58
                             '14': [(1986, 2000), (1372, 1392), (1911, 1925), (1838, 1879),
                                    (3239, 3259), (2833, 2849)],  # 3 4

                             '16': [(2290, 2299), (1120, 1129), (1214, 1220), (227, 236),
                                    (1694, 1700), (3290, 3298), (627, 635), (1909, 1916)],  # 10 11 14 16 17(1) 17(2) 17(4) 18(1) 18(2)
                             '17': [(2282, 2372), (3025, 3140), (3136, 3224)],
                             # 						   '18': [(3477, 3527),(2087, 2155),(1908, 1963),(2196, 2264)],#用4次
                             '18': [(3477, 3527), (541, 571), (2087, 2155), (1908, 1963),
                                    (2196, 2264), (463, 509)],  # 用所有6次
                             # '19': [(2964, 3041),(3159, 3240)],#用2次
                             #    '19': [(299, 377),(2964, 3041),(3159, 3240)],#用所有3次
                             '20': [(94, 123), (1440, 1470), (2498, 2537), (1971, 2009),
                                    (390, 425), (1689, 1738), (2226, 2261), (1393, 1432)],
                             '21': [(1288, 1344), (2627, 2677), (2003, 2084), (2553, 2565)],
                             '22': [(3367, 3425), (3139, 3213), (1263, 1335)],
                             '23': [(3962, 4075), (325, 345), (5104, 5151), (2589, 2660),
                                    (6885, 6947), (8505, 8532), (9580, 9664)]
                             }

        return seizure_time_list[str(self.patient_id)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDF preprocessing on CHB Dataset')
    parser.add_argument('--patient_id', type=int, default=1, metavar='patient id')
    parser.add_argument('--target_preictal_interval', type=int, default=15, metavar='how long we decide as preictal. Default set to 15 min')  # in minute
    parser.add_argument('--seed', type=int, default=1997, metavar='random seed')
    parser.add_argument('--ch_num', type=int, default=18, metavar='number of channel')
    parser.add_argument('--sfreq', type=int, default=256, metavar='sample frequency')
    parser.add_argument('--window_length', type=int, default=5, metavar='sliding window length')  # if stft : 30              else 5
    parser.add_argument('--preictal_step', type=int, default=1, metavar='step of sliding window (second) for preictal data')  # if stft : 5               else 5
    parser.add_argument('--interictal_step', type=int, default=1,
                        metavar='step of sliding window (second) for interictal data')  # if stft : 30             else 5
    parser.add_argument('--doing_STFT', type=bool, default=False, metavar='whether to do STFT')  # if stft : True           else False
    parser.add_argument('--doing_noise', type=bool, default=False, metavar='whether to do noise')
    parser.add_argument('--doing_lowpass_filter', type=bool, default=False, metavar='whether to do low pass filter')  # if stft : False else True
    parser.add_argument('--data_path', type=str, default="CHB", metavar='data path')

    args = parser.parse_args()
    setup_seed(args.seed)

    patient_list = GetPatientList("CHB")
    patient_id = args.patient_id
    patient_name = patient_list[str(patient_id)]
    sei_num = len(GetSeizureList("CHB")[str(patient_id)])
    print("seizure number : {}".format(sei_num))
    sfreq = args.sfreq
    window_length = args.window_length
    preictal_step = args.preictal_step
    interictal_step = args.interictal_step
    doing_STFT = args.doing_STFT
    doing_noise = args.doing_noise
    doing_lowpass_filter = args.doing_lowpass_filter
    target_preictal_interval = args.target_preictal_interval  # 15min
    preictal_interval = args.target_preictal_interval * 60  # 900s
    data_path = GetDataPath(args.data_path)
    print("data path : {}".format(data_path))
    ch_num = GetInputChannel("CHB", patient_id, args.ch_num)
    patient = CHBPatient(patient_id, data_path, ch_num, doing_lowpass_filter, target_preictal_interval)  # 读取病人的 发作 发作前期 发作间期 edf文件
    seizure_time_list = patient.get_seizure_time_list()  # 获取病人的发作时间列表
    print("\nprocessing patient : id {} {}\n seizure time {}\n".format(patient_id, patient_name, seizure_time_list))  # 打印病人的id，名字，发作时间列表
    scale = 0.25

    # create dir to save results
    if doing_STFT:
        mkdir("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s" % (
            data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale)))
    else:
        mkdir("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s" % (
            data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale)))
    stft_shape = ()

    print("clipping ictal and preictal data")  # 剪辑癫痫和癫痫前期数据
    for i, edf_file in enumerate(patient._edf_files_seizure):  # 对于每个带有癫痫发作的edf文件

        # preictal_interval is set to 900s(15min)
        preictal_interval = target_preictal_interval * 60  # preictal_interval设置为900s（15分钟）

        # load data from edf file
        print(edf_file.get_filepath())  # 打印edf文件路径
        ant_data = edf_file.get_preprocessed_data()  # 获取预处理后的数据
        print("seizure {} \n shape {}".format(i + 1, ant_data.shape))  # 打印癫痫发作的形状

        if seizure_time_list[i][0] < target_preictal_interval * 60:  # 15*60=900s
            print("seizure {} : preictal is not enough".format(i + 1))
            supplement_filepath = "%s/%s/seizure-supplement/%s-supplement.edf" % (data_path, patient_name, edf_file.get_filename())

            # 如果有补充文件 则补充前期数据到900s
            # if the supplement edf file exists, load the edf file
            if os.path.exists(supplement_filepath):
                supplement_file = CHBEdfFile(supplement_filepath, patient_id, ch_num)
                print("load supplement file : {}".format(supplement_filepath))
                ant_data2 = supplement_file.get_preprocessed_data()  # 获取补充数据
                print("original label {}".format(seizure_time_list[i]))
                seizure_time_list[i] = (
                    seizure_time_list[i][0] + supplement_file.get_file_duration(), seizure_time_list[i][1] + supplement_file.get_file_duration())
                print("new label {}".format(seizure_time_list[i]))
                ant_data = np.concatenate((ant_data2, ant_data))  # 将补充数据和原数据拼接
                print("new data {}".format(ant_data.shape))

            # 如果没有补充文件 则将前期时长缩短
            # if the supplement edf file does not exist, use as long as we have
            else:
                print("No supplement file")
                preictal_interval = seizure_time_list[i][0]  # 前期时长为癫痫发作开始时间
                # preictal_interval = target_preictal_interval * 60

        # process ictal data
        ictal_list = []  # 癫痫发作数据列表
        ictal_count = 0

        # txtpath = "%spreictal%dno.txt" % (patient_name, i)
        # np.savetxt(txtpath, ant_data)

        while seizure_time_list[i][0] + preictal_step * ictal_count + window_length <= seizure_time_list[i][1]:
            ictal_start = seizure_time_list[i][0] + preictal_step * ictal_count
            ictal_end = seizure_time_list[i][0] + preictal_step * ictal_count + window_length
            if ictal_end == 0:
                ictal_data = ant_data[ictal_start * sfreq:].copy()
            else:
                ictal_data = ant_data[ictal_start * sfreq: ictal_end * sfreq].copy()
                print("ictal_start * sfreq", ictal_start * sfreq, "ictal_end * sfreq", ictal_end * sfreq)


            # whether doing stft
            if doing_noise:

                noise_data = wgn(ictal_data, 8)  # (22, 59, 114)
                noise_data = getSpectral_STFT_T(noise_data)  # (22, 59, 114)
                ictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中

                # ictal_count += 1  # 癫痫发作计数器加1
                shift_data = shiftdata(ictal_data)
                shift_data = getSpectral_STFT_T(shift_data)  # (22, 59, 114)
                ictal_list.append(shift_data)  # 将癫痫发作数据添加到列表中
            # data1 = ictal_data
            if not doing_noise:
                _ = wgn(ictal_data, 8)  # (22, 59, 114)
                _ = shiftdata(ictal_data)
            # data2 = ictal_data

            # whether doing stft
            if doing_STFT:
                ictal_data = getSpectral_STFT(ictal_data)  # (22, 59, 114)
            # print(ictal_data)

            ictal_list.append(ictal_data)  # 将癫痫发作数据添加到列表中 (18,117,114)
            ictal_count += 1  # 癫痫发作计数器加1
        ictal_list = np.array(ictal_list)  # 将癫痫发作数据列表转换为数组
        print("ictal count {} window_length {} preictal_step {}".format(ictal_count, window_length, preictal_step))  # 打印癫痫发作计数器，窗口长度，癫痫前期步长

        # save ictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/ictal%d.npy" % (
                data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), i),
                    ictal_list)
        else:
            save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/ictal%d.npy" % (
                data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), i)
            print("save to {}".format(save_path))
            np.save(save_path, ictal_list)
        print("ictal shape {}".format(ictal_list.shape))

        # process preictal data
        preictal_list = []
        preictal_count = 0
        while seizure_time_list[i][0] + preictal_step * preictal_count + window_length - preictal_interval <= seizure_time_list[i][0]:
            preictal_start = seizure_time_list[i][0] + preictal_step * preictal_count - preictal_interval
            preictal_end = seizure_time_list[i][0] + preictal_step * preictal_count + window_length - preictal_interval
            # print("preictal_start:", preictal_start, "|preictal_end:", preictal_end)
            if preictal_start < 0 & preictal_end >= 0:
                preictal_count += 1
                continue
            if preictal_end == 0:
                preictal_data = ant_data[preictal_start * sfreq:].copy()
            else:
                preictal_data = ant_data[preictal_start * sfreq: preictal_end * sfreq].copy()

            if doing_STFT:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/" % (
                    data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale))
            else:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/" % (
                    data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale))
            # plot_time_intensity(preictal_data, save_path, len(preictal_list), i, classification="preictal")  # 画出interictal数据的时间强度图

            # whether  doing_noise
            x = random.random()
            # print(x)
            if doing_noise and (x < scale):

                noise_data = wgn(preictal_data, 8)  # (22, 59, 114)
                noise_data = getSpectral_STFT_T(noise_data)  # (22, 59, 114)
                preictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中
                # preictal_count += 1  # 癫痫发作计数器加1
            if not doing_noise and (x < scale):

                _ = wgn(preictal_data, 8)  # (22, 59, 114)

            x = random.random()
            # print(x)
            if doing_noise and (x < scale * 2):
                shift_data = shiftdata(preictal_data)
                shift_data = getSpectral_STFT_T(shift_data)  # (22, 59, 114)
                preictal_list.append(shift_data)  # 将癫痫发作数据添加到列表中
            if not doing_noise and (x < scale * 2):
                _ = shiftdata(preictal_data)

            # whether doing stft
            if doing_STFT:
                preictal_data = getSpectral_STFT(preictal_data)  # (22, 59, 114)
                # print("doing STFT {}".format(preictal_data.shape))
            preictal_list.append(preictal_data)
            preictal_count += 1
        preictal_list = np.array(preictal_list)
        print("preictal count {} window_length {} preictal_step {}".format(preictal_count, window_length, preictal_step))
        print("preictal {} preictal start {} preictal end {}".format(i, seizure_time_list[i][0] - preictal_interval, seizure_time_list[i][0]))
        print("all pre shape :", preictal_list.shape)
        # save preictal data to npy file
        if doing_STFT:
            np.save(
                "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/preictal%d.npy" % (
                    data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), i),
                preictal_list)
        else:
            save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/preictal%d.npy" % (
                data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), i)
            print("save to {}".format(save_path))
            np.save(save_path, preictal_list)
        print("preictal shape {}\n".format(preictal_list.shape))
        # print(preictal_list)
        # np.savetxt(txtpath, preictal_list)
        stft_shape = preictal_list.shape

    print("stft_shape ", stft_shape)
    if len(patient._edf_files_unseizure) >= 3 * 5 * sei_num or patient_id == 7:
        scale_int = 0.25
    elif len(patient._edf_files_unseizure) >= 5 * sei_num or patient_id == 9:
        scale_int = 0.5
    else:
        scale_int = 1
    print("len(patient._edf_files_unseizure) ", len(patient._edf_files_unseizure), "len(patient._edf_files_seizure) ", len(patient._edf_files_seizure))
    print("sei_num ", sei_num, "scale_int ", scale_int)
    print("clipping interictal data")

    interictal_list_all = []  # interictal数据列表
    for i, edf_file in enumerate(patient._edf_files_unseizure):  # 遍历每个edf文件
        # load data from edf file
        print(edf_file.get_filepath())
        ant_data = edf_file.get_preprocessed_data()
        print("unseizure {} \n shape {}".format(i + 1, ant_data.shape))  # 打印数据形状

        # process interictal数据
        interictal_list = []
        interictal_count = 0
        while interictal_step * interictal_count + window_length <= edf_file.get_file_duration():  # 如果滑动窗口的长度小于edf文件的时长
            interictal_start = interictal_step * interictal_count
            interictal_end = interictal_step * interictal_count + window_length
            if interictal_end == 0:
                interictal_data = ant_data[interictal_start * sfreq:].copy()
            else:
                interictal_data = ant_data[interictal_start * sfreq:interictal_end * sfreq].copy()

            if doing_STFT:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/" % (
                    data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, str(scale), doing_noise)
            else:
                save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/" % (
                    data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, str(scale), doing_noise)
            # plot_time_intensity(interictal_data, save_path, len(interictal_list), i)  # 画出interictal数据的时间强度图

            # whether  doing_noise
            x = random.random()
            # print(x)
            if doing_noise and (x < scale_int):
                noise_data = wgn(interictal_data, 8)  # (22, 59, 114)
                noise_data = getSpectral_STFT_T(noise_data)  # (22, 59, 114)
                interictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中
                # interictal_count += 1  # 癫痫发作计数器加1
            if not doing_noise and (x < scale_int):
                _ = wgn(interictal_data, 8)  # (22, 59, 114)
                No_noise_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)
                interictal_list.append(No_noise_data)  # 将癫痫发作数据添加到列表中

            # whether  doing_noise
            x = random.random()
            # print(x)
            if doing_noise and (x < scale_int):
                noise_data = wgn(interictal_data, 6)  # (22, 59, 114)
                noise_data = getSpectral_STFT_T(noise_data)  # (22, 59, 114)
                interictal_list.append(noise_data)  # 将癫痫发作数据添加到列表中
                # interictal_count += 1  # 癫痫发作计数器加1
            if not doing_noise and (x < scale_int):
                _ = wgn(interictal_data, 6)  # (22, 59, 114)
                No_noise_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)
                interictal_list.append(No_noise_data)  # 将癫痫发作数据添加到列表中

            x = random.random()
            # print(x)
            if doing_noise and (x < scale_int):
                shift_data = shiftdata(interictal_data)
                shift_data = getSpectral_STFT_T(shift_data)  # (22, 59, 114)
                interictal_list.append(shift_data)  # 将癫痫发作数据添加到列表中
            if not doing_noise and (x < scale_int):
                _ = shiftdata(interictal_data)
                No_noise_data = np.zeros((stft_shape[1], stft_shape[2], stft_shape[3]))  # (22, 59, 114)
                interictal_list.append(No_noise_data)  # 将癫痫发作数据添加到列表中

            # whether doing stft
            if doing_STFT:
                interictal_data = getSpectral_STFT(interictal_data)  # (22, 59, 114)

            interictal_list.append(interictal_data)
            interictal_count += 1
        interictal_list = np.array(interictal_list)
        print("interictal shape {}".format(interictal_list.shape))
        print("interictal count {} window_length {} interictal_step {}".format(interictal_count, window_length, interictal_step))
        print("interictal {} ".format(i))

        # concatenate interictal data
        if len(interictal_list_all) == 0:
            interictal_list_all = interictal_list
        else:
            interictal_list_all = np.vstack((interictal_list_all, interictal_list))  # 将interictal数据添加到列表中
        print("all interictal shape: {}".format(interictal_list_all.shape))

    # #shuffle interictal data and divide into n gourps. n is the number of seizures of each patient #打乱interictal数据并将其分成n组。 n是每个患者的癫痫发作次数

    # 创建索引列表
    index_list = list(range(len(interictal_list_all)))
    # 使用 zip() 函数将原始列表与索引列表组合
    zipped_list = list(zip(interictal_list_all, index_list))
    # 打乱组合后的列表
    random.shuffle(zipped_list)
    # 解压缩新列表，以获得打乱后的列表和相应的原始索引
    shuffled_list, original_indices = zip(*zipped_list)
    # 将结果转换为list类型
    interictal_list_all = np.array(shuffled_list)
    original_indices = np.array(original_indices)
    # print("Shuffled list:", interictal_list_all)
    # print("Original indices:", original_indices)
    # CSV 文件路径
    if doing_STFT:
        csv_file_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/interictal_list_indices_%d.csv" % (
            data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), patient_id)
    else:
        csv_file_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/interictal_list_indices_%d.csv" % (
            data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), patient_id)
    # 检查文件是否存在，如果不存在，则创建文件并写入标题行
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(["shuffled_index", "original_indices"])
    # 将数据写入 CSV 文件
    with open(csv_file_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        # 如果文件非空，则在数据上面隔一行创建新数据
        if os.path.getsize(csv_file_path) > 0:
            writer.writerow([])
        # 将列表索引和值并列写入文件
        for i, value in enumerate(original_indices):
            writer.writerow([i, value])

    count = 0
    interictal_length = len(interictal_list_all) // len(seizure_time_list)  # 每个癫痫发作的数据长度
    print("interictal_length all: {}".format(interictal_length))
    print("interictal_list_all: {}".format(interictal_list_all.shape))
    while (count + 1) * interictal_length <= len(interictal_list_all):  # 如果数据长度小于interictal数据的长度
        interictal_data = interictal_list_all[count * interictal_length: (count + 1) * interictal_length]  # 取出interictal数据

        print(interictal_data.shape)
        all_zero = np.any(interictal_data, axis=(1, 2, 3)) == False
        interictal_data = interictal_data[all_zero == False]
        print(interictal_data.shape)

        # save interictal data to npy file
        if doing_STFT:
            np.save("%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/interictal%d.npy" % (
                data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), count), interictal_data)
        else:
            save_path = "%s/%s/%dmin_%dstep_%dch_STFT_%s_Noise_%s_%s/interictal%d.npy" % (
                data_path, patient_name, target_preictal_interval, preictal_step, ch_num, doing_STFT, doing_noise, str(scale), count)
            print("save to {}".format(save_path))
            np.save(save_path, interictal_data)
        print("interictal count {} : {}".format(count, interictal_data.shape))
        count += 1
        print("interictal count {}  interictal_length {}".format(count, interictal_length))
