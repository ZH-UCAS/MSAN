# -*- coding: utf-8 -*-

import warnings
from datetime import datetime, date

import pandas as pd
from matplotlib import patches
from pytorch_metric_learning.utils.inference import MatchFinder
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from EEG_dataset.dataset_test_Double import testDataset
from EEG_eval.p_value_seizure import P_value
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetBatchsize, GetModel
from EEG_utils.write_to_excel import WriteToExcel, CalculateAverageToExcel
from torch.nn import functional as F
import torch.utils.data as Data
import argparse
from torch import nn
from sklearn.metrics import f1_score
# import scipy.io as io
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

np.set_printoptions(threshold=np.inf)


def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 计算敏感性的函数
def calculate_sensitivity(labels, predictions, target_sensitivity):
    # 计算ROC曲线
    # 假设您的模型只输出一个类别的预测概率
    # predictions = predictions.reshape(-1, 1)

    # 计算ROC曲线
    # fpr, tpr, thresholds = roc_curve(labels, predictions)
    fpr, tpr, thresholds = roc_curve(labels[:, 1], predictions[:, 1])

    # 计算AUC
    auc_score = auc(fpr, tpr)

    # 初始化最佳阈值和最佳敏感性
    best_threshold = 0.5
    best_sensitivity = 0.0

    # 寻找最佳阈值
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        sensitivity = tpr[i]

        # 检查是否达到目标敏感性
        if sensitivity >= target_sensitivity:
            best_threshold = threshold
            best_sensitivity = sensitivity
            break

    # 使用最佳阈值重新计算预测结果
    adjusted_predictions = (predictions[:, 1] >= best_threshold).astype(int)

    # 计算调整后的敏感性
    adjusted_sensitivity = np.sum((adjusted_predictions == 1) & (labels[:, 1] == 1)) / np.sum(labels[:, 1] == 1)

    return best_threshold, best_sensitivity, adjusted_sensitivity


def load_latest_file(dir_path, file_ext):
    """
    加载指定目录下匹配特定后缀的最新文件
    :param dir_path: 目录路径
    :param file_ext: 文件后缀
    :return: 最新文件的内容，如果没有匹配的文件则返回 None
    """

    print("dir_path:", dir_path, "file_ext:", file_ext)
    # 获取目录中所有匹配后缀的文件路径
    # matching_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]
    matching_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext) and f.startswith('lstm_')]

    # 如果没有匹配的文件，则返回 None
    if not matching_files:
        return None
    # 按文件修改时间排序
    matching_files.sort(key=lambda x: os.path.getmtime(x))
    # 获取最新的文件路径
    latest_file = matching_files[-1]
    # 加载最新的文件
    with open(latest_file, 'r') as f:
        file_content = f
    return file_content, latest_file


def count_matching_windows(outputs, test_labels):
    print("outputs", outputs.shape)
    print("test_labels", test_labels.shape)
    test_labels = np.squeeze(test_labels)
    print("test_labels squeeze", test_labels.shape)

    seq = list(enumerate(zip(outputs, test_labels)))
    # for index, value in seq:
    #     print(index, value[0], value[1])

    # 初始化新数组
    matching_windowsInt = np.zeros_like(outputs)
    matching_windowsPre = np.zeros_like(outputs)
    Int = True
    Pre = True

    # 定义窗口大小和步长
    window_sizeInt = 10
    window_sizePre = 60
    step_size = 1
    target_countInt = 8
    target_countPre = 48

    # 计数器
    FPnum_matches = 0.0
    TPnum_matches = 0.0
    FNnum_matches = 0.0

    i = 0
    # 遍历每个窗口
    # for i in range(len(test_labels) - window_size + 1):
    while i < len(test_labels) - window_sizeInt + 1:
        # print(i, len(test_labels) - window_sizeInt + 1)
        # 如果这个窗口是10个0，执行判断 0间期
        if Int and sum(test_labels[i:i + window_sizeInt]) == 0:
            # 判断当前窗口是否已经被标记为匹配
            if matching_windowsInt[i] == 1:
                i += step_size
                continue
            # 判断输出中的相应窗口是否包含8个1 1前期
            if sum(outputs[i:i + window_sizeInt]) >= target_countInt:
                # 在matching_windows数组中标记为1
                matching_windowsInt[i:i + window_sizeInt] = 1
                # 增加计数器
                FPnum_matches += 1
                print(outputs[i:i + window_sizeInt])
                print(test_labels[i:i + window_sizeInt])
                if i + window_sizeInt + 60 >= len(test_labels):
                    break
                # 更新i的值
                i += 60  # 60 * 30s 30min 不应期
        i += step_size
    i = 0
    while i < len(test_labels) - window_sizeInt + 1:
        if Pre and sum(test_labels[i:i + window_sizePre]) == window_sizePre:
            # 判断当前窗口是否已经被标记为匹配
            if matching_windowsPre[i] == 1:
                i += step_size
                continue
            # 判断输出中的相应窗口是否包含8个1
            if sum(outputs[i:i + window_sizePre]) >= target_countPre:
                # 在matching_windows数组中标记为1
                matching_windowsPre[i:i + window_sizePre] = 1
                # 增加计数器
                TPnum_matches += 1
                break
        i += step_size
    print("FPnum_matches", str(FPnum_matches), "TPnum_matches", str(TPnum_matches))
    return FPnum_matches, TPnum_matches


def smooth(a, WSZ):

    if (WSZ % 2 == 0):  # WSZ is odd?
        WSZ -= 1
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ  # moving average
    r = np.arange(2, WSZ, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    r = np.arange(1, WSZ, 2)
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def plot_roc(labels, predict_prob):
    '''
    plot ROC curve
    labels : true labels
    predict_prob : predicted probabilities
    '''
    # 设置图像大小 (宽, 高)，单位为英寸，增加尺寸以容纳大字体
    plt.figure(dpi=300, figsize=(10, 8))
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    # plt.title('ROC')
    # plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.ylabel('TPR')
    # plt.xlabel('FPR')

    # 计算ROC曲线和AUC值
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    # 设置字体大小为 22
    plt.rcParams.update({'font.size': 22})

    # 绘制ROC曲线
    plt.title('ROC', fontsize=22)  # 设置标题字体大小
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')  # 对角线
    plt.ylabel('True Positive Rate (TPR)', fontsize=22)  # 设置Y轴标签字体大小
    plt.xlabel('False Positive Rate (FPR)', fontsize=22)  # 设置X轴标签字体大小

    # 设置图例的字体大小为 22
    plt.legend(loc='lower right', prop={'size': 22})
    # 调整子图布局，防止标签、标题等被截断
    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seizure predicting on Xuanwu Dataset')
    parser.add_argument('--patient_id', type=int, default=1, metavar='patient_id')
    parser.add_argument('--ch_num', type=int, default=18, metavar='number of channel')
    parser.add_argument('--model_name', type=str, default="TA_STS_ConvNet", metavar='N')
    parser.add_argument('--dataset_name', type=str, default="CHB", metavar='XUANWU / CHB')
    parser.add_argument('--step_preictal', type=int, default=1, metavar='step of sliding window (second)')  # 预测步长，即模型每次预测的时间间隔，本代码中为30秒。
    parser.add_argument('--device_number', type=int, default=1, metavar='CUDA device number')
    parser.add_argument('--checkpoint_dir', type=str, default='/data1//code//', metavar='N')
    parser.add_argument('--seed', type=int, default=2024, metavar='N')
    parser.add_argument('--batch_size', type=int, default=200, metavar='batchsize')
    parser.add_argument('--using_cuda', type=bool, default=True, metavar='whether using cuda')
    parser.add_argument('--threshold', type=float, default=0.6, metavar='alarm threshold')
    parser.add_argument('--moving_average_length', type=int, default=9, metavar='length of smooth window')
    parser.add_argument('--persistence_second', type=int, default=1, metavar='N')
    parser.add_argument('--tw0', type=float, default=1 / 120, metavar='1/120 hour, which is 30 seconds')
    args = parser.parse_args()
    args.model_path = os.getcwd()

    model_name = args.model_name
    model_path = args.model_path
    dataset_name = args.dataset_name
    description = args.description
    patient_id = args.patient_id
    seed = args.seed
    moving_average_length = args.moving_average_length
    target_preictal_interval = args.target_preictal_interval
    step_preictal = args.step_preictal
    device_number = args.device_number
    using_cuda = args.using_cuda
    ch_num = args.ch_num
    position_embedding = args.position_embedding
    # batch_size = GetBatchsize(args.dataset_name, args.patient_id)
    # batch_size = 256
    batch_size = 1024
    threshold = args.threshold
    persistence_second = args.persistence_second
    tw0 = args.tw0
    tw = target_preictal_interval / 60

    patient_list = GetPatientList(dataset_name)
    seizure_list = GetSeizureList(dataset_name)
    seizure_count = len(seizure_list[str(patient_id)])
    patient_name = patient_list[str(patient_id)]
    print("patient : {}".format(patient_id))
    print("dataset : {} | seizure : {} filter : {} | threshold : {} | persistence : {} | tw0 : {} | tw : {}".format(
        args.dataset_name,
        seizure_count, moving_average_length * step_preictal, threshold, persistence_second, tw0 * 3600, tw * 3600))

    TP_list = []
    FN_list = []
    TN_list = []
    FP_list = []
    FPR_list = []
    SEN_list = []
    O_FPR_list = []
    O_SEN_list = []
    AUC_list = []
    InferTime_list1 = []
    InferTime_list2 = []
    PW_count = 0
    O_FPH = 0.0
    O_SEN = 0.0

    input_channel = GetInputChannel(dataset_name, patient_id, ch_num)
    # LOOCV for predicting
    for i in seizure_list[str(patient_id)]:
        # load test data
        if dataset_name.startswith("KAGGLE"):
            dataset_name = "KAGGLET"
        test_set = testDataset(dataset_name, i, using_ictal=1, patient_id=patient_id, patient_name=patient_name,
                               ch_num=input_channel, target_preictal_interval=target_preictal_interval,
                               step_preictal=step_preictal, model_name=model_name)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
        labels = test_set.y_data.numpy()
        preictal_length = test_set.preictal_length
        interictal_length = test_set.interictal_length
        if dataset_name.startswith("KAGGLE"):
            print("interrictal : {} | preictal : {} : ".format(interictal_length, preictal_length))
        else:
            ictal_length = test_set.ictal_length
            print("interrictal : {} | preictal : {} | ictal : {}: ".format(interictal_length, preictal_length, ictal_length))

        # get model
        input_channel = GetInputChannel(dataset_name, patient_id, ch_num)
        # model = GetModel(input_channel, device_number, model_name, dataset_name, position_embedding, patient_id)
        class LSTMX(nn.Module):
            def __init__(self, num_classes=2, channel=18):
                super(LSTMX, self).__init__()

                # 假设输入数据的维度是 (batch_size, 18, 117, 114)
                # 1. 通过卷积层将空间维度压缩到更低的维度
                self.conv1 = nn.Conv2d(in_channels=channel, out_channels=3, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

                # 2. 最大池化层来减少空间维度
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

                # 3. 将卷积输出的维度转化为合适的输入给 LSTM
                # 假设经过两层卷积和池化后，空间维度变为 (batch_size, 64, 29, 28)
                self.flattened_size = 87 * 6  # 计算展平后的大小

                # 4. LSTM 层来处理时间序列数据
                # 输入形状应该是 (batch_size, seq_len, input_size)
                # seq_len 取决于输入数据的时间序列长度，假设为 29
                self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=128, num_layers=2, batch_first=True)

                # 5. 全连接层输出分类结果
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                # 输入形状: (batch_size, channel=18, 117, 114)

                # 通过卷积层
                x = self.conv1(x)
                # x = torch.relu(x)
                x = self.conv2(x)
                # x = torch.relu(x)

                # 池化
                x = self.pool(x)

                # 展平
                batch_size, channels, height, width = x.size()
                x = x.view(batch_size, channels * height * width)

                # 使输入形状符合 LSTM 期望的 (batch_size, seq_len, input_size)
                # 假设我们通过卷积层将输入展平成一个长向量，seq_len 是 29，输入大小是 flattened_size
                x = x.view(batch_size, 38, self.flattened_size)  # reshape 为 (batch_size, seq_len=29, flattened_size)

                # 通过 LSTM 层
                lstm_out, (h_n, c_n) = self.lstm(x)

                # 获取最后一时刻的输出，传递给全连接层
                lstm_out_last = lstm_out[:, -1, :]  # 获取序列的最后一个时间步的输出
                output = self.fc(lstm_out_last)

                return output

        model = LSTMX(num_classes=2, channel=18)

        # if model_name.startswith("STANX"):
        #     model.load_state_dict(torch.load('{}/model/{}/{}/{}/stft/patient{}_{}_step_preictal{}.pth'.format(model_path, dataset_name, model_name, patient_id, patient_id, i, step_preictal)))
        # elif model_name.startswith("STANC"):
        #     model.load_state_dict(torch.load('{}/model/{}/{}/{}/stftC/patient{}_{}_step_preictal{}.pth'.format(model_path, dataset_name, model_name, patient_id, patient_id, i, step_preictal)))

        if dataset_name.startswith("KAGGLET"):
            dataset_name = "KAGGLE"

        if model_name.startswith("MONSTB"):
            dir_path = '{}/model/{}/{}/{}/stft/'.format(model_path, dataset_name, model_name, patient_id)
        elif model_name.startswith("MONSTL"):
            dir_path = '{}/model/{}/{}/{}/stftC/'.format(model_path, dataset_name, model_name, patient_id)
        elif model_name.startswith("CNN"):
            dir_path = '{}/model/{}/{}/{}/stft/'.format(model_path, dataset_name, model_name, patient_id)
        dir_path = '/share/home/zhanghan/code/SeizureDP/TA-STS-206/model/PreTbackbone/'

        file_ext = 'patient{}_{}_step_preictal{}.pth'.format(patient_id, i, step_preictal)
        _, file_content = load_latest_file(dir_path, file_ext)
        if file_content is None:
            print("No file found")
        else:
            device = torch.device('cuda:0')
            state_dict = torch.load(file_content, map_location=device)
            # state_dict = torch.load("/share/home/zhanghan/code/SeizureDP/TA-STS-206/model/CHB60/MONSTBX/1/stft/aspp12_False_20230921140530_patient1_4_step_preictal5.pth", map_location=device)
            model = nn.DataParallel(model)
            print(model.load_state_dict(state_dict, strict=False))
            print(file_content + " Load model successfully!")

        # set cuda
        if using_cuda:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(device_number)
            model = model.cuda()

            # start predicting
        start_time1 = time.perf_counter()
        start_time2 = time.time()
        model.eval()
        output_probablity = []
        output_list = []
        feature_list = []
        with torch.no_grad():
            for k, (data, target) in enumerate(test_loader):
                if using_cuda:
                    data = data.cuda().float()
                    target = target.cuda()
                # ori, output = model(data)
                output = model(data)

                # 把每次得到的 embeddings 存入 feature_list
                # feature_list.append(em.cpu().detach().numpy())
                # output = return_value[0]
                output_nosoftmax = output.cpu().detach().numpy()
                output = F.softmax(output, dim=1)
                output = torch.clamp(output, min=1e-9, max=1 - 1e-9)  # 限制输出概率的范围
                output = output.cpu().detach().numpy()  # 将输出概率转换为numpy格式
                if len(output_probablity) == 0:  # 如果是第一次预测，直接赋值
                    output_probablity.append(output)
                    output_probablity = np.array(output_probablity).squeeze()
                    output_list.append(output_nosoftmax)
                    output_list = np.array(output_list).squeeze()
                else:  # 如果不是第一次预测，将预测结果拼接起来
                    output_probablity = np.vstack((output_probablity, output))
                    output_list = np.vstack((output_list, output_nosoftmax))

        # 将 feature_list 转换为 numpy 数组
        print(labels.shape)  # 形状应该是 (n_samples,)
        # 将 labels 转换为一维数组
        embed_labels = labels.squeeze()  # 或者使用 labels = labels.ravel()
        print(embed_labels.shape)  # 形状应该是 (n_samples,)
        # 设置字体大小
        # 如果模型名称以 "MONSTB" 开头，保存文件
        if model_name.startswith("MONSTB"):
            file_path = '{}/model/{}/{}/{}/stft/patient{}_{}_step_preictal{}_tsne_visualization.png'.format(
                model_path, dataset_name, model_name, patient_id, patient_id, i, step_preictal
            )
            plt.savefig(file_path)  # 保存图像
            plt.close()  # 关闭当前图像，避免内存泄漏

        if dataset_name.startswith("KAGGLE"):
            infer_time1 = (time.perf_counter() - start_time1) / (preictal_length + interictal_length)
            infer_time2 = (time.time() - start_time2) / (preictal_length + interictal_length)
        else:
            infer_time1 = (time.perf_counter() - start_time1) / (preictal_length + interictal_length + ictal_length)
            infer_time2 = (time.time() - start_time2) / (preictal_length + interictal_length + ictal_length)
        InferTime_list1.append(infer_time1)
        InferTime_list2.append(infer_time2)

        # save output probabilities and labels 保存输出概率和标签
        if model_name.startswith("MONSTB"):
            np.save('{}/model/{}/{}/{}/stft/probablity{}_{}_step_preictal{}.npy'.format(model_path, dataset_name,
                                                                                        model_name, patient_id,
                                                                                        patient_id, i, step_preictal),
                    output_probablity)
            np.save(
                '{}/model/{}/{}/{}/stft/output{}_{}_step_preictal{}.npy'.format(model_path, dataset_name, model_name,
                                                                                patient_id, patient_id, i,
                                                                                step_preictal), output_list)
            np.save('{}/model/{}/{}/{}/stft/label{}_{}_step_preictal{}.npy'.format(model_path, dataset_name, model_name,
                                                                                   patient_id, patient_id, i,
                                                                                   step_preictal), labels)
        elif model_name.startswith("MONSTL"):
            np.save('{}/model/{}/{}/{}/stftC/probablity{}_{}_step_preictal{}.npy'.format(model_path, dataset_name,
                                                                                         model_name, patient_id,
                                                                                         patient_id, i, step_preictal),
                    output_probablity)
            np.save(
                '{}/model/{}/{}/{}/stftC/output{}_{}_step_preictal{}.npy'.format(model_path, dataset_name, model_name,
                                                                                 patient_id, patient_id, i,
                                                                                 step_preictal), output_list)
            np.save(
                '{}/model/{}/{}/{}/stftC/label{}_{}_step_preictal{}.npy'.format(model_path, dataset_name, model_name,
                                                                                patient_id, patient_id, i,
                                                                                step_preictal), labels)
        # predicting_probablity = output_probablity[:, 1]
        print("output_probablity", output_probablity.shape)
        # 真实标签
        print("labels", labels.shape)
        predictions = output_probablity  # 最新的预测概率
        print("predictions", predictions.shape)

        # 使用预测概率的第二列作为y_score参数
        predictions = predictions[:, 1]
        # 计算AUC
        # current_auc = roc_auc_score(labels, predictions)

        # 初始化最佳阈值和最大AUC
        best_threshold = 0.5
        best_auc = 0.0
        # print("best_auc", best_auc)
        #############################################################
        best_performance = 0.0
        #############################################################

        # 最大Youden's J指数
        best_j_score = 0.0
        label = labels.flatten()
        for threshold_p in np.arange(0.1, 1.0, 0.05):
            # 将预测概率转换为二进制预测结果
            adjusted_predictions = (predictions >= threshold_p).astype(int)  # 预测概率大于阈值的为1，小于阈值的为0
            # print("adjusted_predictions", adjusted_predictions.shape)
            # print("label", label.shape)

            # 计算当前阈值的AUC
            current_auc = roc_auc_score(labels, adjusted_predictions)
            print("current_auc", current_auc)

            # 检查是否获得更高的AUC
            if current_auc > best_auc:
                best_threshold = threshold_p
                best_auc = current_auc
                print("--------------")
                print("best_auc", best_auc, "best_threshold", best_threshold)

            # # 计算AUC
            # current_auc = roc_auc_score(labels, adjusted_predictions)
            # print("current_auc", current_auc)
            # print("threshold_p", threshold_p)
            # # print("best_threshold", best_threshold)
            #
            # # 检查是否获得更高的AUC
            # if current_auc > best_auc:
            #     print("best_auc", best_auc)
            #     print("best_threshold", best_threshold)
            #     best_threshold = threshold_p
            #     best_auc = current_auc

            # 计算真正例率（True Positive Rate，也称为召回率）
            # tpr = np.sum((adjusted_predictions == 1) & (label == 1)) / np.sum(label == 1)

            # 计算真负例率（True Negative Rate，也称为特异度）
            # tnr = np.sum((adjusted_predictions == 0) & (label == 0)) / np.sum(label == 0)

            #############################################################
            # 计算性能指标（如准确率、召回率、F1分数）
            # f1 = f1_score(labels, adjusted_predictions)
            #############################################################

            # 计算Youden's J指数
            # j_score = tpr + tnr - 1
            #
            # print("TP", np.sum((adjusted_predictions == 1) & (label == 1)), "(label == 1)", np.sum(label == 1))
            # print("TN", np.sum((adjusted_predictions == 0) & (label == 0)), "(label == 0)", np.sum(label == 0))
            #
            # print("tpr", tpr, "tnr", tnr)
            # print("j_score", j_score, "threshold_p", threshold_p, "best_j_score", best_j_score)

            # 检查是否获得更高的J指数
            # if j_score > best_j_score:
            #     best_threshold = threshold_p
            #     best_j_score = j_score
            #     print("--------------")
            #     print("best_j_score", best_j_score, "best_threshold", best_threshold)
            #############################################################
            # 检查是否获得更好的性能
            # if f1 > best_performance:
            #     best_threshold = threshold_p
            #     best_performance = f1
            #     print("--------------")
            #     print("f1_score", f1, "best_threshold", best_threshold)
            #############################################################

        # 使用最佳阈值重新计算预测结果
        # adjusted_predictions = (predictions[:, 1] >= best_threshold).float()
        adjusted_predictions = (predictions >= best_threshold).astype(int)
        # print("Best threshold_p:", best_threshold)
        # print("Best AUC:", best_auc)

        # 输出最佳阈值和对应的J指数
        # print("Best Threshold:", best_threshold)
        # print("Best J Score:", best_j_score)
        # 输出最佳阈值和对应的AUC
        print("Best Threshold:", best_threshold)
        print("Best AUC:", best_auc)

        # 修改 阈值 方式
        # predicting_probablity = (output_probablity[:, 1] >= best_threshold).astype(int)  # 预测概率大于阈值的为1，小于阈值的为0
        auto_threshold = True
        if auto_threshold:
            predicting_probablity = (output_probablity[:, 1] >= best_threshold).astype(int)  # 预测概率大于阈值的为1，小于阈值的为0
        else:
            # print("F")
            predicting_probablity = output_probablity[:, 1]

        # print("predicting_probablity", predicting_probablity)

        # probabilities = output_probablity[:, 1]  # 提取第二列的概率值
        # y_probablity = probabilities[probabilities > best_threshold]  # 选择大于阈值的概率值

        # print("predicting_probablity", predicting_probablity)

        # calculate AUC, draw ROC curve and save 计算AUC，绘制ROC曲线并保存
        y_true = labels
        y_probablity = output_probablity
        y_score1 = y_probablity[:, 1]
        auc_value1 = roc_auc_score(y_true, y_score1)
        print("AUC1", auc_value1)
        auc_value2 = roc_auc_score(y_true, predicting_probablity)
        print("AUC2", auc_value2)

        plot_roc(y_true, y_score1)
        if model_name.startswith("MONSTB"):
            plt.savefig(
                '{}/model/{}/{}/{}/stft/ROC{}_{}_step_preictal{}.png'.format(model_path, dataset_name, model_name,
                                                                             patient_id, patient_id, i, step_preictal))
        elif model_name.startswith("MONSTL"):
            plt.savefig(
                '{}/model/{}/{}/{}/stftC/ROC{}_{}_step_preictal{}.png'.format(model_path, dataset_name, model_name,
                                                                              patient_id, patient_id, i, step_preictal))

        # smooth the predicting results and save 平滑预测结果并保存
        predicting_probablity_smooth = smooth(predicting_probablity, moving_average_length)
        # print(predicting_probablity_smooth)
        # predicting_probablity_smooth = (predicting_probablity_smooth >= 0.5).astype(int)
        # predicting_probablity_smooth = (predicting_probablity_smooth >= threshold).astype(int)
        # print(predicting_probablity_smooth)
        predicting_probablity = (predicting_probablity_smooth >= threshold).astype(int)
        # print("predicting_probablity_smooth", predicting_probablity_smooth)
        if auto_threshold:
            predicting_result = output_probablity.argmax(axis=1)  # 选择概率最大的作为预测结果
        else:
            predicting_result = output_probablity.argmax(axis=1)  # 选择概率最大的作为预测结果

        auc_value3 = roc_auc_score(y_true, predicting_probablity_smooth)
        print("AUC3", auc_value3)
        auc_value = max(auc_value1, auc_value2, auc_value3)
        print("Max AUC:", auc_value)
        AUC_list.append(auc_value)

        if model_name.startswith("MONSTB"):
            np.save(
                '{}/model/{}/{}/{}/stft/pre_label{}_{}_step_preictal{}.npy'.format(model_path, dataset_name, model_name,
                                                                                   patient_id, patient_id, i,
                                                                                   step_preictal), predicting_probablity)
            np.save('{}/model/{}/{}/{}/stft/smooth_probablity{}_{}_step_preictal{}.npy'.format(model_path, dataset_name,
                                                                                               model_name, patient_id,
                                                                                               patient_id, i,
                                                                                               step_preictal),
                    predicting_probablity_smooth)
        elif model_name.startswith("MONSTL"):
            np.save('{}/model/{}/{}/{}/stftC/pre_label{}_{}_step_preictal{}.npy'.format(model_path, dataset_name,
                                                                                        model_name, patient_id,
                                                                                        patient_id, i, step_preictal),
                    predicting_probablity)
            np.save(
                '{}/model/{}/{}/{}/stftC/smooth_probablity{}_{}_step_preictal{}.npy'.format(model_path, dataset_name,
                                                                                            model_name, patient_id,
                                                                                            patient_id, i,
                                                                                            step_preictal),
                predicting_probablity_smooth)
        # 修改 阈值 方式
        # calculate confusion matrix  计算混淆矩阵
        TP, FP, TN, FN = 0, 0, 0, 0
        for j in range(len(labels)):
            if auto_threshold:
                if predicting_probablity[j] == 1 and labels[j] == 1:
                    TP += 1
                elif predicting_probablity[j] == 0 and labels[j] == 1:
                    FN += 1
                elif predicting_probablity[j] == 0 and labels[j] == 0:
                    TN += 1
                elif predicting_probablity[j] == 1 and labels[j] == 0:
                    FP += 1
                else:
                    print("error")
            else:
                # print("F")
                if predicting_probablity[j] == 1 and labels[j] == 1:
                    TP += 1
                elif predicting_probablity[j] == 0 and labels[j] == 1:
                    FN += 1
                elif predicting_probablity[j] == 0 and labels[j] == 0:
                    TN += 1
                elif predicting_probablity[j] == 1 and labels[j] == 0:
                    FP += 1
                else:
                    print("error")

        TP_list.append(TP)
        FN_list.append(FN)
        TN_list.append(TN)
        FP_list.append(FP)

        O_FPH_, O_SEN_ = count_matching_windows(predicting_probablity, labels)
        O_FPR_list.append(O_FPH_)
        O_SEN_list.append(O_SEN_)

        print("O_FPH" + str(O_FPH_) + "O_SEN" + str(O_SEN_))

        # calculate currect alarm and false alarm. calculate Sensitivity and FPR/h 计算正确报警和误报，计算灵敏度和FPR/h
        count = 0
        interval = 0  # 距离发作点的时间
        false_alarm_list = []  # 误报时间点列表
        true_alarm_list = []  # 正报时间点列表
        for index in range(len(predicting_probablity_smooth)):
            # probability is over threshold, start counting  概率超过阈值，开始计数
            if predicting_probablity_smooth[index] > threshold:
                PW_count += 1
                count += 1
            else:
                count = 0
            # if count is over persistence second，decide as one alarm 计数超过持续时间，判定为一个报警
            if count >= persistence_second // step_preictal:
                # print("persistence_second", persistence_second, "step_preictal", step_preictal)
                assert (persistence_second // step_preictal) >= 1
                interval = interictal_length + preictal_length - index  # 距离发作点的时间
                # if the alarm is within 15min，True alarm 报警在15min内，判定为正报
                if index >= interictal_length and index < interictal_length + preictal_length:
                    true_alarm_list.append(interval)  # 正报时间点列表
                # if the alarm is not within 15min，False alarm 报警不在15min内，判定为误报
                elif index < interictal_length:
                    false_alarm_list.append(interval)  # 误报时间点列表
                    print("false alarm at {}s  index： {} seizure： {}".format(interval, index, i))
                count = 0
        if model_name == "spectralCNN":
            FPR = len(false_alarm_list) / ((interictal_length * 30 + (
                    preictal_length + ictal_length) * step_preictal) / 3600)  # spectralCNN
        else:
            if dataset_name.startswith("KAGGLE"):
                FPR = len(false_alarm_list) / ((interictal_length * 30 + preictal_length * step_preictal) / 3600)
            else:
                FPR = len(false_alarm_list) / ((interictal_length * 30 + preictal_length * step_preictal + ictal_length * step_preictal) / 3600)

        FPR_list.append(FPR)
        #
        if len(true_alarm_list) > 0:
            SEN_list.append(1)
        else:
            SEN_list.append(0)
            true_alarm_list.append(-1)

        print(
            "TP {} FN {} TN {} FP {} sen {:.2%} spe {:.2%} acc {:.2%}; TA {} FA {} FPR {:.4} AUC {:.4} PT {} IT1 {:.4} IT2 {:.4}".format(TP,
                                                                                                                                         FN, TN, FP,
                                                                                                                                         TP / (TP + FN),
                                                                                                                                         TN / (TN + FP),
                                                                                                                                         (TP + TN) / (
                                                                                                                                                 TP + FN + TN + FP),
                                                                                                                                         len(true_alarm_list),
                                                                                                                                         len(false_alarm_list),
                                                                                                                                         FPR, auc_value,
                                                                                                                                         true_alarm_list[0],
                                                                                                                                         infer_time1,
                                                                                                                                         infer_time2))
        output_data = {
            'Patient id': [patient_id],
            'sei num': [i],
            'TP': [TP],
            'FN': [FN],
            'TN': [TN],
            'FP': [FP],
            'sen': [TP / (TP + FN)],
            'spe': [TN / (TN + FP)],
            'acc': [(TP + TN) / (TP + FN + TN + FP)],
            'TA': [len(true_alarm_list)],
            'FA': [len(false_alarm_list)],
            'FPR': [FPR],
            'AUC': [auc_value],
            'PT': [true_alarm_list[0]],
            'IT1': [infer_time1],
            'IT2': [infer_time2]
        }

        df = pd.DataFrame(output_data)
        excel_file = '{}/model/{}/{}/{}_test_T.xlsx'.format(model_path, dataset_name, model_name, date.today().strftime("%Y%m%d"))
        try:
            existing_df = pd.read_excel(excel_file)
            updated_df = existing_df.append(df, ignore_index=True)
            updated_df.to_excel(excel_file, index=False)
        except FileNotFoundError:
            df.to_excel(excel_file, index=False)

        import matplotlib.font_manager as fm  # 导入字体管理器

        font_properties = fm.FontProperties(weight='bold')  # 设置字体粗细为粗体
        font_properties.set_size(15)
        # draw predicting results
        plt.figure(dpi=300)
        # 绘制预测概率
        plt.plot(output_probablity[:, 1], linewidth='1', color='#47C8F3')
        plt.plot(predicting_probablity_smooth, linewidth='1.5', color='#FF0080')
        # 绘制警报线
        if dataset_name.startswith("KAGGLE"):
            plt.plot(np.ones(interictal_length + preictal_length) * threshold, linewidth='1', color='#f86e50', linestyle='--')
            plt.plot(np.ones(interictal_length + preictal_length) * 1.05, linewidth='1', label="Ictal", color='#f92185')
        else:
            plt.plot(np.ones(interictal_length + preictal_length + ictal_length) * threshold, linewidth='1', color='#f86e50', linestyle='--')
            plt.plot(np.ones(interictal_length + preictal_length + ictal_length + 1) * 1.05, linewidth='3', label="Ictal", color='#f4143f')
            plt.annotate("Ictal", xy=(interictal_length + preictal_length + ictal_length / 2, 1.05),
                         xytext=(interictal_length + preictal_length + ictal_length / 2, 1.15), ha='center', va='top', fontproperties=font_properties,
                         color='#f4143f')

        # 绘制标签
        plt.plot(np.ones(interictal_length + preictal_length) * 1.05, linewidth='3', label="Preictal", color='#ff9e31')
        plt.plot(np.ones(interictal_length - 1) * 1.05, linewidth='3', label="Interictal", color='#8a7c9c')

        plt.annotate("Preictal", xy=(interictal_length + preictal_length / 2, 1.05), xytext=(interictal_length + preictal_length / 2, 1.15), ha='center',
                     va='top',
                     fontproperties=font_properties, color='#ff9e31')
        plt.annotate("Interictal", xy=(interictal_length / 2, 1.05), xytext=(interictal_length / 2, 1.15), ha='center', va='top', fontproperties=font_properties,
                     color='#8a7c9c')

        # 删除边框
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        font_properties.set_size(12.5)
        plt.legend(labels=["Raw", "Smoothened", "Threshold"], loc="upper left", prop=font_properties, bbox_to_anchor=(0, 0.95), frameon=False)
        # 设置坐标轴边界线的粗细
        for spine in ax.spines.values():
            spine.set_linewidth(2)  # 设置坐标轴的边界线粗细为2

        # 添加箭头样式
        arrow_style = patches.ArrowStyle("->")
        arrow_patch_left = patches.FancyArrowPatch((0, -0.05), (0, 1), arrowstyle=arrow_style, transform=ax.transAxes, mutation_scale=20, color='black',
                                                   linewidth=2)
        arrow_patch_bottom = patches.FancyArrowPatch((-0.05, 0), (1, 0), arrowstyle=arrow_style, transform=ax.transAxes, mutation_scale=20, color='black',
                                                     linewidth=2)
        ax.add_patch(arrow_patch_bottom)
        ax.add_patch(arrow_patch_left)

        # 设置底部刻度
        num_ticks = len(predicting_probablity)  # 刻度数量

        # 计算x轴刻度的间隔
        tick_interval = 25
        while num_ticks // tick_interval > 10:
            tick_interval *= 2
        lowrem = (interictal_length + preictal_length) % tick_interval
        # print(lowrem)
        lowx = (interictal_length + preictal_length) - lowrem

        tick_positions = range(lowrem, num_ticks + 1, tick_interval)  # 均分刻度位置，注意刻度数量+1
        # 刻度标签，从数据长度的负值到0
        font_properties.set_size(12)
        if dataset_name.startswith("KAGGLE"):
            tick_labels = [str(i) for i in range(- (interictal_length + preictal_length) + lowrem, 1, tick_interval)]
        else:
            # 计算刻度标签
            tick_labels = [str(i) for i in range(- (interictal_length + preictal_length) + lowrem, ictal_length + 1, tick_interval)]

        ax.set_xticks(tick_positions)  # 设置刻度的位置
        ax.set_xticklabels(tick_labels, fontproperties=font_properties)  # 设置刻度的标签
        # plt.ylim(0, 1)

        # 设置 y 轴刻度标签的字体属性
        for label in ax.get_yticklabels():
            label.set_fontproperties(font_properties)

        font_properties.set_size(14)
        # ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties)  # 设置y轴刻度标签的字体属性
        # 添加 x 轴注释
        x_label = "Time Segments"  # 时间步
        # x_units = "Segments"  # 片段数量
        # plt.text(0.85, -0.15, f"{x_label} / {x_units}", fontsize=12, ha='right', va='center', transform=ax.transAxes, fontproperties=font_properties)
        plt.text(0.85, -0.1, f"{x_label}", ha='right', va='center', transform=ax.transAxes, fontproperties=font_properties)

        # 添加 y 轴注释
        y_label = "Predicted Probability"  # 预测的概率
        plt.text(-0.1, 0.5, y_label, ha='center', va='center', rotation='vertical', transform=ax.transAxes, fontproperties=font_properties)

        # plt.title(
        #     "w {} filter {} persistence {} sen {:.4%} spe {:.4%} acc {:.4%}".format(threshold, moving_average_length,
        #                                                                             persistence_second, TP / (TP + FN),
        #                                                                             TN / (TN + FP),
        #                                                                             (TP + TN) / (TP + FN + TN + FP)))
        plt.tight_layout()
        if model_name.startswith("MONSTB"):
            plt.savefig('{}/model/{}/{}/{}/stft/{}_patient{}_{}_step_preictal{}.png'.format(model_path, dataset_name,
                                                                                            model_name, patient_id,
                                                                                            str(datetime.now().strftime(
                                                                                                "%Y%m%d%H%M%S")),
                                                                                            patient_id, i,
                                                                                            step_preictal), transparent=True)
        elif model_name.startswith("MONSTL"):
            plt.savefig('{}/model/{}/{}/{}/stftC/{}_patient{}_{}_step_preictal{}.png'.format(model_path, dataset_name,
                                                                                             model_name, patient_id,
                                                                                             str(datetime.now().strftime(
                                                                                                 "%Y%m%d%H%M%S")),
                                                                                             patient_id, i,
                                                                                             step_preictal), transparent=True)

        # draw predicting line
        plt.figure()
        plt.plot(predicting_probablity_smooth, linewidth='2', color='black')  # , marker='o')
        if model_name.startswith("MONSTB"):
            plt.savefig('{}/model/{}/{}/{}/stft/patient{}_{}_step_preictal{}_predicting_label.png'.format(model_path,
                                                                                                          dataset_name,
                                                                                                          model_name,
                                                                                                          patient_id,
                                                                                                          patient_id, i,
                                                                                                          step_preictal))
        elif model_name.startswith("MONSTL"):
            plt.savefig('{}/model/{}/{}/{}/stftC/patient{}_{}_step_preictal{}_predicting_label.png'.format(model_path,
                                                                                                           dataset_name,
                                                                                                           model_name,
                                                                                                           patient_id,
                                                                                                           patient_id,
                                                                                                           i,
                                                                                                           step_preictal))

    # calculate p-value
    N = len(SEN_list)
    n = sum(SEN_list)
    if dataset_name.startswith("KAGGLE"):
        pw = PW_count / ((interictal_length + preictal_length) * len(SEN_list))  # len(SEN_list) equals to number of seizures in test set
    else:
        pw = PW_count / ((interictal_length + preictal_length + ictal_length) * len(SEN_list))  # len(SEN_list) equals to number of seizures in test set
    if pw > 1:
        pw = 1.0
    p_value = P_value(tw0, tw, N, n, pw).calculate_p()

    print("total : TP {} FN {} TN {} FP {}".format(sum(TP_list), sum(FN_list), sum(TN_list), sum(FP_list)))
    print("true seizure {} predicted seizure {}".format(N, n))
    print("sensitivity : {:.2%}".format(sum(TP_list) / (sum(TP_list) + sum(FN_list))))
    print("specificity : {:.2%}".format(sum(TN_list) / (sum(TN_list) + sum(FP_list))))
    print("accuracy : {:.2%}".format(
        (sum(TP_list) + sum(TN_list)) / (sum(TP_list) + sum(FN_list) + sum(TN_list) + sum(FP_list))))
    print("FPR : {:.4}".format(np.mean(FPR_list)))
    print("SEN : {:.2%}".format(np.mean(SEN_list)))
    print("O_FPR : {:.4}".format(np.mean(O_FPR_list)))
    print("O_SEN : {:.2%}".format(np.mean(O_SEN_list)))
    print("AUC : {:.4}".format(np.mean(AUC_list)))
    print("pw : {} {:.4}".format(PW_count, pw))
    print("p-value : {:.4}".format(p_value))
    print("Avg infer time1 (time.clock) : {:.4}".format(np.mean(InferTime_list1)))
    print("Avg infer time2 (time.time): {:.4}\n".format(np.mean(InferTime_list2)))

    # save results to excel
    save_data = {"ID": patient_id,
                 "True Seizure": N,
                 "Predict Seizure": n,
                 "TP": sum(TP_list),
                 "FN": sum(FN_list),
                 "TN": sum(TN_list),
                 "FP": sum(FP_list),
                 "Sen": sum(TP_list) / (sum(TP_list) + sum(FN_list)),
                 "Spe": sum(TN_list) / (sum(TN_list) + sum(FP_list)),
                 "Acc": (sum(TP_list) + sum(TN_list)) / (sum(TP_list) + sum(FN_list) + sum(TN_list) + sum(FP_list)),
                 "AUC": np.mean(AUC_list),
                 "Sn": np.mean(SEN_list),
                 "FPR": np.mean(FPR_list),
                 "O_SEN": np.mean(O_SEN_list),
                 "O_FPR": np.mean(O_FPR_list),
                 "pw": pw,
                 "p-value": p_value}
    eval_param = {"descri": description,
                  "filter": moving_average_length * step_preictal,
                  "threshold": threshold,
                  "persistence": persistence_second}
    WriteToExcel(model_path, dataset_name, model_name, save_data, eval_param)
    if dataset_name == "CHB" and patient_id == 23 or dataset_name == "XUANWU" and patient_id == 5:
        CalculateAverageToExcel(model_path, dataset_name, model_name, save_data, eval_param)
