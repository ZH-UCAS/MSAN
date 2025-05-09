

from __future__ import print_function

import copy
import sys
from datetime import datetime

import torch.utils.data
import argparse
import numpy as np
import torch.utils.data as Data
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

sys.path.append('')
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetBatchsize, GetModel, GetLoss, mkdir, GetDataType, GetDataPath
import torch
import random
import os


def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def load_latest_file(dir_path, file_ext):
    """
    加载指定目录下匹配特定后缀的最新文件
    :param dir_path: 目录路径
    :param file_ext: 文件后缀
    :return: 最新文件的内容，如果没有匹配的文件则返回 None
    """

    print("dir_path:", dir_path, "file_ext:", file_ext)
    # 获取目录中所有匹配后缀的文件路径
    matching_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(file_ext)]

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


def main(seizure_list, test, LOO, patient_name, args):
    patient_id = args.patient_id
    cuda = args.cuda
    device_number = args.device_number
    seed = args.seed
    ch_num = args.ch_num
    batch_size = args.batch_size
    model_name = args.model_name
    loss = args.loss
    dataset_name = args.dataset_name
    checkpoint_dir = args.checkpoint_dir
    target_preictal_interval = args.target_preictal_interval
    transfer_learning = args.transfer_learning
    first = args.first
    scaler = 1
    lock_model = args.lock_model
    domain_adaptation = args.domain_adaptation
    augmentation = args.augmentation
    step_preictal = args.step_preictal
    balance = args.balance
    using_ictal = args.using_ictal
    position_embedding = args.position_embedding

    # cuda and random seed
    cuda = cuda and torch.cuda.is_available()
    torch.cuda.set_device(device_number)
    print("set cuda device : ", device_number)
    setup_seed(seed)

    # dataset loader. training set and test set
    input_channel = GetInputChannel(dataset_name, patient_id, ch_num)
    data, label = [], []

    data_path = GetDataPath(dataset_name)  # data path
    data_type = GetDataType(dataset_name)  # EEG/IEEG
    patient_id = patient_id  # patient id
    patient_name = patient_name  # patient name
    ch_num = ch_num  # number of channels
    target_preictal_interval = target_preictal_interval  # how long we decide as preictal. Default set to 15 min
    step_preictal = step_preictal  # step of sliding window

    mkdir("./Cluster")

    # LOOCV for n times for each patient. n is the number of seizures
    for i in seizure_list:
        # data loading from npy files
        preIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_True_Cluster/preictal%d.npy" % (data_path, patient_name, target_preictal_interval, step_preictal, ch_num, i)))
        # 175 18 59 114
        interIctal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_True_Cluster/interictal%d.npy" % (data_path, patient_name, target_preictal_interval, step_preictal, ch_num, i)))
        # 342 18 59 114
        Ictal = np.load(("%s/%s/%dmin_%dstep_%dch_STFT_True_Cluster/ictal%d.npy" % (data_path, patient_name, target_preictal_interval, step_preictal, ch_num, i)))

        # shape transpose
        if (len(preIctal.shape) == 3):
            preIctal = preIctal.transpose(0, 2, 1)  # (180, 19, 1280)
            if Ictal.shape[0] > 0:
                Ictal = Ictal.transpose(0, 2, 1)  # (38, 19, 1280)
            interIctal = interIctal.transpose(0, 2, 1)

        # whether to balance interictal and preictal data
        ind = np.arange(0, len(interIctal))
        np.random.shuffle(ind)
        if scaler == 1 and len(preIctal) < len(interIctal):
            data.append(interIctal[ind[:int(scaler * len(preIctal))], :, :])
            label.append(np.zeros((int(scaler * len(preIctal)), 1)))
            label.append(np.ones((preIctal.shape[0], 1)))
            # TODO
        else:
            data.append(interIctal)
            label.append(np.zeros((interIctal.shape[0], 1)))
            label.append(np.ones((preIctal.shape[0], 1)))
        data.append(preIctal)

        # whether to use ictal data for training
        if using_ictal == 1:
            # print("using ictal data")
            if Ictal.shape[0] > 0:
                data.append(Ictal)
                label.append(np.full((Ictal.shape[0], 1), 2))
        print('seizure {} : preictal {} | Ictal {} | interIctal {}'.format(i, preIctal.shape, Ictal.shape, interIctal.shape))
        # print("data" + str(len(data)) + "label" + str(len(label)))

    # numpy to torch
    data, label = np.array(data), np.array(label)
    data, label = np.concatenate(data, 0), np.concatenate(label, 0)
    print("data" + str(data.shape) + "label" + str(label.shape))
    # if (len(preIctal.shape) == 3):
    #     data = data[:, np.newaxis, :, :].astype('float32')
    # elif (len(preIctal.shape)==4):#spectralCNN
    # data = data[:, np.newaxis, :, :, :].astype('float32')
    label = label.astype('int64')
    data_2d = data.reshape(data.shape[0], -1)


    # Apply PCA to reduce the data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_2d)

    x_data = torch.from_numpy(data_2d)  # ([2592, 1, 19, 1280])
    y_data = torch.from_numpy(label)  # ([2592, 1])
    lens = data.shape[0]
    print("Target {} Dataset : {} {}".format(str(augmentation), x_data.shape, y_data.shape))
    print("preIctal {} | interIctal {}\n".format(sum(np.array(label) == 1), sum(np.array(label) == 0)))

    # training
    print("Training...")

    sse = []
    # k_values = list(range(3, 25))  # 尝试的簇数量范围
    k_values = list(range(7, 8))  # 尝试的簇数量范围

    # for k in k_values:
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(x_data)
    #     sse.append(kmeans.inertia_)

    for n_clusters in k_values:
        # 使用MiniBatchKMeans进行聚类
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, random_state=0).fit(data_2d)
        # 获取聚类结果
        labels = kmeans.labels_
        sse.append(kmeans.inertia_)
        # 可视化聚类结果
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.scatter(range(len(labels)), labels, s=5)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Cluster Labels')
        ax.set_title("P_" + str(patient_id) + ' Clustering Results')
        # 自适应坐标轴范围
        plt.autoscale()
        # plt.show()
        dir_path = "./Cluster/"
        plt.savefig(dir_path + str(datetime.now().strftime("%Y%m%d%H%M%S")) + "_P_" + str(patient_id) + '.png')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)



        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        markers = ['*', 'o', 'x', '+', 'v', 's', 'p', 'h', 'd']
        for cluster_label in np.unique(labels):
            for original_label in np.unique(label[labels == cluster_label]):
                color_index = cluster_label % len(colors)
                marker_index = original_label % len(markers)
                current_marker = markers[marker_index]
                label = np.squeeze(label)
                mask = (labels == cluster_label) & (label == original_label)
                data_pca_cluster_label = data_pca[mask]
                ax.scatter(data_pca_cluster_label[:, 0], data_pca_cluster_label[:, 1],
                           s=7, label=f"Cluster {cluster_label}, Original Label {original_label}",
                           alpha=0.7, color=colors[color_index], marker=current_marker)
                print("color_index", color_index, "marker_index", marker_index, "current_marker", current_marker)
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title("P_" + str(patient_id) + ' Clustering Results')
        ax.legend()
        # plt.show()
        plt.savefig(dir_path + str(datetime.now().strftime("%Y%m%d%H%M%S")) + "_PCA_P_" + str(patient_id) + '.png')
        plt.close()

    # plt.plot(k_values, sse, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('SSE')
    # plt.title('Elbow Method')
    # plt.show()

    # mkdir and save weights
    if model_name.startswith("STANX"):
        mkdir("{}/model/{}/{}/{}/stft/".format(checkpoint_dir, dataset_name, model_name, patient_id))
    elif model_name.startswith("STANC"):
        mkdir("{}/model/{}/{}/{}/stftC/".format(checkpoint_dir, dataset_name, model_name, patient_id))

    return


if __name__ == "__main__":
    # Parse the JSON arguments
    parser = argparse.ArgumentParser(description='Seizure predicting on Xuanwu/CHB Dataset')
    parser.add_argument('--patient_id', type=int, default=1, metavar='patient id')
    parser.add_argument('--device_number', type=int, default=6, metavar='CUDA device number')
    parser.add_argument('--ch_num', type=int, default=15, metavar='number of channel')
    parser.add_argument('--dataset_name', type=str, default="CHB", metavar='dataset name : XUANWU / CHB')
    parser.add_argument('--target_preictal_interval', type=int, default=15, metavar='how long we decide as preictal. Default set to 15 min')
    parser.add_argument('--step_preictal', type=int, default=30, metavar='step of sliding window (second)')  # 窗口滑动
    parser.add_argument('--loss', type=str, default="CE", metavar='CE:cross entropy   FL:focal loss')
    parser.add_argument('--seed', type=int, default=200, metavar='random seed')
    parser.add_argument('--lock_model', type=bool, default=False, metavar="whether locking the shallow layers")
    parser.add_argument('--domain_adaptation', type=bool, default=False, metavar="whether train using domain adaption (Loss=CE+LocalMMD+GlobalMMD)")
    parser.add_argument('--position_embedding', type=bool, default=False, metavar="whether train use position embedding")
    parser.add_argument('--log-interval', type=int, default=4, metavar='N')
    parser.add_argument('--to_train', type=bool, default=True)
    parser.add_argument('--TestWhenTraining', type=int, default=0, metavar="whether to test when training on each epoch")
    parser.add_argument('--cuda', type=bool, default=True, metavar="whether to use cuda")
    parser.add_argument("--augmentation", type=int, default=0, metavar='whether to use data augmentation , default=1 use data augmentation')
    parser.add_argument("--using_ictal", type=int, default=1, metavar='whether to use ictal data , default=1 use ictal data')
    parser.add_argument('--balance', type=int, default=1, metavar='whether to balance preictal and interictal data')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batchsize')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='learning rate')
    parser.add_argument('--test_every', type=int, default=5, metavar='N')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, metavar='N')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='N')
    parser.add_argument('--num_epochs', type=int, default=350, metavar='number of epochs')
    parser.add_argument('--early_stop_patience', type=int, default=15, metavar='N')
    parser.add_argument('--first', type=bool, default=True, metavar='shifou diyici yunxing')
    args = parser.parse_args()

    # get patient list, patient id, seizure list, etc
    patient_list = GetPatientList(args.dataset_name)
    patient_id = args.patient_id
    patient_name = patient_list[str(patient_id)]
    seizure_list = GetSeizureList(args.dataset_name)
    seizure_count = len(seizure_list[str(patient_id)])
    args.batch_size = 48
    args.checkpoint_dir = os.getcwd()  #
    print("dataset : {} \npatient {} \nseizure count : {}\n".format(args.dataset_name, patient_id, seizure_count))

    # LOOCV for each patient.
    test = None
    seizure_list = list(set(seizure_list[str(patient_id)]))
    main(seizure_list, test, None, patient_name, args)
