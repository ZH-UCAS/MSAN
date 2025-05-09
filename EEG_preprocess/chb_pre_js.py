

from __future__ import print_function

import copy
from datetime import datetime

import torch.utils.data
import argparse
import numpy as np
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torch import nn

from EEG_dataset.dataset_pre import preDataset
from EEG_utils.eeg_utils import GetPatientList, GetSeizureList, GetInputChannel, GetBatchsize, GetModel, GetLoss, mkdir
import torch
import random
import os

# 设置CUDA_VISIBLE_DEVICES环境变量
import sys

sys.path.append('.')
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_seed(seed):
    '''
    set up seed for cuda and numpy
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


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


def main(train, test, LOO, patient_name, args):
    '''
    training process
    train : seizures for training. example : [1,2,3,4,5], which means seizre 1-5 are used for training
    test : seizure for testing. example : [0], which means seizre 0 is used for training
    patient_name: patient name
    args : args
    '''
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
    lock_model = args.lock_model
    domain_adaptation = args.domain_adaptation
    augmentation = args.augmentation
    step_preictal = args.step_preictal
    balance = args.balance
    using_ictal = args.using_ictal
    position_embedding = args.position_embedding

    # init_distributed_mode(args)

    # cuda and random seed
    cuda = cuda and torch.cuda.is_available()
    torch.cuda.set_device(device_number)
    print("set cuda device : ", device_number)
    setup_seed(seed)

    # dataset loader. training set and test set
    input_channel = GetInputChannel(dataset_name, patient_id, ch_num)
    train_set = preDataset(dataset_name, train, ite=1, augmentation=augmentation, using_ictal=using_ictal,
                             scaler=balance,
                             patient_id=patient_id, patient_name=patient_name, ch_num=input_channel,
                             target_preictal_interval=target_preictal_interval, step_preictal=step_preictal)

    print(train_set)

    return


if __name__ == "__main__":
    # Parse the JSON arguments
    parser = argparse.ArgumentParser(description='Seizure predicting on Xuanwu/CHB Dataset')
    parser.add_argument('--patient_id', type=int, default=1, metavar='patient id')
    parser.add_argument('--device_number', type=int, default=6, metavar='CUDA device number')
    parser.add_argument('--ch_num', type=int, default=15, metavar='number of channel')
    parser.add_argument('--model_name', type=str, default="TA_STS_ConvNet", metavar='used model')
    parser.add_argument('--dataset_name', type=str, default="CHB", metavar='dataset name : XUANWU / CHB')
    parser.add_argument('--target_preictal_interval', type=int, default=15,
                        metavar='how long we decide as preictal. Default set to 15 min')
    parser.add_argument('--step_preictal', type=int, default=30, metavar='step of sliding window (second)')  # 窗口滑动步长
    parser.add_argument('--loss', type=str, default="CE", metavar='CE:cross entropy   FL:focal loss')
    parser.add_argument('--seed', type=int, default=20221110, metavar='random seed')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        metavar="whether using transfer learning, loading pre-trained model, train on n-1, test on 1, LOOCV ")
    parser.add_argument('--lock_model', type=bool, default=False, metavar="whether locking the shallow layers")
    parser.add_argument('--domain_adaptation', type=bool, default=False,
                        metavar="whether train using domain adaption (Loss=CE+LocalMMD+GlobalMMD)")
    parser.add_argument('--position_embedding', type=bool, default=False,
                        metavar="whether train use position embedding")
    parser.add_argument('--checkpoint_dir', type=str, default='/data1/zhanghan/code/TA206/', metavar='model save path')
    parser.add_argument('--log-interval', type=int, default=4, metavar='N')
    parser.add_argument('--to_train', type=bool, default=True)
    parser.add_argument('--TestWhenTraining', type=int, default=0,
                        metavar="whether to test when training on each epoch")
    parser.add_argument('--cuda', type=bool, default=True, metavar="whether to use cuda")
    parser.add_argument("--augmentation", type=int, default=0,
                        metavar='whether to use data augmentation , default=1 use data augmentation')
    parser.add_argument("--using_ictal", type=int, default=1,
                        metavar='whether to use ictal data , default=1 use ictal data')
    parser.add_argument('--balance', type=int, default=1, metavar='whether to balance preictal and interictal data')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batchsize')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='learning rate')
    parser.add_argument('--test_every', type=int, default=5, metavar='N')
    parser.add_argument('--learning_rate_decay', type=float, default=0.8, metavar='N')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='N')
    parser.add_argument('--num_epochs', type=int, default=320, metavar='number of epochs')
    # parser.add_argument('--early_stop_patience', type=int, default=16, metavar='N')
    parser.add_argument('--early_stop_patience', type=int, default=15, metavar='N')
    parser.add_argument('--first', type=bool, default=False, metavar='shifou diyici yunxing')
    args = parser.parse_args()

    # get patient list, patient id, seizure list, etc
    patient_list = GetPatientList(args.dataset_name)
    patient_id = args.patient_id
    patient_name = patient_list[str(patient_id)]
    seizure_list = GetSeizureList(args.dataset_name)
    seizure_count = len(seizure_list[str(patient_id)])
    if args.dataset_name.startswith("CHB"):
        args.batch_size = 300
    elif args.dataset_name.startswith("XUANWU"):
        args.batch_size = 75
    args.checkpoint_dir = os.getcwd()  # /home/al/GLH/code/seizure_predicting_seeg/no_TAL
    print("dataset : {} \npatient {} \nseizure count : {}\n".format(args.dataset_name, patient_id, seizure_count))

    # LOOCV for each patient.
    for LOO in seizure_list[str(patient_id)]:
        print(LOO)
        # if LOO == 1:
        #     continue
        # if LOO == 0:
        #     continue
        # if LOO == 2:
        #     continue
        # if LOO == 3:
        #     continue
        test = LOO
        train = list(set(seizure_list[str(patient_id)]) - set([test]))
        main(train, test, LOO, patient_name, args)
