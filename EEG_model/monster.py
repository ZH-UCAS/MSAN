# coding: utf-8
# swin transformer as the backbone
import argparse
import datetime
import errno
import json
import math
import os
import os.path as osp
import sys
import time
# coding: utf-8
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import auc, roc_curve
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from EEG_model.ASPP import ASPP, DeepLabHead, DeepLabHead_Tiny
from EEG_model.STANX import SwinTransformerC, SwinTransformer

import torch
import torch.nn as nn
import math
import sys
from denoising_diffusion_pytorch.classifier_free_guidance import Unet, GaussianDiffusion
import torch.nn.functional as F
sys.path.append('.')


# ###################################### self-defined model ######################################
# swin transformer as the backbone

class SwinTransformerWithLSTM(nn.Module):
    def __init__(self, swinB, num_classes=2):
        super(SwinTransformerWithLSTM, self).__init__()
        self.swinB = swinB

        # Initialize LSTM layers
        self.lstm1 = nn.LSTM(192, 32, batch_first=True)
        self.lstm2 = nn.LSTM(384, 32, batch_first=True)
        self.lstm3 = nn.LSTM(768, 32, batch_first=True)
        self.lstm4 = nn.LSTM(768, 32, batch_first=True)

        self.lstmlayer = nn.LSTM(32, 1, batch_first=True)
        self.lstm_hn = nn.LSTM(4, 1, batch_first=True)

        self.fc0 = nn.Linear(768, num_classes)
        self.num_classes = num_classes

        # ########## for the myLSTMcell ######################
        self.fc_layer = nn.Linear(2048, 1024)
        self.fc_HN = nn.Linear(2048, 1024)
        self.tanh = nn.Tanh()
        # ####################################################
        self.output_fc_layer = nn.Linear(1024, num_classes)
        self.output_fc_HN = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.swinB.upsample(x.float())
        x, H, W = self.swinB.patch_embed(x)
        x = self.swinB.pos_drop(x)
        layer_num = 0
        result = []
        x1, x2, x3, x4 = [], [], [], []
        hn1, hn2, hn3, hn4 = [], [], [], []
        cn1, cn2, cn3, cn4 = [], [], [], []

        for layer in self.swinB.layers:
            x, H, W = layer(x, H, W)

            if layer_num == 0:
                x1 = x  # B 870 192
                # x1 = x1.permute(0, 2, 1)
                # x1 = x1.view(x1.shape[0], x1.shape[1], int(math.sqrt(x1.shape[2])), int(math.sqrt(x1.shape[2])))
                # x1 = x1.flatten(2).transpose(1, 2)  # Reshape for LSTM input
                x1, (hn1, cn1) = self.lstm1(x1)  # Apply LSTM # B 870 32

            elif layer_num == 1:
                x2 = x  # B 225 384
                # x2 = x2.permute(0, 2, 1)
                # x2 = x2.view(x2.shape[0], x2.shape[1], int(math.sqrt(x2.shape[2])), int(math.sqrt(x2.shape[2])))
                # x2 = x2.flatten(2).transpose(1, 2)  # Reshape for LSTM input
                x2, (hn2, cn2) = self.lstm2(x2)  # Apply LSTM # B 225 32

            elif layer_num == 2:
                x3 = x  # B 64 768
                # x3 = x3.permute(0, 2, 1)
                # x3 = x3.view(x3.shape[0], x3.shape[1], int(math.sqrt(x3.shape[2])), int(math.sqrt(x3.shape[2])))
                # x3 = x3.flatten(2).transpose(1, 2)  # Reshape for LSTM input
                x3, (hn3, cn3) = self.lstm3(x3)  # Apply LSTM # B 64 32

            elif layer_num == 3:
                x4 = x  # B 16 1536
                # x4 = x4.permute(0, 2, 1)
                # x4 = x4.view(x4.shape[0], x4.shape[1], int(math.sqrt(x4.shape[2])), int(math.sqrt(x4.shape[2])))
                # x4 = x4.flatten(2).transpose(1, 2)  # Reshape for LSTM input
                x4, (hn4, cn4) = self.lstm4(x4)  # Apply LSTM # B 16 32

            layer_num = layer_num + 1

        x = self.swinB.norm(x)  # B L C
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        logits0 = self.fc0(x)
        logits0_ori = logits0

        ##################### lstmlayer #####################
        result = torch.cat((x1, x2, x3, x4), dim=1)  # B  870+225+64+16=1175 32
        res, (hn, cn) = self.lstmlayer(result)
        # logits1 = self.output_fc_layer(hn) # 1 B 1

        ##################### HN #####################
        result_hn = torch.cat((hn1, hn2, hn3, hn4), dim=0)  # 4 B 32
        result_hn = result_hn.permute(1, 2, 0)  # B 4 32
        result_hn, (_hn, _cn) = self.lstm_hn(result_hn)
        result_hn = torch.flatten(result_hn, 1)
        logits1 = self.output_fc_HN(result_hn)

        return logits0_ori, logits1


# ###################################### BX model ######################################
# swin transformer as the backbone

class SwinTransformerWithLSTM_BX(nn.Module):
    def __init__(self, swinB, num_classes=2):
        super(SwinTransformerWithLSTM_BX, self).__init__()
        self.swinB = swinB

        # Initialize LSTM layers
        # self.lstm1 = nn.LSTM(768, 768, num_layers=1, batch_first=True, bias=False)
        # self.lstm2 = nn.LSTM(768, 768, num_layers=1, batch_first=True, bias=False)
        # self.lstm3 = nn.LSTM(768, 768, num_layers=1, batch_first=True, bias=False)
        # self.lstm4 = nn.LSTM(768, 768, num_layers=1, batch_first=True, bias=False)

        self.lstm_A = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_B = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_C = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_D = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)


        #
        # self.lstmlayer = nn.LSTM(32, 1, batch_first=True)
        # self.lstm_hn = nn.LSTM(4, 1, batch_first=True)
        #
        self.fc0 = nn.Linear(768, num_classes)
        self.num_classes = num_classes


        self.aspp01 = DeepLabHead(192, num_classes, drlist=[1, 2, 3])
        self.aspp02 = DeepLabHead(384, num_classes, drlist=[1, 2, 3])
        self.aspp3 = DeepLabHead(768, num_classes, drlist=[1])
        self.aspp4 = DeepLabHead(768, num_classes, drlist=[1])


        self.LSTM_output_fc_Pro = nn.Linear(512, 384)
        self.LSTM_output_bn_Pro = nn.BatchNorm1d(384)
        self.LSTM_output_relu_Pro = nn.ReLU()
        self.LSTM_output_fc_E = nn.Linear(384, 384)
        # self.LSTM_output_reluB = nn.ReLU()
        self.LSTM_output_fc_X = nn.Linear(384, 2)

        ##################### 6.0plus #####################

        ##################### 5.0 #####################
        # self.aspp1=DeepLabHead(192, num_classes)
        # self.aspp2=DeepLabHead_Tiny(192, num_classes)
        ##################### 5.0 #####################

        # ########## cnn  ######################
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.relu = nn.ReLU()

        ##################### 4.0 #####################

        ##################### 6.0plus #####################
        self.conv11 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=4, stride=4, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(768)

        self.conv21 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=4, stride=2, padding=1, bias=False)  # (28, 28)->(14, 14)
        self.bn21 = nn.BatchNorm2d(768)
        self.bn31 = nn.BatchNorm2d(768)

        self.bn41 = nn.BatchNorm2d(768)
        ##################### 6.0plus #####################

        # ########## for MLP1 MLP2 ######################

        # self.MLP1 = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        # )
        # self.MLP2 = nn.Sequential(
        #     nn.Linear(768, 768),
        #     nn.ReLU(),
        #     nn.Linear(768, 768)
        # )
        self.MLP_A = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.MLP_B = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # ########## for MLP3 MLP4 ######################
        # self.MLP3 = nn.Sequential(
        #     nn.Linear(768, 384),
        #     nn.ReLU(),
        #     nn.Linear(384, 192)
        # )
        # self.MLP4 = nn.Sequential(
        #     nn.Linear(768, 384),
        #     nn.ReLU(),
        #     nn.Linear(384, 192)
        # )

        # ########## for MLP7 MLP8 ######################
        # self.MLP7 = nn.Sequential(
        #     nn.Linear(3072, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 768)
        # )
        # self.MLP8 = nn.Sequential(
        #     nn.Linear(3072, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 768)
        # )

        # ########## for the myLSTMcell ######################
        self.tanh = nn.Tanh()
        # ####################################################

        # ####################### 78 #############################
        self.weight1 = nn.Parameter(torch.tensor(0.1))
        self.weight2 = nn.Parameter(torch.tensor(0.2))
        self.weight3 = nn.Parameter(torch.tensor(0.3))
        self.weight4 = nn.Parameter(torch.tensor(0.4))
        # ####################### 78 #############################

        # ####################################################

    def forward(self, x):  # x: B 18 117 114

        x = self.swinB.upsample(x.float())  # B 18 224 224
        x = self.swinB.conv1x1(x)  # B 3 224 224
        x = self.relu(x)  # B 3 224 224


        x, H, W = self.swinB.patch_embed(x)  # B 3136 96        W:56 H:56
        x = self.swinB.pos_drop(x)  # B 3136 96
        layer_num = 0
        result = []
        x1, x2, x3, x4 = [], [], [], []
        hn1, hn2, hn3, hn4 = [], [], [], []
        cn1, cn2, cn3, cn4 = [], [], [], []

        for layer in self.swinB.layers:
            # IN X B 3136 96 # OUT X B 870 192 H:59 W:58
            x, H, W = layer(x, H, W)
            # IN X1 (B 784 192) # H:28 W:28
            # IN X2 (B 196 384) # H:14 W:14
            # IN X3 (B 49  768) # H:7  W:7
            # IN X4 (B 49  768) # H:7  W:7
            if layer_num == 0:
                x1 = x  # X1 (B 784 192) # H:28 W:28
                x1 = x1.permute(0, 2, 1)
                x1 = x1.view(x1.shape[0], x1.shape[1], int(math.sqrt(x1.shape[2])), int(math.sqrt(x1.shape[2])))

            elif layer_num == 1:
                x2 = x  # X2 (B 196 384) # H:14 W:14
                x2 = x2.permute(0, 2, 1)
                x2 = x2.view(x2.shape[0], x2.shape[1], int(math.sqrt(x2.shape[2])), int(math.sqrt(x2.shape[2])))

            elif layer_num == 2:
                x3 = x  # X3 (B 49  768) # H:7  W:7
                x3 = x3.permute(0, 2, 1)
                x3 = x3.view(x3.shape[0], x3.shape[1], int(math.sqrt(x3.shape[2])), int(math.sqrt(x3.shape[2])))

            elif layer_num == 3:
                x4 = x  # X4 (B 49  768) # H:7  W:7
                x4 = x4.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], x4.shape[1], int(math.sqrt(x4.shape[2])), int(math.sqrt(x4.shape[2])))

            layer_num = layer_num + 1

        x = self.swinB.norm(x)  # B L C # IN X B 49 768 # OUT XB 49 768 H:7 W:7
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1 # IN X B 49 768 # OUT XB 1 768 H:8 W:8
        x = torch.flatten(x, 1)
        logits0 = self.fc0(x)
        logits0_ori = logits0


        ##################### 6.0plus #####################
        F4 = x4  # IN X4 (B 49  768) # H:7  W:7
        F3 = x3  # IN X3 (B 49  768) # H:7  W:7
        F2 = x2  # IN X2 (B 196 384) # H:14 W:14
        F1 = x1  # IN X1 (B 784 192) # H:28 W:28
        # ####################### 78 #############################
        F4 = self.aspp4(F4)  # b 768 7 7
        F3 = self.aspp3(F3)  # b 768 7 7
        F2 = self.aspp02(F2)  # b 384 14 14
        F2 = self.conv21(F2)  # (b, 768, 7, 7)
        F1 = self.aspp01(F1)  # b 192 28 28
        F1 = self.conv11(F1)  # b 768 7 7
        # ####################### 78 #############################

        F4_feature_maps = F4  # (b, 768, 7, 7)
        F3_feature_maps = F3
        F2_feature_maps = F2
        F1_feature_maps = F1

        F4 = self.maxpool1(F4).view(x.shape[0], -1)  # (b, 768)
        F3 = self.maxpool1(F3).view(x.shape[0], -1)  # (b, 768)
        F2 = self.maxpool1(F2).view(x.shape[0], -1)  # (b, 768)
        F1 = self.maxpool1(F1).view(x.shape[0], -1)  # (b, 768)

        F4_splits = F4_feature_maps.view(F4_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F3_splits = F3_feature_maps.view(F3_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F2_splits = F2_feature_maps.view(F2_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F1_splits = F1_feature_maps.view(F1_feature_maps.shape[0], 768, 49)  # (b, 768, 49)

        F4_splits = F4_splits.permute(0, 2, 1)  # (b, 49, 768)
        F3_splits = F3_splits.permute(0, 2, 1)  # (b, 49, 768)
        F2_splits = F2_splits.permute(0, 2, 1)  # (b, 49, 768)
        F1_splits = F1_splits.permute(0, 2, 1)  # (b, 49, 768)
        ##################### 6.0plus #####################


        ##################### 7.0 #####################


        # ####################### 78 #############################
        FF_weighted = self.weight1 * F1 + self.weight2 * F2 + self.weight3 * F3 + self.weight4 * F4
        # FF_weighted = F4
        # ####################### 78 #############################

        Hidden = self.MLP_A(FF_weighted)  # (b, 768)
        Hidden = Hidden.unsqueeze(0)  # (1 b, 768)
        Cell = self.MLP_B(FF_weighted)  # (b, 768)
        Cell = Cell.unsqueeze(0)  # (1 b, 768)
        ##################### 7.0 #####################

        LSTM_output1, (Hidden1, Cell1) = self.lstm_A(F1_splits, (Hidden, Cell))  # 输入 (b, 49, 768) (1,b,768) (1,b,768) #输出 (b, 49, 768)(1,b,768)(1,b,768)
        LSTM_output2, (Hidden2, Cell2) = self.lstm_B(F2_splits, (Hidden1, Cell1))
        LSTM_output3, (Hidden3, Cell3) = self.lstm_C(F3_splits, (Hidden2, Cell2))
        LSTM_output4, (Hidden4, Cell4) = self.lstm_D(F4_splits, (Hidden3, Cell3))
        # print(LSTM_output4[:, -1, :])
        # print(Hidden4)
        # print(torch.all(torch.eq(Hidden4, LSTM_output4[:, -1, :])))
        Hidden4 = torch.squeeze(Hidden4, 0)

        Feature_map = self.LSTM_output_fc_Pro(Hidden4)
        Feature_map = self.LSTM_output_bn_Pro(Feature_map)
        Feature_map = self.LSTM_output_relu_Pro(Feature_map)
        embeddings = self.LSTM_output_fc_E(Feature_map)
        # Feature_map = self.LSTM_output_reluB(embeddings)
        LSTM_logits = self.LSTM_output_fc_X(Feature_map)
        # print("FIX MODEL")
        # LSTM_logits = self.LSTM_output_fc(Hidden4)
        return logits0_ori, LSTM_logits, embeddings

        ##################### 6.0plus #####################

        # return logits0_ori, LSTM_logits


class SwinTransformer_KAGGLE(nn.Module):
    def __init__(self, swinB, num_classes=2):
        super(SwinTransformer_KAGGLE, self).__init__()
        self.swinB = swinB

        self.lstm_A = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_B = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_C = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.lstm_D = nn.LSTM(768, 512, num_layers=1, batch_first=True, bias=False)
        self.fc0 = nn.Linear(768, num_classes)
        self.num_classes = num_classes
        ##################### 6.0plus #####################
        self.aspp1 = DeepLabHead(192, num_classes, drlist=[1, 3, 6, 9])
        self.aspp2 = DeepLabHead(384, num_classes, drlist=[1, 2, 4, 6])
        self.aspp3 = DeepLabHead(768, num_classes, drlist=[1, 2, 3])
        self.aspp4 = DeepLabHead(768, num_classes, drlist=[1, 2, 3])
        ##################### 消融实验 #####################
        # self.aspp2 = DeepLabHead(384, num_classes, drlist=[1, ])
        # self.aspp3 = DeepLabHead(768, num_classes, drlist=[1, ])
        # self.aspp4 = DeepLabHead(768, num_classes, drlist=[1, ])

        ##################### 消融实验 #####################

        self.LSTM_output_fc_Pro = nn.Linear(512, 384)
        self.LSTM_output_bn_Pro = nn.BatchNorm1d(384)
        self.LSTM_output_relu_Pro = nn.ReLU()
        self.LSTM_output_fc_E = nn.Linear(384, 384)
        self.LSTM_output_fc_X = nn.Linear(384, 2)
        # ########## cnn  ######################
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=4, stride=4, padding=1, bias=False)
        self.bn11 = nn.BatchNorm2d(768)
        self.conv21 = nn.Conv2d(in_channels=768, out_channels=768, kernel_size=4, stride=2, padding=1, bias=False)  # (28, 28)->(14, 14)
        self.bn21 = nn.BatchNorm2d(768)
        self.bn31 = nn.BatchNorm2d(768)
        self.bn41 = nn.BatchNorm2d(768)

        self.MLP_A = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.MLP_B = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.tanh = nn.Tanh()
        self.weight1 = nn.Parameter(torch.tensor(0.2))
        self.weight2 = nn.Parameter(torch.tensor(0.2))
        self.weight3 = nn.Parameter(torch.tensor(0.2))
        self.weight4 = nn.Parameter(torch.tensor(0.4))
        # ####################################################

    def forward(self, x):  # x: B 18 117 114

        x = self.swinB.upsample(x.float())  # B 18 224 224
        x = self.swinB.conv1x1(x)  # B 3 224 224
        x, H, W = self.swinB.patch_embed(x)  # B 3136 96        W:56 H:56
        x = self.swinB.pos_drop(x)  # B 3136 96
        layer_num = 0
        result = []
        x1, x2, x3, x4 = [], [], [], []
        hn1, hn2, hn3, hn4 = [], [], [], []
        cn1, cn2, cn3, cn4 = [], [], [], []

        for layer in self.swinB.layers:
            # IN X B 3136 96 # OUT X B 870 192 H:59 W:58
            x, H, W = layer(x, H, W)
            # IN X1 (B 784 192) # H:28 W:28
            # IN X2 (B 196 384) # H:14 W:14
            # IN X3 (B 49  768) # H:7  W:7
            # IN X4 (B 49  768) # H:7  W:7
            if layer_num == 0:
                x1 = x  # X1 (B 784 192) # H:28 W:28
                x1 = x1.permute(0, 2, 1)
                x1 = x1.view(x1.shape[0], x1.shape[1], int(math.sqrt(x1.shape[2])), int(math.sqrt(x1.shape[2])))

            elif layer_num == 1:
                x2 = x  # X2 (B 196 384) # H:14 W:14
                x2 = x2.permute(0, 2, 1)
                x2 = x2.view(x2.shape[0], x2.shape[1], int(math.sqrt(x2.shape[2])), int(math.sqrt(x2.shape[2])))

            elif layer_num == 2:
                x3 = x  # X3 (B 49  768) # H:7  W:7
                x3 = x3.permute(0, 2, 1)
                x3 = x3.view(x3.shape[0], x3.shape[1], int(math.sqrt(x3.shape[2])), int(math.sqrt(x3.shape[2])))

            elif layer_num == 3:
                x4 = x  # X4 (B 49  768) # H:7  W:7
                x4 = x4.permute(0, 2, 1)
                x4 = x4.view(x4.shape[0], x4.shape[1], int(math.sqrt(x4.shape[2])), int(math.sqrt(x4.shape[2])))

            layer_num = layer_num + 1

        x = self.swinB.norm(x)  # B L C # IN X B 49 768 # OUT XB 49 768 H:7 W:7
        x = self.swinB.avgpool(x.transpose(1, 2))  # B C 1 # IN X B 49 768 # OUT XB 1 768 H:8 W:8
        x = torch.flatten(x, 1)
        logits0 = self.fc0(x)
        logits0_ori = logits0
        ##################### 6.0plus #####################
        F4 = x4  # IN X4 (B 49  768) # H:7  W:7
        F3 = x3  # IN X3 (B 49  768) # H:7  W:7
        F2 = x2  # IN X2 (B 196 384) # H:14 W:14
        F1 = x1  # IN X1 (B 784 192) # H:28 W:28
        # ####################### 78 #############################
        F4 = self.aspp4(F4)  # b 768 7 7
        # F4 = self.relu(self.bn41(F4))  # (b, 768, 7, 7)
        F3 = self.aspp3(F3)  # b 768 7 7
        # F3 = self.relu(self.bn31(F3))  # (b, 768, 7, 7)
        F2 = self.aspp2(F2)  # b 384 14 14
        # F2 = self.relu(self.bn21(self.conv21(F2)))  # (b, 768, 7, 7)
        F2 = self.conv21(F2)  # (b, 768, 7, 7)
        F1 = self.aspp1(F1)  # b 192 28 28
        # F1 = self.relu(self.bn12(self.conv12(self.relu(self.bn11(self.conv11(F1))))))  # b 768 7 7
        F1 = self.conv11(F1)  # b 768 7 7

        F4_feature_maps = F4  # (b, 768, 7, 7)
        F3_feature_maps = F3
        F2_feature_maps = F2
        F1_feature_maps = F1

        F4 = self.maxpool1(F4).view(x.shape[0], -1)  # (b, 768)
        F3 = self.maxpool1(F3).view(x.shape[0], -1)  # (b, 768)
        F2 = self.maxpool1(F2).view(x.shape[0], -1)  # (b, 768)
        F1 = self.maxpool1(F1).view(x.shape[0], -1)  # (b, 768)

        F4_splits = F4_feature_maps.view(F4_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F3_splits = F3_feature_maps.view(F3_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F2_splits = F2_feature_maps.view(F2_feature_maps.shape[0], 768, 49)  # (b, 768, 49)
        F1_splits = F1_feature_maps.view(F1_feature_maps.shape[0], 768, 49)  # (b, 768, 49)

        F4_splits = F4_splits.permute(0, 2, 1)  # (b, 49, 768)
        F3_splits = F3_splits.permute(0, 2, 1)  # (b, 49, 768)
        F2_splits = F2_splits.permute(0, 2, 1)  # (b, 49, 768)
        F1_splits = F1_splits.permute(0, 2, 1)  # (b, 49, 768)
        # ####################### 78 #############################
        # FF_weighted = self.weight1 * F1 + self.weight2 * F2 + self.weight3 * F3 + self.weight4 * F4
        FF_weighted = self.weight1 * F1 + self.weight2 * F2 + self.weight3 * F3 + self.weight4 * F4
        # FF_weighted = F4
        # ####################### 78 #############################

        Hidden = self.MLP_A(FF_weighted)  # (b, 768)
        Hidden = Hidden.unsqueeze(0)  # (1 b, 768)
        Cell = self.MLP_B(FF_weighted)  # (b, 768)
        Cell = Cell.unsqueeze(0)  # (1 b, 768)
        ##################### 7.0 #####################
        LSTM_output1, (Hidden1, Cell1) = self.lstm_A(F1_splits, (Hidden, Cell))  # 输入 (b, 49, 768) (1,b,768) (1,b,768) #输出 (b, 49, 768)(1,b,768)(1,b,768)
        LSTM_output2, (Hidden2, Cell2) = self.lstm_B(F2_splits, (Hidden1, Cell1))
        LSTM_output3, (Hidden3, Cell3) = self.lstm_C(F3_splits, (Hidden2, Cell2))
        LSTM_output4, (Hidden4, Cell4) = self.lstm_D(F4_splits, (Hidden3, Cell3))
        Hidden4 = torch.squeeze(Hidden4, 0)

        Feature_map = self.LSTM_output_fc_Pro(Hidden4)
        Feature_map = self.LSTM_output_bn_Pro(Feature_map)
        Feature_map = self.LSTM_output_relu_Pro(Feature_map)
        embeddings = self.LSTM_output_fc_E(Feature_map)
        # Feature_map = self.LSTM_output_reluB(embeddings)
        LSTM_logits = self.LSTM_output_fc_X(Feature_map)
        # print("FIX MODEL")
        # LSTM_logits = self.LSTM_output_fc(Hidden4)
        return logits0_ori, LSTM_logits, embeddings






class SeizurePredictionCNN(nn.Module):
    def __init__(self, num_channels):
        super(SeizurePredictionCNN, self).__init__()
        # First Convolution Block
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=(5, 5), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(16)
        # Second Convolution Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        # Third Convolution Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 1 * 1, 256)  # Adjust this depending on the input size
        self.fc2 = nn.Linear(256, 2)  # Two output classes (preictal and interictal)

        # Dropout Layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        # print("f")
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (10, 10))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers with Dropout
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x


from torch_geometric.nn import ChebConv


class CNN_LSTM_SeizurePrediction(nn.Module):
    def __init__(self, num_channels=18, lstm_hidden_size=64, num_classes=2):
        super(CNN_LSTM_SeizurePrediction, self).__init__()

        # Convolutional layers (CNN part)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

        # MaxPooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # CNN forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.pool(x)

        # Reshape for LSTM input
        x = x.view(x.size(0), -1, 256)  # (batch_size, sequence_length, feature_size)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output from LSTM
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        x = self.fc2(x)

        return x




def MONSTBX(num_classes: int = 2, inchan: int = 3, channel=18, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    F = SwinTransformer(in_chans=inchan,
                        patch_size=4,
                        window_size=7,
                        embed_dim=96,
                        depths=(2, 2, 6, 2),
                        num_heads=(3, 6, 12, 24),
                        num_classes=num_classes,
                        channel=channel,
                        **kwargs)
    model = SwinTransformerWithLSTM_BX(F, num_classes=2)
    return model






def MONST_KAGGLE(num_classes: int = 2, inchan: int = 3, channel=18, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    F = SwinTransformer(in_chans=inchan,
                        patch_size=4,
                        window_size=7,
                        embed_dim=96,
                        depths=(2, 2, 6, 2),
                        num_heads=(3, 6, 12, 24),
                        num_classes=num_classes,
                        channel=channel,
                        **kwargs)
    model = SwinTransformer_KAGGLE(F, num_classes=2)
    return model


def MONST_layer(num_classes: int = 2, inchan: int = 18, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    F = SwinTransformerC(in_chans=inchan,
                         patch_size=4,
                         window_size=7,
                         embed_dim=96,
                         depths=(2, 2, 6, 2),
                         num_heads=(3, 6, 12, 24),
                         num_classes=num_classes,
                         **kwargs)
    model = SwinTransformerWithLSTM(F, num_classes=2)
    return model


def main():
    # model = MONST()
    # model = MONST_layer()
    model = MONSTBX().cuda()
    # model = SwinTransformerWithLSTM_BX()
    from torchsummary import summary
    # summary(model, (18, 224, 224))
    from torchviz import make_dot

    # summary(model, torch.zeros((1, 3, 32, 32)))
    for name, module in model.named_modules():
        print(name, module)

