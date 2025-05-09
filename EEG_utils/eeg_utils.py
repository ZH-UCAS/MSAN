

from EEG_model.monster import MONST_layer, MONSTBX, MONST_KAGGLE, SeizurePredictionCNN, CNN_LSTM_SeizurePrediction


def GetPatientList(dataset):
    """
    Get Patient Name List
    """
    patient_list = []
    if dataset == "XUANWU":
        patient_list = {'1': 'chb01',
                        '2': 'chb02',
                        '3': 'chb03',
                        '4': 'chb04'}
    elif dataset == "XW90":
        patient_list = {'1': 'chb01',
                        '2': 'chb02',
                        '3': 'chb03',
                        '4': 'chb04'}
    elif dataset == "CHB":
        patient_list = {'1': 'chb01',
                        '2': 'chb02',
                        '3': 'chb03',
                        '5': 'chb05',
                        '6': 'chb06',
                        '7': 'chb07',
                        '8': 'chb08',
                        '9': 'chb09',
                        '10': 'chb10',
                        '11': 'chb11',
                        '13': 'chb13',
                        '14': 'chb14',
                        '16': 'chb16',
                        '17': 'chb17',
                        '18': 'chb18',
                        '20': 'chb20',
                        '21': 'chb21',
                        '22': 'chb22',
                        '23': 'chb23'}
    elif dataset == "CHB30":
        patient_list = {'1': 'chb01',
                        '2': 'chb02',
                        '3': 'chb03',
                        '5': 'chb05',
                        '6': 'chb06',
                        '7': 'chb07',
                        '8': 'chb08',
                        '9': 'chb09',
                        '10': 'chb10',
                        '11': 'chb11',
                        '13': 'chb13',
                        '14': 'chb14',
                        '16': 'chb16',
                        '17': 'chb17',
                        '18': 'chb18',
                        '20': 'chb20',
                        '21': 'chb21',
                        '22': 'chb22',
                        '23': 'chb23'}
    elif dataset == "CHB60":
        patient_list = {'1': 'chb01',
                        '2': 'chb02',
                        '3': 'chb03',
                        '5': 'chb05',
                        '6': 'chb06',
                        '7': 'chb07',
                        '8': 'chb08',
                        '9': 'chb09',
                        '10': 'chb10',
                        '11': 'chb11',
                        '13': 'chb13',
                        '14': 'chb14',
                        '16': 'chb16',
                        '17': 'chb17',
                        '18': 'chb18',
                        '20': 'chb20',
                        '21': 'chb21',
                        '22': 'chb22',
                        '23': 'chb23'}
    elif dataset.startswith("KAGGLE"):
        # elif dataset == "KAGGLE":
        patient_list = {'1': 'Dog_1',
                        '2': 'Dog_2',
                        '3': 'Dog_3',
                        '4': 'Dog_4',
                        '5': 'Dog_5',
                        '6': 'Patient_1',
                        '7': 'Patient_2'}
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return patient_list


def GetSeizureList(dataset):
    """
    Get Seizure List
    """
    seizure_list = []
    if dataset == "XUANWU":
        seizure_list = {'1': [0, 1, 2, 3, 4, 5, 6, 7],
                        '2': [0, 1, 2],
                        '3': [0, 1, 2, 3, 4]}
    elif dataset == "XW90":
        seizure_list = {'1': [0, 1, 2, 3, 4, 5, 6, 7],
                        '2': [0, 1, 2],
                        '3': [0, 1, 2, 3, 4],
                        '4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                        }
    elif dataset == "CHB":
        seizure_list = {'1': [0, 1, 2, 3, 4, 5, 6],
                        '2': [0, 1, 2],
                        '3': [0, 1, 2, 3, 4, 5],
                        '5': [0, 1, 2, 3, 4],
                        '6': [0, 1, 2, 3, 4, 5, 6],
                        '7': [0, 1, 2],
                        '8': [0, 1, 2, 3, 4],
                        '9': [0, 1, 2, 3],
                        '10': [0, 1, 2, 3, 4, 5],
                        '11': [0, 1, 2],
                        '13': [0, 1, 2, 3, 4],
                        '14': [0, 1, 2, 3, 4, 5],
                        '16': [0, 1, 2, 3, 4, 5, 6, 7],
                        '17': [0, 1, 2],
                        '18': [0, 1, 2, 3, 4, 5],
                        '20': [0, 1, 2, 3, 4, 5, 6, 7],
                        '21': [0, 1, 2, 3],
                        '22': [0, 1, 2],
                        '23': [0, 1, 2, 3, 4, 5, 6], }
    elif dataset == "CHB30":
        seizure_list = {'1': [0, 1, 2, 3, 4, 5, 6],
                        '2': [0, 1, 2],
                        '3': [0, 1, 2, 3, 4, 5],
                        '5': [0, 1, 2, 3, 4],
                        '6': [0, 1, 2, 3, 4, 5, 6],
                        '7': [0, 1, 2],
                        '8': [0, 1, 2, 3, 4],
                        '9': [0, 1, 2, 3],
                        '10': [0, 1, 2, 3, 4, 5],
                        '11': [0, 1, 2],
                        '13': [0, 1, 2, 3, 4],
                        '14': [0, 1, 2, 3, 4, 5],
                        '16': [0, 1, 2, 3, 4, 5, 6, 7],
                        '17': [0, 1, 2],
                        '18': [0, 1, 2, 3, 4, 5],
                        '20': [0, 1, 2, 3, 4, 5, 6, 7],
                        '21': [0, 1, 2, 3],
                        '22': [0, 1, 2],
                        '23': [0, 1, 2, 3, 4, 5, 6], }
    elif dataset == "CHB60":
        seizure_list = {'1': [0, 1, 2, 3, 4, 5, 6],
                        '2': [0, 1, 2],
                        '3': [0, 1, 2, 3, 4, 5],
                        '5': [0, 1, 2, 3, 4],
                        '6': [0, 1, 2, 3, 4, 5, 6],
                        '7': [0, 1, 2],
                        '8': [0, 1, 2, 3, 4],
                        '9': [0, 1, 2, 3],
                        '10': [0, 1, 2, 3, 4, 5],
                        '11': [0, 1, 2],
                        '13': [0, 1, 2, 3, 4],
                        '14': [0, 1, 2, 3, 4, 5],
                        '16': [0, 1, 2, 3, 4, 5, 6, 7],
                        '17': [0, 1, 2],
                        '18': [0, 1, 2, 3, 4, 5],
                        '20': [0, 1, 2, 3, 4, 5, 6, 7],
                        '21': [0, 1, 2, 3],
                        '22': [0, 1, 2],
                        '23': [0, 1, 2, 3, 4, 5, 6], }
    elif dataset.startswith("KAGGLE"):
        # elif dataset == "KAGGLE":
        seizure_list = {'1': [0, 1, 2, 3],
                        '2': [0, 1, 2, 3, 4, 5, 6],
                        '3': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        '4': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                        '5': [0, 1, 2, 3, 4],
                        '6': [0, 1, 2],
                        '7': [0, 1, 2], }
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return seizure_list


def GetDataPath(dataset):
    """
    Get Data Path
    """
    data_path = ""
    if dataset == "XUANWU":
        data_path = "/data1/zhanghan/data/XW30"
    elif dataset == "CHB":
        data_path = "/data1/zhanghan/data/TA206"
    elif dataset == "CHB30":
        data_path = "/data1/zhanghan/data/CHBMIT30/"
    elif dataset == "CHB60":
        # data_path = "/data1/zhanghan/data/CHBMIT60/"
        data_path = "/share/home/zhanghan/data/CHBMIT60/"
        # data_path = "/share/home/zhanghan/data/CHBMIT/CHBMIT60/"

    elif dataset == "KAGGLE":
        data_path = "/data1/zhanghan/data/KAGGLE/"
        # data_path = "/share/home/zhanghan/data/KAGGLE/"
    elif dataset == "KAGGLET":
        data_path = "/share/home/zhanghan/data/KAGGLE/"
    elif dataset == "XW90":
        data_path = "/share/home/zhanghan/data/XW60/"
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return data_path


def GetDataType(dataset):
    """
    Get Data Type
    """
    data_type = ""
    if dataset == "XUANWU":
        data_type = "EEG"
    elif dataset == "CHB":
        data_type = "EEG"
    elif dataset == "CHB30":
        data_type = "EEG"
    elif dataset == "CHB60":
        data_type = "EEG"
    elif dataset == "XW90":
        data_type = "EEG"
    elif dataset.startswith("KAGGLE"):
        # elif dataset == "KAGGLE":
        data_type = "EEG"
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return data_type


def GetInputChannel(dataset, patient_id, ch_num):
    '''
    Get Model Input Channel Number for each patient
    '''
    if dataset == "XUANWU":  # ANT
        if ch_num == 0:
            # ch_num_list = {'1': 108,
            #                '2': 108,
            #                '3': 108}
            ch_num_list = {'1': 133,
                           '2': 115,
                           '3': 189}
            return ch_num_list[str(patient_id)]
    if dataset == "XW90":  # ANT
        if ch_num == 0:
            ch_num_list = {'1': 70,
                           '2': 70,
                           '3': 70,
                           '4': 70}
            # ch_num_list = {'1': 133,
            #                '2': 115,
            #                '3': 189,
            #                '4': 189}
            return ch_num_list[str(patient_id)]
    if dataset.startswith("KAGGLE"):
        ch_num_list = {'1': 16,  # 400hz
                       '2': 16,  # 400hz
                       '3': 16,  # 400hz
                       '4': 16,  # 400hz
                       '5': 15,  # 400hz
                       '6': 15,  # 5000hz
                       '7': 24}  # 5000hz
        return ch_num_list[str(patient_id)]
    elif dataset == "CHB":
        if ch_num != 18:
            print("\nplease input correct channel number for CHB name\n")
            return
        ch_num = 18
    elif dataset == "CHB30":
        if ch_num != 18:
            print("\nplease input correct channel number for CHB name\n")
            return
        ch_num = 18
    elif dataset == "CHB60":
        ch_num = 18
    else:
        print("\nplease input correct dataset name\n")
        return
    return ch_num


def GetBatchsize(dataset, patient_id):
    '''
    Get Batchsize for each patient
    '''
    batchsize = 256
    if dataset == "XUANWU":  # ANT
        batchsize = 256
    elif dataset == "CHB":
        batchsize = 128 if patient_id == 20 or patient_id == 21 else 128
    elif dataset == "CHB30":
        batchsize = 128 if patient_id == 20 or patient_id == 21 else 128
    elif dataset == "CHB60":
        batchsize = 128 if patient_id == 20 or patient_id == 21 else 128
    elif dataset.startswith("KAGGLE"):
        batchsize = 128
    elif dataset.startswith("XW90"):
        batchsize = 128
    else:
        print("\nplease input correct dataset name\n")
        exit()
    return batchsize


import scipy.io as io
from EEG_model.TA_STS_ConvNet import TA_STS_ConvNet
from EEG_model.STANX import STANX as STAN, SwinTransformer, STANC


def GetModel(input_channel, device_number, model_name, dataset_name, position_embedding, patient_id=0):
    '''
    Get Model
    '''
    if model_name.startswith("TA_STS_ConvNet"):
        SE_hidden_layer = 6
        model = TA_STS_ConvNet(96, input_channel, SE_hidden_layer, device_number, position_embedding)
    elif model_name.startswith("STANX"):
        model = STAN(num_classes=2)
    elif model_name.startswith("STANC"):
        model = STANC(num_classes=2)
    elif model_name.startswith("MONSTB"):
        if dataset_name.startswith("CHB"):
            # model = MONSTBX(num_classes=2, channel=18)
            model = SeizurePredictionCNN(num_channels=18)

        elif dataset_name.startswith("XUANWU"):
            model = MONSTBX(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
        elif dataset_name.startswith("XW90"):
            model = MONSTBX(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
        elif dataset_name.startswith("KAGGLE"):
            model = MONST_KAGGLE(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
        elif dataset_name.startswith("XW90"):
            model = MONSTBX(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))

    elif model_name.startswith("CNN2018"):
        if dataset_name.startswith("CHB"):
            model = SeizurePredictionCNN(num_channels=18)
    elif model_name.startswith("CNN2022"):
        if dataset_name.startswith("CHB"):
            model = CNN_LSTM_SeizurePrediction(num_channels=18)

    # elif model_name.startswith("LDM"):
    #     if dataset_name.startswith("CHB"):
    #         model = EEGLDM(num_classes=2, channel=18)
    #     elif dataset_name.startswith("XUANWU"):
    #         model = EEGLDM(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
    #     elif dataset_name.startswith("XW90"):
    #         model = EEGLDM(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
    #     elif dataset_name.startswith("KAGGLE"):
    #         model = EEGLDM(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
    #     elif dataset_name.startswith("XW90"):
    #         model = EEGLDM(num_classes=2, channel=GetInputChannel(dataset_name, patient_id, 0))
    elif model_name.startswith("MONSTL"):
        model = MONST_layer(num_classes=2)
    else:
        print("mode name incorrect : {}".format(model_name))
        exit()
    return model


from EEG_model.seizureNetLoss import CE_Loss, FocalLoss


def GetLoss(loss):
    if loss == "CE":
        Loss = CE_Loss()
    elif loss == "FL":
        Loss = FocalLoss()
    else:
        print("Loss {} does not exist".format(loss))
        exit()
    # print("CrossEntropy Loss weights:", Loss.crossEntropy.weight)

    return Loss


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' create successfully')
        return True
    else:
        print(path + ' path already exist')
        return False
