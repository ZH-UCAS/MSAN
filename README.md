# 1. 数据准备
请从以下地址下载 CHB-MIT 数据集：
http://archive.physionet.org/physiobank/database/chbmit/

# 2. 数据预处理

## - 如果仅需处理单个患者的数据，执行：
make preprocess

## - 若需处理所有患者的数据，执行：
make preprocess_chb

# 3. 模型训练

## - 针对某一位患者进行训练：
make train

## - 若训练全部患者数据，执行：
make train_chb

# 4. 模型评估

## - 评估单个患者模型：
make eval

## - 评估所有患者模型：
make eval_chb

# 如有任何遗漏的依赖项或其他配置问题，请及时联系我。
