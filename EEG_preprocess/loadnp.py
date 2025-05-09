# import numpy as np
#
# # 从.npy文件加载NumPy数组
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal0.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal1.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal2.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal3.npy"
# # file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal4.npy"
# # file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal5.npy"
# # file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/preictal6.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/interictal2.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/interictal1.npy"
# file_path = "/data1/zhanghan/data/CHBMIT60/chb01/60min_5step_18ch_STFT_True_Noise_True/interictal0.npy"
#
#
# loaded_array = np.load(file_path)
#
# # 计算最大值、最小值和均值
# max_value = np.max(loaded_array)
# min_value = np.min(loaded_array)
# mean_value = np.mean(loaded_array)
#
# # 打印结果
# print("最大值: {}".format(max_value))
# print("最小值: {}".format(min_value))
# print("均值: {}".format(mean_value))


import torch
import matplotlib.pyplot as plt
from datetime import datetime


class ImagePlotter:
    def __init__(self, b, channels, figsize=(5, 5), cmap='gray'):
        self.b = b
        self.channels = channels
        self.figsize = figsize
        self.cmap = cmap

    def plot_images(self, images, target, filename_prefix, epoch):
        plt.figure(figsize=(self.figsize[0] * self.b, self.figsize[1] * self.channels))

        for i in range(self.b):
            for channel in range(self.channels):
                img = images[i, channel].cpu().detach().numpy()
                plt.subplot(self.channels, self.b, channel * self.b + i + 1)
                plt.imshow(img, cmap=self.cmap)
                plt.axis('off')

                # 在图像下方添加文本标记
                plt.text(0.5, -0.1, f'Target: {target[i]}', ha='center', va='center', transform=plt.gca().transAxes)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{filename_prefix}_{timestamp}_{epoch}.png"
        plt.savefig(filename)
        plt.close()
        print(f"{filename_prefix} images saved")


# 示例测试用例
def test_image_plotter():
    b = 5
    channels = 3
    plotter = ImagePlotter(b=b, channels=channels)

    # 模拟数据
    images = torch.randn(b, channels, 224, 224)
    target = torch.randint(0, 2, (b,))

    # 测试采样图像
    plotter.plot_images(images, target, 'sampled_images', epoch=0)

    # 测试原始数据图像
    plotter.plot_images(images, target, 'original_images', epoch=0)


# 运行测试用例
test_image_plotter()


