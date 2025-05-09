import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from thop import profile
from monster import MONSTBX,CNN_LSTM_SeizurePrediction
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
        self.fc1 = nn.Linear(64 * 1 * 1, 8)
        self.fc2 = nn.Linear(8, 2)

        # Dropout Layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2, 2))
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2, 2))
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, (24, 24))

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers with Dropout
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim=1)

        return x


 # MONSTBX,CNN_LSTM_SeizurePrediction
# Set input dimensions
batch_size = 128
num_channels = 3
height = 224
width = 224
input_tensor = torch.randn(batch_size, num_channels, height, width)

# Initialize model
model = SeizurePredictionCNN(num_channels)

# Calculate FLOPs and Parameters
flops, params = profile(model, inputs=(input_tensor,))
print(f"FLOPs: {flops}, Parameters: {params}")

# Measure Inference Time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
input_tensor = input_tensor.to(device)

# Warm up
for _ in range(10):
    model(input_tensor)

# Measure time
start_time = time.time()
with torch.no_grad():
    for _ in range(100):
        model(input_tensor)
end_time = time.time()

inference_time = (end_time - start_time) / 100
print(f"Inference time: {inference_time} seconds per batch of {batch_size}")
