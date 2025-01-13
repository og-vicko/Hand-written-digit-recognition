import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # 1 input channel (grayscale image), 32 output channels (filters)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # Second convolution layer

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # MaxPooling with 2x2 window

        # Fully Connected (FC) Layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Flattened layer (assuming input images are 28x28, after pooling will be 7x7)
        self.fc2 = nn.Linear(128, 10)  # 10 output classes for digit recognition (0-9)

    def forward(self, x):
        # Applying the first convolutional layer and activation
        x = self.pool(F.relu(self.conv1(x)))

        # Applying the second convolutional layer and activation
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the image for fully connected layer
        x = x.view(-1, 64 * 7 * 7)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

