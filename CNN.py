import torch.nn as nn

# Define the CNN class that inherits from the nn.Module class
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        # Define the first convolutional layer with 16 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        # Define the second convolutional layer with 16 filters of size 3x3
        # followed by a batch normalization layer, a ReLU activation function, and a max pooling layer with kernel size 2x2 and stride 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Define the third convolutional layer with 64 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Define the fourth convolutional layer with 64 filters of size 3x3
        # followed by a batch normalization layer and a ReLU activation function
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Define the fifth convolutional layer with 64 filters of size 3x3 and padding of 1
        # followed by a batch normalization layer, a ReLU activation function, and a max pooling layer with kernel size 2x2 and stride 2
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Define the fully connected layers with 128 units each and ReLU activation functions
        # followed by a linear layer with num_classes units
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    # Define the forward pass method for the CNN
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x