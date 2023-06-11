#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import necessary packages
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator


# In[3]:


# print the version of the medmnist package
print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
# set the data flag and download flag
data_flag = 'chestmnist'
download = True


# In[4]:


# set hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 250
lr = 0.001


# In[5]:


# get the information of the current dataset
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])


# In[6]:


# get the corresponding dataset class based on the data_flag variable
DataClass = getattr(medmnist, info['python_class'])

# preprocessing steps to be applied on the dataset
data_transform = transforms.Compose([
    transforms.ToTensor(), # convert data to tensor
    transforms.Normalize(mean=[.5], std=[.5]) # normalize data
])

# load train, test datasets using DataClass, and apply preprocessing steps
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

# load dataset for PIL images
pil_dataset = DataClass(split='train', download=download)

# create data loaders for the datasets
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True) # training data loader
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False) # data loader for evaluation on training set
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False) # test data loader


# In[7]:


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


# In[8]:


# Instantiate the CNN model with the specified number of input channels and output classes
model = CNN(in_channels=n_channels, num_classes=n_classes)

# Define the loss function based on the task type
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

# Define the optimizer with stochastic gradient descent and a specified learning rate and momentum
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


# In[9]:


# train loop
for epoch in range(NUM_EPOCHS):
    # initialize counters for training and testing accuracy
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0
    
    # set the model in training mode
    model.train()
    
    # loop through training data
    for inputs, targets in tqdm(train_loader):
        # reset the gradients
        optimizer.zero_grad()
        # forward pass through the model
        outputs = model(inputs)
        
        # calculate the loss based on the task type
        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
        
        # calculate gradients
        loss.backward()
        # update weights
        optimizer.step()

# test function
def test(split):
    # set the model in evaluation mode
    model.eval()
    # initialize empty tensors for true and predicted values
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    # select the data loader based on the split type
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    # loop through the data loader
    with torch.no_grad():
        for inputs, targets in data_loader:
            # forward pass through the model
            outputs = model(inputs)

            # adjust targets and outputs based on the task type
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            # concatenate true and predicted values
            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        # convert tensors to numpy arrays
        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        # instantiate an Evaluator object and calculate metrics
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)
    
        # print the results
        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


# In[10]:


# Evaluation of the model
print('==> Evaluating ...')
# evaluate on training data
test('train')
# evaluate on testing data
test('test')


# In[31]:


state_dict = torch.load('model1.pth')
model.load_state_dict(state_dict)


# In[32]:


# This loop iterates over all the elements in the tensor and returns the i-th element after flattening the tensor using the view() method.

# Here is a brief explanation of what each part of the loop does:

# tensor.numel() returns the total number of elements in the tensor.
# tensor.view(-1) flattens the tensor into a 1D tensor. The -1 argument tells PyTorch to infer the size of the flattened dimension based on the other dimensions of the tensor.
# tensor.view(-1)[i] returns the i-th element of the flattened tensor.
# So, essentially, this loop iterates over all the elements of the tensor in a flattened manner, allowing you to perform some operation on each element individually.


# In[59]:


model.state_dict()


# In[33]:




# Iterate over all keys in the state dictionary
for key in state_dict:
    tensor = state_dict[key]
    # Iterate over all elements of the tensor
    for i in range(tensor.numel()):
        tensor.view(-1)[i] = int(tensor.view(-1)[i] * 100000)
        print(tensor.view(-1)[i])
        


# In[ ]:


state_dict


# In[60]:


# Save the state dictionary of the trained model
torch.save(model.state_dict(), 'model3.pth')


# In[ ]:


# Load the state dictionary into a new model object
model = CNN(in_channels=n_channels, num_classes=n_classes)
model.load_state_dict(torch.load('model_state_dict.pth'))


# In[ ]:


#Data model for storing state_dict
from app import db

class ModelState(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    state_dict = db.Column(db.LargeBinary)

    def __repr__(self):
        return '<ModelState %r>' % self.name


# In[ ]:


#Save it database
from app import db
from models import ModelState
import torch

# Load the state_dict from a file
state_dict = torch.load('model_state_dict.pth')

# Create a new model state object and save it to the database
model_state = ModelState(name='my_model_state', state_dict=state_dict)
db.session.add(model_state)
db.session.commit()


# In[ ]:




