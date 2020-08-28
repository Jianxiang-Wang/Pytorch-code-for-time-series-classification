import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

#using own dataset or not
own_dataset = False
train_data_folder = './Datasets/Train/train.csv'
test_data_folder = './Datasets/Test/test.csv'
val_data_folder = './Datasets/Val/val.csv'

# Data_transform
transform_train = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914,), (0.2023,))
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, ), (0.2023,))
    ])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, ), (0.2023,))
    ])

print('==> Preparing data..')
def DataFromCSV(csv_path):
    Data = pd.read_csv(csv_path, engine='python', header=None).values
    data = Data[:, 1:]
    label = Data[:, 0]
    return data, label

class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, features, timestep, transforms=None):
 
        self.data = pd.read_csv(csv_path, engine='python', header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.features = features
        self.timestep = timestep
        self.transforms = transforms
 
    def __getitem__(self, index):
        single_image_label = self.labels[index]
        data_as_np = np.asarray(self.data.iloc[index][1:]).reshape(self.features, self.timestep).astype(float)

        if self.transforms is not None:
            data_as_tensor = self.transforms(data_as_np)
        return (data_as_tensor, single_image_label)
 
    def __len__(self):
        return len(self.data.index)



def load():
    print('using custom dataset')
    trainset = DatasetFromCSV(train_data_folder,300,3,transform_train)
    testset = DatasetFromCSV(test_data_folder,300,3,transform_test)
    valset = DatasetFromCSV(val_data_folder,300,3,transform_val)
    
    trainloader = DataLoader(trainset, batch_size=30, num_workers=2)
    testloader = DataLoader(testset, batch_size=30, num_workers=2)
    valloader = DataLoader(valset, batch_size=30, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, valloader, classes

        






