# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:39:42 2023

@author: rohan
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torchvision import transforms

import pandas as pd
import numpy as np
import pydicom

from sklearn.model_selection import train_test_split

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories

def get_train_valid(train_dataset):
    
    patients = train_dataset['patient_id'].drop_duplicates()
    train, valid = train_test_split(patients, train_size= 0.8, test_size = 0.2)
    train = train_dataset[train_dataset['patient_id'].isin(train)]
    valid = train_dataset[train_dataset['patient_id'].isin(valid)]
    
    return train, valid

class MammoDataset(Dataset):
    
    def __init__(self, df, transform = None):
        
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        image_path = self.df.iloc[idx, 2]
        
        # Load the DICOM image
        dicom_image = pydicom.dcmread(image_path)
        # Convert DICOM image to NumPy array
        image = dicom_image.pixel_array.astype(np.uint8)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        if self.transform is not None:
            image = self.transform(image)
            
        label = self.df.iloc[idx, 3]
        label = torch.tensor(label)
        
        return image, label
    
def get_dataloaders(train, valid, batch_size = 5):
    
    data_transforms = {
        'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                ]),
        'valid': transforms.Compose([
                 transforms.ToPILImage(),
                 transforms.Resize((224, 224)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
             ]),
        }
    
    train_dataset = MammoDataset(train, transform = data_transforms['train'])
    valid_dataset = MammoDataset(valid, transform = data_transforms['valid'])

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True)
    dataloaders = {'train' : train_dataloader, 'valid' : valid_dataloader}
    
    return dataloaders