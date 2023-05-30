# -*- coding: utf-8 -*-
"""
Created on Sat May 27 19:27:51 2023

@author: rohan
"""
import torch
import pandas as pd
from dataset import get_train_valid, get_dataloaders
from utils import plot_training, n_p, get_count

from vgg16 import VGG16
from train import train_model, get_metrics

# read training set and the metadata file
df_train_set = pd.read_csv(r'C:\IUPUI\PLHILab\Mammography\CBIS DDSM\training\calc_case_description_train_set.csv')
df_metadata = pd.read_csv(r'C:\IUPUI\PLHILab\Mammography\CBIS DDSM\training\manifest-ujvjeoX91618930289175451743\metadata.csv')

### Files preprocess

# pick file name and set labels
df_train_set['image_file_name'] = df_train_set['image file path'].str.split('/').apply(lambda x : x[0])
df_train_set.loc[df_train_set['pathology'].str.startswith('BENIGN'), 'pathology'] = 0
df_train_set.loc[df_train_set['pathology'] == 'MALIGNANT', 'pathology'] = 1

# train dataset does not have right file paths, take it from metadata
df_metadata = df_metadata[['Subject ID', 'File Location']]
df_metadata['File Location'] = df_metadata['File Location'].str[2:]
local_path = r'C:\IUPUI\PLHILab\Mammography\CBIS DDSM\training\manifest-ujvjeoX91618930289175451743'
df_metadata['File Location'] = local_path + '\\' + df_metadata['File Location'] + '\\' + '1-1.dcm'
df_train_set = df_train_set.merge(df_metadata.rename(columns = {'Subject ID' : 'image_file_name'}), on = ['image_file_name'], how = 'left')

# drop rows where file location is not available
df_train_set = df_train_set.dropna(subset = ['File Location'])
# subset required columns
df_train_set = df_train_set[['patient_id', 'image_file_name', 'File Location', 'pathology']]

### Dataloaders

# train validation split
train_data, valid_data = get_train_valid(df_train_set)

# get dataloaders
dataloaders = get_dataloaders(train_data, valid_data, batch_size = 10)
dataset_sizes = {'train': len(train_data), 'valid' : len(valid_data)}

# prepare & run model
data_cat = ['train', 'valid']
# tai = total abnormal images, tni = total normal images
tai = {'train': get_count(train_data, 1), 'valid': get_count(valid_data, 1)}
tni = {'train': get_count(train_data, 0), 'valid': get_count(valid_data, 0)}
Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

print('tai:', tai)
print('tni:', tni, '\n')
print('Wt0 train:', Wt0['train'])
print('Wt0 valid:', Wt0['valid'])
print('Wt1 train:', Wt1['train'])
print('Wt1 valid:', Wt1['valid'])

class Loss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0

    def forward(self, inputs, targets, phase):
        loss = torch.nn.functional.binary_cross_entropy(inputs, targets,
                                                        weight=(self.Wt1[phase] * targets + self.Wt0[phase] * (1 - targets)))
        return loss
    
model = VGG16(num_classes=1)
model = model.cuda()

criterion = Loss(Wt1, Wt0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

# #### Train model
model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=10)

torch.save(model.state_dict(), 'models/model.pth')

# Evaluation
model = VGG16(num_classes=1)
model = model.cuda()
model.load_state_dict(torch.load(r'models/model.pth'))

# valid accuracy
get_metrics(model, criterion, dataloaders, dataset_sizes)
