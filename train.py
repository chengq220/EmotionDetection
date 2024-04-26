import torch
import torch.nn as nn
from models.model import ED_Model1 
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

batch_size = 256
alpha = 0.01
classes = 1
epoch = 100
save_freq = 25
device = "cpu"

trainData = Dataset("datasets/maskedData")
trainDataloader = DataLoader(trainData, batch_size=batch_size, collate_fn=custom_collate, shuffle=True, drop_last = True)

testData = Dataset("datasets/maskedData", train=False)
testLoader = DataLoader(testData)

model = CNN(output_dim=classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
criterion = nn.BCEWithLogitsLoss()


# Training loop
for epoch in range(epoch):
    train_loss = 0
    total_samples = 0
    for i, (feature, label) in tqdm(enumerate(trainDataloader), total=len(trainDataloader)):
        prediction = model(feature.to(device)).squeeze()
        loss = criterion(prediction, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        train_loss += loss.item() * feature.size(0)
        total_samples += feature.size(0)
    train_loss /= total_samples
    print(train_loss)
    if(epoch % save_freq == 0):
        model.save_network(save_dir = "saves", name = 'train')  
model.save_network(save_dir = "saves", name = 'train') 
