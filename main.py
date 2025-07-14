# main.py

from data import FaceDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision
from model import buildModel
import torch
import torch.nn as nn
from tqdm import tqdm


weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

trainDataset = FaceDataset("data/train", transform = transform)
trainDataloader = DataLoader(trainDataset, shuffle = True, batch_size = 32)

model0 = buildModel(len(trainDataset.classes), nutshell = False)
lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params = model0.parameters(), lr = 0.001)

epochs = 10
losses = []


for epoch in range(epochs):
    model0.train()
    trainLossBatch = 0.0

    loop = tqdm(trainDataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    
    for image, label in loop:
        yTrainPred = model0(image)
        trainLoss = lossFunction(yTrainPred, label)

        optimizer.zero_grad()
        trainLoss.backward()
        optimizer.step()

        trainLossBatch += trainLoss.item()

        loop.set_postfix(batch_loss=trainLoss.item())

    averageLoss = trainLossBatch / len(trainDataloader)
    losses.append(f"Epoch {epoch + 1}/{epochs} - Loss: {averageLoss:.4f}")
    print(losses[-1])

torch.save(model0.state_dict(), "weights/modelWeights.pth")
print("Model has been saved ✅")


    