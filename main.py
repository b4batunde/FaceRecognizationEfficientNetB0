from data import FaceDataset
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision
from model import buildModel



weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

trainDataset = FaceDataset("data/train", transform = transform)
trainDataloader = DataLoader(trainDataset, shuffle = True, batch_size = 32)

model0_0 = buildModel(len(trainDataset.classes), True)