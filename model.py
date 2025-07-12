import torchvision
from torchvision import models
from torchvision import transforms
from torchinfo import summary
import torch.nn as nn
from data import FaceDataset

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transform = weights.transforms()

mainDataset = FaceDataset("data/train", transform = transform)

def buildModel(numClasses : int, nutshell : bool):
    model = torchvision.models.efficientnet_b0(weights = weights)

    for param in model.features.parameters():
        param.requires_grad = False

    if nutshell == True:
        summary(
            model = model,
            input_size = (32, 3, 224, 224),
            col_names = ["input_size", "output_size", "num_params", "trainable"],
            col_width = 20,
            row_settings = ["var_names"]
        )

    model.classifier = nn.Sequential(
        nn.Dropout(p = 0.2, inplace = True),
        nn.Linear(in_features = 1280, out_features = numClasses, bias = True)
    )

    return model


model0_0 = buildModel(len(mainDataset.classes), nutshell = True)
