import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

class FaceDataset(Dataset):
    def __init__(self, rootDir : Path, transform = None):
        self.rootDir = rootDir
        self.transform = transform
        self.classes = sorted(os.listdir(rootDir))
        self.imagePaths = []
        self.labels = []

        for labelIndex, className in enumerate(self.classes):
            classFolder = os.path.join(rootDir, className)
            for fileName in os.listdir(classFolder):
                self.image_paths.append(os.path.join(classFolder, fileName))
                self.labels.append(labelIndex)
        
    def __len__(self):
            return len(self.imagePaths)
        
    def __getitem__(self, idx):
        image = Image.open(self.imagePaths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
        
        