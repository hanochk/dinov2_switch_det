import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Map string labels to integer classes
LABEL_MAP = {
    "pin_space_LEFT": 0,
    "pin_space_RIGHT": 1
}

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, crop_size=224):
        self.samples = []
        self.transform = transform
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(subdir, file)
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            first_field = f.readline().split(",")[0].strip()
                            if first_field in LABEL_MAP:
                                self.samples.append((img_path, LABEL_MAP[first_field])) #TODO HK add crop coordinations into labels

    # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        # if model == "dinov2":
        #     normalize = T.Normalize([0.5], [0.5])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

"""
# Create datasets
train_dataset = CustomImageDataset("dataset/train", transform=transform)
val_dataset = CustomImageDataset("dataset/validation", transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ✅ Sample usage
# for images, labels in train_loader:
#     print(images.shape, labels)
#     break

"""
