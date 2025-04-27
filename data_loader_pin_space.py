import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
# Map string labels to integer classes
LABEL_MAP = {
    "pin_space_LEFT": 0,
    "pin_space_RIGHT": 1,
    "far_pin_space_LEFT" : 0,
    "far_pin_space_RIGHT" : 1
}
from torch.utils.data import Dataset, DataLoader
import albumentations as A

collect_metadata = True
import pandas as pd
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, crop_size=224):
        labels_acm = list()
        self.samples = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(subdir, file)
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    if os.path.exists(label_path):
                        with open(label_path, "r") as f:
                            first_field = f.readline().split(",")[0].strip()
                            if collect_metadata:
                                test_qual = [x for x in label_path.split("/") if 'Test' in x][0]
                                labels_acm.append({'file':img_path, 'label': first_field, 'test_qualifier':test_qual})
                            if first_field in LABEL_MAP:
                                self.samples.append((img_path, LABEL_MAP[first_field])) #TODO HK add crop coordinations into labels

        if collect_metadata:
            self.df = pd.DataFrame(labels_acm)
    # Image transformations
        if 0:
            self.transform = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
            ])
        # if model == "dinov2":
        # mean = [0.485, 0.456, 0.406] # imageNet
        mean = [0.39933288, 0.34880158, 0.28759238] # railvision
        # std = [0.229, 0.224, 0.225]
        std = [0.19252066, 0.18807131, 0.20751753]

        normalize = transforms.Normalize(mean=mean, std=std)
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((crop_size, crop_size)),
                # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
                transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1),  # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/augmentations.py#L64
                # transforms.CenterCrop(crop_size),
                normalize,
            ])
        # HK TODO
        # A.PadIfNeeded(min_height=448, min_width=448),

        # train_transform = Compose([
        #     RandomResizedCrop(size=(crop_size, crop_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
        #     RandomHorizontalFlip(p=0.5),
        #     ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])

        self.transform_unnorm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size)
        ])

        # if model == "dinov2":
        #     normalize = T.Normalize([0.5], [0.5])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        unnorm_img = self.transform_unnorm(image)
        if self.transform:
            image = self.transform(image)
        return image, label, unnorm_img

def create_dataloader(train_set_path, crop_size=224, num_workers=1, batch_size=32, shuffle=True):
    dataset = CustomImageDataset(root_dir=train_set_path, crop_size=crop_size)
    return  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


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
""" 
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

From HF transformr:
Fine_tune_DINOv2_for_image_classification_[minimal].ipynb

train_transform = Compose([
    RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

m_acm = list()
for ix , (imgs, labels, unnorm_img) in enumerate(tqdm(train_dataloader)):
    m_acm.append(unnorm_img.mean(dim=[0, 2, 3]).detach().cpu().numpy())
print(np.stack(m_acm).mean(0))

std_acm = list()
for ix , (imgs, labels, unnorm_img) in enumerate(tqdm(train_dataloader)):
    std_acm.append(unnorm_img.std(dim=[0, 2, 3]).detach().cpu().numpy())
np.stack(std_acm).mean(0)
"""
