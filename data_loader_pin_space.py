import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
import cv2
import torchvision.transforms.functional as F

# Map string labels to integer classes
LABEL_MAP = {
    "pin_space_LEFT": 0,
    "pin_space_RIGHT": 1,
    "far_pin_space_LEFT" : 0,
    "far_pin_space_RIGHT" : 1
}
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class CustomCrop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return F.crop(img, self.top, self.left, self.height, self.width)


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)



collect_metadata = True
import pandas as pd

class CustomImageDataset(Dataset):

    def __init__(self, root_dir, crop_size=224, augment=False,
                 flip_hor_prob=0.3, flip_hor=False, gaussiansolar=False,
                 colorjitter=False, affine=False, crop_upper=False):

        self.augment = augment
        self.flip_hor_prob = flip_hor_prob
        self.flip_hor = flip_hor
        self.gaussiansolar = gaussiansolar
        self.affine = affine
        self.colorjitter = colorjitter
        self.crop_upper = crop_upper

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

        # if model == "dinov2":
        # mean = [0.485, 0.456, 0.406] # imageNet
        mean = [0.39933288, 0.34880158, 0.28759238] # railvision
        # std = [0.229, 0.224, 0.225]
        std = [0.19252066, 0.18807131, 0.20751753]
        normalize = transforms.Normalize(mean=mean, std=std)


        if self.augment:
            self.transform_op = [
                transforms.Resize((crop_size, crop_size)),
            ]
            self.transform_unnorm =[
                transforms.Resize((crop_size, crop_size)),
            ]

            if self.crop_upper:
                self.transform_op.append(CustomCrop(top=0, left=0, height=int(crop_size*2/3), width=crop_size))
                self.transform_op.append(transforms.Resize((crop_size, crop_size)))

                self.transform_unnorm.append(CustomCrop(top=0, left=0, height=int(crop_size*2/3), width=crop_size))
                self.transform_unnorm.append(transforms.Resize((crop_size, crop_size)))

            # HK TODO In object detection/segmentation, order becomes more critical, and libraries like Albumentations handle coordinate updates carefully.
            transform_affine = transforms.Compose(
                [transforms.RandomAffine(
                degrees=10,  # Rotate between -30 and +30 degrees
                translate=(0.1, 0.1),  # Translate by up to 10% in both x and y
                scale=(0.9, 1.1),  # Scale between 90% and 110%
                shear=10,  # Shear by up to 10 degrees
                fill=0)  # Fill color for empty pixels (black)
            ])
            # Solarize TODO
            global_transfo2_extra = transforms.Compose(
                [
                    GaussianBlur(p=0.1),
                    transforms.RandomSolarize(threshold=128/255, p=0.2),
                ]
            )

            global_transfo1_extra = GaussianBlur(p=1.0) #https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/augmentations.py#L92

            color_jittering = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.3,# was 0.8 HK
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

            if self.affine:
                self.transform_op.append(transform_affine)
                self.transform_unnorm.append(transform_affine)
            if self.colorjitter:
                self.transform_op.append(color_jittering)
                self.transform_unnorm.append(color_jittering)
            if self.gaussiansolar:
                self.transform_op.append(global_transfo2_extra)
                self.transform_unnorm.append(global_transfo2_extra)

            # self.transform_op.append(transforms.Resize((crop_size, crop_size)))
            self.transform_op.append(transforms.ToTensor())
            self.transform_unnorm.append(transforms.ToTensor())

            self.transform_unnorm = transforms.Compose(self.transform_unnorm)

            # CutMixTransform(alpha=1.0),  # Custom CutMix (if implemented as tensor op)
            self.transform_op += [normalize]
            self.transform = transforms.Compose(self.transform_op)

            # self.transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Resize((crop_size, crop_size)),
            #     # color_jittering,
            #     # global_transfo1_extra,
            #     # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
            #     # transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1),  # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/augmentations.py#L64
            #     # transforms.CenterCrop(crop_size),
            #     normalize,
            # ])
        else:
            self.transform_op = [
                transforms.Resize((crop_size, crop_size)),
            ]
            self.transform_unnorm =[
                transforms.Resize((crop_size, crop_size)),
            ]

            # if self.crop_upper:
            #     self.transform_op.append(CustomCrop(top=0, left=0, height=int(crop_size*2/3), width=crop_size))
            #     self.transform_op.append(transforms.Resize((crop_size, crop_size)))
            #
            #     self.transform_unnorm.append(CustomCrop(top=0, left=0, height=int(crop_size*2/3), width=crop_size))
            #     self.transform_unnorm.append(transforms.Resize((crop_size, crop_size)))

            self.transform_op.append(transforms.ToTensor())
            self.transform_unnorm.append(transforms.ToTensor())

            self.transform_unnorm = transforms.Compose(self.transform_unnorm)

            # CutMixTransform(alpha=1.0),  # Custom CutMix (if implemented as tensor op)
            self.transform_op += [normalize]
            self.transform = transforms.Compose(self.transform_op)

            # self.transform = transforms.Compose([
            #         transforms.Resize((crop_size, crop_size)),
            #         transforms.ToTensor(),
            #         # transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
            #         # transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.2, hue=0.1),  # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/augmentations.py#L64
            #         # transforms.CenterCrop(crop_size),
            #         normalize,
            #     ])
            # self.transform_unnorm = transforms.Compose([
            #         transforms.Resize((crop_size, crop_size)),
            #         transforms.ToTensor(),
            # ])

        # HK TODO
        # A.PadIfNeeded(min_height=448, min_width=448),

        # train_transform = Compose([
        #     RandomResizedCrop(size=(crop_size, crop_size), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
        #     RandomHorizontalFlip(p=0.5),
        #     ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
        #     ToTensor(),
        #     Normalize(mean=mean, std=std),
        # ])


        # if model == "dinov2":
        #     normalize = T.Normalize([0.5], [0.5])


    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _mirror_horizontally(image):
        """
        Expects images as numpy array of shape (height, width, channel)
        """
        if len(image.shape) == 3:
            return np.flip(image, axis=1)
        elif len(image.shape) == 4:
            return np.flip(image, axis=2)

    @staticmethod
    def _mirror_vertically(image):
        """
        Expects images as numpy array of shape (channel, height, width)
        """
        return np.flip(image, axis=1)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # imageio.imwrite(os.path.join(debug_save_path, str(ii + ix * 64) + '_lbl_' + str(label) + 'imio.png'),
        #           (image))  # Save the image
        # image.save(os.path.join(debug_save_path, str('img0')+ '11.png'))
        unnorm_img = self.transform_unnorm(image)
        if self.flip_hor and self.augment:
            if random.random() < self.flip_hor_prob:
                image = self._mirror_horizontally(np.array(image))
                label = label ^ 1 # i=nvert label
                image = Image.fromarray(image)
        #         Image.fromarray(image).save(os.path.join(debug_save_path, str('img0')+ '11_flip.png'))

        if self.transform:
            try:  # no suduer can fail
                image = self.transform(image)
            except Exception as e:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",e)

        return image, label, unnorm_img
# imageio.imwrite(os.path.join(debug_save_path, str(ii + ix * 64) + '_lbl_' + str(label) + 'imio_trans_yy.png'),
#           (unnorm_img.detach().cpu().permute(1, 2, 0).numpy()*255).astype('uint8'))  # Save the image
def create_dataloader(train_set_path, crop_size=224, num_workers=1, batch_size=32, shuffle=True, augment=False,
                      flip_hor=False, gaussiansolar=False,
                      colorjitter=False, affine=False, crop_upper=False):

    dataset = CustomImageDataset(root_dir=train_set_path, crop_size=crop_size, augment=augment,
                                 flip_hor=flip_hor, gaussiansolar=gaussiansolar,
                                 colorjitter=colorjitter, affine=affine, crop_upper=crop_upper)

    return  DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


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
