import torch
import torch.nn as nn
from torchvision.models import resnet50
import numpy as np

def load_dinov2(flavour='small'):
    if flavour == 'small':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif flavour == 'base':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif flavour == 'large':
        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    dinov2_vits14.eval()
    dinov2_vits14 = dinov2_vits14.cuda()
    return dinov2_vits14

def load_resnet50():
    resnet = resnet50(pretrained=True)
    resnet.fc = nn.Identity()
    resnet.eval()
    resnet = resnet.cuda()
    return resnet

def load_backbone(model="dinov2", flavour='small'):
    if model == "dinov2":
        backbone = load_dinov2(flavour=flavour)
    elif model == "resnet50":
        backbone = load_resnet50()
    return backbone

def load_classifier(fc_dim, n_cls, checkpoint=None, dropout=0.0):
    classifier = Classifier(fc_dim=fc_dim, n_cls=n_cls, dropout=dropout)
    if checkpoint is not None:
        classifier.load_state_dict(torch.load(checkpoint))
    classifier = classifier.cuda()    
    return classifier


class Classifier(nn.Module):
    def __init__(self, fc_dim, n_cls=17, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        if 1:
            self.fc1_dim = 128
        else:
            self.fc1_dim = 2 ** int(np.log2(fc_dim / 2))

        self.fc1 = nn.Sequential(
            nn.Linear(fc_dim, self.fc1_dim),
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dim, n_cls),
            nn.Softmax(), 
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x