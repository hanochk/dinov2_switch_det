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

def load_classifier(fc_dim, n_cls, checkpoint=None, dropout=0.0, var_hidden_dim=False):
    classifier = Classifier(fc_dim=fc_dim, n_cls=n_cls, dropout=dropout, var_hidden_dim=var_hidden_dim)
    if checkpoint is not None:
        classifier.load_state_dict(torch.load(checkpoint))
    classifier = classifier.cuda()    
    return classifier


class Classifier(nn.Module):
    def __init__(self, fc_dim, n_cls=17, dropout=0.0, var_hidden_dim=False):
        super().__init__()
        self.fc_dim = fc_dim
        self.drop = nn.Dropout(dropout)
        self.var_hidden_dim = var_hidden_dim

        if self.var_hidden_dim:
            self.fc1_dim = 2 ** int(np.log2(fc_dim / 2))
        else:
            self.fc1_dim = 128

        self.fc1 = nn.Sequential(
            nn.Linear(fc_dim, self.fc1_dim),
            nn.ReLU(), 
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dim, n_cls),
            nn.Softmax(), 
        )
        self._init_weights(module=self.fc1)
        self._init_weights(module=self.fc2)

    def _init_weights(self, module, init_range: float = 0.1) -> None:
        for name, sub_module in module.named_children():
            if type(sub_module) == nn.Linear:
                sub_module.weight.data.uniform_(-init_range, init_range)
                if hasattr(sub_module.bias, 'data'):
                    sub_module.bias.data.zero_()

    def model_init(self):
        # weight initialization
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x