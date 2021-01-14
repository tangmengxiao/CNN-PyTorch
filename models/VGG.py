import torch.nn as nn
import torch
import torch.functional as F


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features  # get feature extractor
        self.classifer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512*7*7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xarier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

# network config dict
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_features(cfg: list):
    layers = []
    in_channels = 3  # initial input image channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # update in_channels
    return nn.Sequential(*layers)  # * 非关键字参数


def vgg(model_name='vgg16', **kwargs): # kwargs for VGG construction
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model num {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg, **kwargs))
    return model

