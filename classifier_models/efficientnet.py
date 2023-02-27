import torchvision.models
from torch import nn


class EfficientNet(torchvision.models.EfficientNet):
    
    def __new__(cls, pretrained=False):
        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = torchvision.models.efficientnet_b4(num_classes=2, weights=weights)
        model.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(1792, 2, bias=True)
        )
        model.__class__= EfficientNet
        return model
    
    def __init__(self, pretrained, **args):
        pass
    
    
    def get_target_layers(self):
        return [self.layer4[-1]]