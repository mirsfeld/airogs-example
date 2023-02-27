import torchvision.models
from torch import nn


class ResNet(torchvision.models.ResNet):
    
    def __new__(cls, pretrained=False):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet50(num_classes=2, weights=weights)
        model.fc = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(2048, 2, bias=True)
        )
        model.__class__= ResNet
        return model
    
    def __init__(self, pretrained, **args):
        pass
    
    
    def get_target_layers(self):
        return [self.layer4[-1]]