import torchvision.models
from torch import nn
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial
# import timm.models.maxxvit

class MaxViT(torchvision.models.MaxVit):
# class MaxViT(timm.models.maxxvit.MaxxVit):

    def __new__(cls, input_channels=3, pretrained=False):
        
        if pretrained:
            # model = timm.models.maxxvit.maxvit_rmlp_small_rw_224(pretrained=True, num_classes=2)
            # model = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.DEFAULT)
            model = torchvision.models.maxvit_t(weights=None)
            model.stem.insert(0, nn.Conv2d(input_channels, input_channels, kernel_size=2, stride=2))
            # for block in model.blocks[:-1]:
            #     for param in block.parameters():
            #         param.requires_grad = False
            model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(512),
                nn.Linear(512, 512),
                nn.Tanh(),
                nn.Linear(512, 2, bias=False),
        )
        else:
            stem_channels = 64
            model = torchvision.models.MaxVit(
                input_size=(448, 448),
                stem_channels=stem_channels,
                partition_size=7, 
                block_channels=[64, 128, 256, 512],
                block_layers=[2, 2, 5, 2],
                head_dim=32,
                stochastic_depth_prob=0.2,
                num_classes=2
            )
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99)
            model.stem = nn.Sequential(
                Conv2dNormActivation(
                    input_channels,
                    stem_channels,
                    3,
                    stride=2,
                    norm_layer=norm_layer,
                    activation_layer=nn.GELU,
                    bias=False,
                    inplace=None,
                ),
                Conv2dNormActivation(
                    stem_channels, stem_channels, 3, stride=1, norm_layer=None, activation_layer=None, bias=True
                ),
            )
        # model = torchvision.models.maxvit_t(num_classes=2, weights=None)
        model.__class__ = MaxViT
        return model
    
    def __init__(self, input_channels=3, pretrained=False, **kwargs):
        pass

    # def forward(self, x):
    #     x = self.stem(x)
    #     for block in self.blocks:
    #         x = block(x)
    #     return self.classifier(x), x

    def get_target_layers(self):
        return [self.blocks[-1].layers[-1]]