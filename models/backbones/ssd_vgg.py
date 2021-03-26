import torch.nn as nn
from models.backbones.vgg import VGG
from models.backbones.utils import kaiming_init, normal_init, constant_init


class SSDVGG(VGG):
    extra_setting = {
        300: (256, "S", 512, 128, "S", 256, 128, 256, 128, 256),
        512: (256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128)
    }

    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.):
        super(SSDVGG, self).__init__(
            depth=depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices
        )
        assert input_size in (300, 512)
        self.input_size = input_size
        self.features.add_module(
            str(len(self.features)),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        )
        self.features.add_module(
            str(len(self.features)),
            nn.ReLU(inplace=True)
        )
        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )
        self.features.add_module(
            str(len(self.features)),
            nn.ReLU(inplace=True)
        )
        self.out_feature_indices = out_feature_indices
        self.inplanes = 1024
        self.extra = self._make_extra_layers(self.extra_setting[input_size])


    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        number_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == "S":
                self.inplanes = outplane
                continue
            k = kernel_sizes[number_layers % 2]
            if outplanes[i] == "S":
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            number_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)



