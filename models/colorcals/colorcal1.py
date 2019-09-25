# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

class Colorcal(nn.Module):
    """Apply learnable 3 channel scale and bias to an image to handle un(color)calibrated cameras."""
    def __init__(self, allcameras):
        super(Colorcal, self).__init__()

        self.allcameras = allcameras

        self.conv = nn.ModuleDict({
            k: nn.Conv2d(3, 3, 1, 1, 0, groups=3) for k in self.allcameras})

        for k in self.allcameras:
            self.conv[k].weight.data[:] = 1.
            self.conv[k].bias.data.zero_()

    def forward(self, image, camindex):
        return torch.cat([self.conv[self.allcameras[camindex[i].item()]](image[i:i+1, :, :, :]) for i in range(image.size(0))])
