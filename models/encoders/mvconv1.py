# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

import models.utils

class Encoder(torch.nn.Module):
    def __init__(self, ninputs, tied=False):
        super(Encoder, self).__init__()

        self.ninputs = ninputs
        self.tied = tied

        self.down1 = nn.ModuleList([nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2))
                for i in range(1 if self.tied else self.ninputs)])
        self.down2 = nn.Sequential(
                nn.Linear(256 * self.ninputs * 4 * 3, 512), nn.LeakyReLU(0.2))
        height, width = 512, 334
        ypad = ((height + 127) // 128) * 128 - height
        xpad = ((width + 127) // 128) * 128 - width
        self.pad = nn.ZeroPad2d((xpad // 2, xpad - xpad // 2, ypad // 2, ypad - ypad // 2))
        self.mu = nn.Linear(512, 256)
        self.logstd = nn.Linear(512, 256)

        for i in range(1 if self.tied else self.ninputs):
            models.utils.initseq(self.down1[i])
        models.utils.initseq(self.down2)
        models.utils.initmod(self.mu)
        models.utils.initmod(self.logstd)

    def forward(self, x, losslist=[]):
        x = self.pad(x)
        x = [self.down1[0 if self.tied else i](x[:, i*3:(i+1)*3, :, :]).view(-1, 256 * 3 * 4) for i in range(self.ninputs)]
        x = torch.cat(x, dim=1)
        x = self.down2(x)

        mu, logstd = self.mu(x) * 0.1, self.logstd(x) * 0.01
        if self.training:
            z = mu + torch.exp(logstd) * torch.randn(*logstd.size(), device=logstd.device)
        else:
            z = mu

        losses = {}
        if "kldiv" in losslist:
            losses["kldiv"] = torch.mean(-0.5 - logstd + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * logstd), dim=-1)

        return {"encoding": z, "losses": losses}
