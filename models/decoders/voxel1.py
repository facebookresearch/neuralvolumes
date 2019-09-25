# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils

class ConvTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(ConvTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        # build template convolution stack
        self.template1 = nn.Sequential(nn.Linear(self.encodingsize, 1024), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 1024, 512
        for i in range(int(np.log2(self.templateres)) - 1):
            template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if inchannels == outchannels:
                outchannels = inchannels // 2
            else:
                inchannels = outchannels
        template2.append(nn.ConvTranspose3d(inchannels, 4, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template2(self.template1(encoding).view(-1, 1024, 1, 1, 1))

class LinearTemplate(nn.Module):
    def __init__(self, encodingsize=256, outchannels=4, templateres=128):
        super(LinearTemplate, self).__init__()

        self.encodingsize = encodingsize
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(
            nn.Linear(self.encodingsize, 8), nn.LeakyReLU(0.2),
            nn.Linear(8, self.templateres ** 3 * self.outchannels))

        for m in [self.template1]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template1(encoding).view(-1, self.outchannels, self.templateres, self.templateres, self.templateres)

def gettemplate(templatetype, **kwargs):
    if templatetype == "conv":
        return ConvTemplate(**kwargs)
    elif templatetype == "affinemix":
        return LinearTemplate(**kwargs)
    else:
        return None

class ConvWarp(nn.Module):
    def __init__(self, displacementwarp=False, **kwargs):
        super(ConvWarp, self).__init__()

        self.displacementwarp = displacementwarp

        self.warp1 = nn.Sequential(
                nn.Linear(256, 1024), nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(
                nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.ConvTranspose3d(256, 3, 4, 2, 1))
        for m in [self.warp1, self.warp2]:
            models.utils.initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=0)[None].astype(np.float32)))

    def forward(self, encoding):
        finalwarp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1, 1)) * (2. / 1024)
        if not self.displacementwarp:
            finalwarp = finalwarp + self.grid
        return finalwarp

class AffineMixWarp(nn.Module):
    def __init__(self, **kwargs):
        super(AffineMixWarp, self).__init__()

        self.quat = models.utils.Quaternion()

        self.warps = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3*16))
        self.warpr = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 4*16))
        self.warpt = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3*16))
        self.weightbranch = nn.Sequential(
                nn.Linear(256, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, 16*32*32*32))
        for m in [self.warps, self.warpr, self.warpt, self.weightbranch]:
            models.utils.initseq(m)

        zgrid, ygrid, xgrid = np.meshgrid(
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32),
                np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((xgrid, ygrid, zgrid), axis=-1)[None].astype(np.float32)))

    def forward(self, encoding):
        warps = self.warps(encoding).view(encoding.size(0), 16, 3)
        warpr = self.warpr(encoding).view(encoding.size(0), 16, 4)
        warpt = self.warpt(encoding).view(encoding.size(0), 16, 3) * 0.1
        warprot = self.quat(warpr.view(-1, 4)).view(encoding.size(0), 16, 3, 3)

        weight = torch.exp(self.weightbranch(encoding).view(encoding.size(0), 16, 32, 32, 32))

        warpedweight = torch.cat([
            F.grid_sample(weight[:, i:i+1, :, :, :],
                torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                    warprot[:, None, None, None, i, :, :]), dim=5) *
                    warps[:, None, None, None, i, :], padding_mode='border')
            for i in range(weight.size(1))], dim=1)

        warp = torch.sum(torch.stack([
            warpedweight[:, i, :, :, :, None] *
            (torch.sum(((self.grid - warpt[:, None, None, None, i, :])[:, :, :, :, None, :] *
                warprot[:, None, None, None, i, :, :]), dim=5) *
                warps[:, None, None, None, i, :])
            for i in range(weight.size(1))], dim=1), dim=1) / torch.sum(warpedweight, dim=1).clamp(min=0.001)[:, :, :, :, None]

        return warp.permute(0, 4, 1, 2, 3)

def getwarp(warptype, **kwargs):
    if warptype == "conv":
        return ConvWarp(**kwargs)
    elif warptype == "affinemix":
        return AffineMixWarp(**kwargs)
    else:
        return None

class Decoder(nn.Module):
    def __init__(self, templatetype="conv", templateres=128,
            viewconditioned=False, globalwarp=True, warptype="affinemix",
            displacementwarp=False):
        super(Decoder, self).__init__()

        self.templatetype = templatetype
        self.templateres = templateres
        self.viewconditioned = viewconditioned
        self.globalwarp = globalwarp
        self.warptype = warptype
        self.displacementwarp = displacementwarp

        if self.viewconditioned:
            self.template = gettemplate(self.templatetype, encodingsize=256+3,
                    outchannels=3, templateres=self.templateres)
            self.templatealpha = gettemplate(self.templatetype, encodingsize=256,
                    outchannels=1, templateres=self.templateres)
        else:
            self.template = gettemplate(self.templatetype, templateres=self.templateres)

        self.warp = getwarp(self.warptype, displacementwarp=self.displacementwarp)

        if self.globalwarp:
            self.quat = models.utils.Quaternion()

            self.gwarps = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 3))
            self.gwarpr = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 4))
            self.gwarpt = nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.2),
                    nn.Linear(128, 3))

            initseq = models.utils.initseq
            for m in [self.gwarps, self.gwarpr, self.gwarpt]:
                initseq(m)

    def forward(self, encoding, viewpos, losslist=[]):
        scale = torch.tensor([25., 25., 25., 1.], device=encoding.device)[None, :, None, None, None]
        bias = torch.tensor([100., 100., 100., 0.], device=encoding.device)[None, :, None, None, None]

        # run template branch
        viewdir = viewpos / torch.sqrt(torch.sum(viewpos ** 2, dim=-1, keepdim=True))
        templatein = torch.cat([encoding, viewdir], dim=1) if self.viewconditioned else encoding
        template = self.template(templatein)
        if self.viewconditioned:
            # run alpha branch without viewpoint information
            template = torch.cat([template, self.templatealpha(encoding)], dim=1)
        # scale up to 0-255 range approximately
        template = F.softplus(bias + scale * template)

        # compute warp voxel field
        warp = self.warp(encoding) if self.warp is not None else None

        if self.globalwarp:
            # compute single affine transformation
            gwarps = 1.0 * torch.exp(0.05 * self.gwarps(encoding).view(encoding.size(0), 3))
            gwarpr = self.gwarpr(encoding).view(encoding.size(0), 4) * 0.1
            gwarpt = self.gwarpt(encoding).view(encoding.size(0), 3) * 0.025
            gwarprot = self.quat(gwarpr.view(-1, 4)).view(encoding.size(0), 3, 3)

        losses = {}

        # tv-L1 prior
        if "tvl1" in losslist:
            logalpha = torch.log(1e-5 + template[:, -1, :, :, :])
            losses["tvl1"] = torch.mean(torch.sqrt(1e-5 +
                (logalpha[:, :-1, :-1, 1:] - logalpha[:, :-1, :-1, :-1]) ** 2 +
                (logalpha[:, :-1, 1:, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2 +
                (logalpha[:, 1:, :-1, :-1] - logalpha[:, :-1, :-1, :-1]) ** 2))

        return {"template": template, "warp": warp,
                **({"gwarps": gwarps, "gwarprot": gwarprot, "gwarpt": gwarpt} if self.globalwarp else {}),
                "losses": losses}
