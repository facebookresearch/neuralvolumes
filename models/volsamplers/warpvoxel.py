# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class VolSampler(nn.Module):
    def __init__(self, displacementwarp=False):
        super(VolSampler, self).__init__()

        self.displacementwarp = displacementwarp

    def forward(self, pos, template, warp=None, gwarps=None, gwarprot=None, gwarpt=None, viewtemplate=False, **kwargs):
        valid = None
        if not viewtemplate:
            if gwarps is not None:
                pos = (torch.sum(
                    (pos - gwarpt[:, None, None, None, :])[:, :, :, :, None, :] *
                    gwarprot[:, None, None, None, :, :], dim=-1) *
                    gwarps[:, None, None, None, :])
            if warp is not None:
                if self.displacementwarp:
                    pos = pos + F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
                else:
                    valid = torch.prod((pos > -1.) * (pos < 1.), dim=-1).float()
                    pos = F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
        val = F.grid_sample(template, pos)
        if valid is not None:
            val = val * valid[:, None, :, :, :]
        return val[:, :3, :, :, :], val[:, 3:, :, :, :]
