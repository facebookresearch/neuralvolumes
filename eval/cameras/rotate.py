# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, period=128):
        self.length = length
        self.period = period
        self.width, self.height = 480, 640

        self.focal = np.array([1000. * (self.width / 480.), 1000. * (self.width / 480.)], dtype=np.float32)
        self.princpt = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float32)

    def __len__(self):
        return self.length

    def get_allcameras(self):
        return ["rotate"]

    def get_krt(self):
        return {"rotate": {
                "focal": self.focal,
                "princpt": self.princpt,
                "size": np.array([self.width, self.height])}}

    def __getitem__(self, idx):
        t = (np.cos(idx * 2. * np.pi / self.period) * 0.5 + 0.5)
        x = np.cos(t * 0.5 * np.pi + 0.25 * np.pi) * 3.
        y = 0.5
        z = np.sin(t * 0.5 * np.pi + 0.25 * np.pi) * 3.
        campos = np.array([x, y, z], dtype=np.float32)

        lookat = np.array([0., 0., 0.], dtype=np.float32)
        up = np.array([0., -1., 0.], dtype=np.float32)
        forward = lookat - campos
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        camrot = np.array([right, up, forward], dtype=np.float32)

        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixelcoords = np.stack((px, py), axis=-1)

        return {"campos": campos,
                "camrot": camrot,
                "focal": self.focal,
                "princpt": self.princpt,
                "pixelcoords": pixelcoords}
