# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, dataset, encoder, decoder, volsampler, colorcal, dt, stepjitter=0.01, estimatebg=False):
        super(Autoencoder, self).__init__()

        self.estimatebg = estimatebg
        self.allcameras = dataset.get_allcameras()

        self.encoder = encoder
        self.decoder = decoder
        self.volsampler = volsampler
        self.bg = nn.ParameterDict({
            k: nn.Parameter(torch.ones(3, v["size"][1], v["size"][0]), requires_grad=estimatebg)
            for k, v in dataset.get_krt().items()})
        self.colorcal = colorcal
        self.dt = dt
        self.stepjitter = stepjitter

        self.imagemean = dataset.imagemean
        self.imagestd = dataset.imagestd

        if dataset.known_background():
            dataset.get_background(self.bg)

    # omit background from state_dict if it's not being estimated
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(Autoencoder, self).state_dict(destination, prefix, keep_vars)
        if not self.estimatebg:
            for k in self.bg.keys():
                del ret[prefix+"bg."+k]
        return ret

    def forward(self, iternum, losslist, camrot, campos, focal, princpt, pixelcoords, validinput,
            fixedcamimage=None, encoding=None, keypoints=None, camindex=None,
            image=None, imagevalid=None, viewtemplate=False,
            outputlist=[]):
        result = {"losses": {}}

        # encode input or get encoding
        if encoding is None:
            encout = self.encoder(fixedcamimage, losslist)
            encoding = encout["encoding"]
            result["losses"].update(encout["losses"])

        # decode
        decout = self.decoder(encoding, campos, losslist)
        result["losses"].update(decout["losses"])

        # NHWC
        raydir = (pixelcoords - princpt[:, None, None, :]) / focal[:, None, None, :]
        raydir = torch.cat([raydir, torch.ones_like(raydir[:, :, :, 0:1])], dim=-1)
        raydir = torch.sum(camrot[:, None, None, :, :] * raydir[:, :, :, :, None], dim=-2)
        raydir = raydir / torch.sqrt(torch.sum(raydir ** 2, dim=-1, keepdim=True))

        # compute raymarching starting points
        with torch.no_grad():
            t1 = (-1.0 - campos[:, None, None, :]) / raydir
            t2 = ( 1.0 - campos[:, None, None, :]) / raydir
            tmin = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                   torch.max(torch.min(t1[..., 1], t2[..., 1]),
                             torch.min(t1[..., 2], t2[..., 2])))
            tmax = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                   torch.min(torch.max(t1[..., 1], t2[..., 1]),
                             torch.max(t1[..., 2], t2[..., 2])))

            intersections = tmin < tmax
            t = torch.where(intersections, tmin, torch.zeros_like(tmin)).clamp(min=0.)
            tmin = torch.where(intersections, tmin, torch.zeros_like(tmin))
            tmax = torch.where(intersections, tmax, torch.zeros_like(tmin))

        # random starting point
        t = t - self.dt * torch.rand_like(t)

        raypos = campos[:, None, None, :] + raydir * t[..., None] # NHWC
        rayrgb = torch.zeros_like(raypos.permute(0, 3, 1, 2)) # NCHW
        rayalpha = torch.zeros_like(rayrgb[:, 0:1, :, :]) # NCHW

        # raymarch
        done = torch.zeros_like(t).bool()
        while not done.all():
            valid = torch.prod(torch.gt(raypos, -1.0) * torch.lt(raypos, 1.0), dim=-1).byte()
            validf = valid.float()

            sample_rgb, sample_alpha = self.volsampler(raypos[:, None, :, :, :], **decout, viewtemplate=viewtemplate)

            with torch.no_grad():
                step = self.dt * torch.exp(self.stepjitter * torch.randn_like(t))
                done = done | ((t + step) >= tmax)

            contrib = ((rayalpha + sample_alpha[:, :, 0, :, :] * step[:, None, :, :]).clamp(max=1.) - rayalpha) * validf[:, None, :, :]

            rayrgb = rayrgb + sample_rgb[:, :, 0, :, :] * contrib
            rayalpha = rayalpha + contrib

            raypos = raypos + raydir * step[:, :, :, None]
            t = t + step

        if image is not None:
            imagesize = torch.tensor(image.size()[3:1:-1], dtype=torch.float32, device=pixelcoords.device)
            samplecoords = pixelcoords * 2. / (imagesize[None, None, None, :] - 1.) - 1.

        # color correction / bg
        if camindex is not None:
            rayrgb = self.colorcal(rayrgb, camindex)

            if pixelcoords.size()[1:3] != image.size()[2:4]:
                bg = F.grid_sample(
                        torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(campos.size(0))], dim=0),
                        samplecoords)
            else:
                bg = torch.stack([self.bg[self.allcameras[camindex[i].item()]] for i in range(campos.size(0))], dim=0)

            rayrgb = rayrgb + (1. - rayalpha) * bg.clamp(min=0.)

        if "irgbrec" in outputlist:
            result["irgbrec"] = rayrgb
        if "ialpharec" in outputlist:
            result["ialpharec"] = rayalpha

        # opacity prior
        if "alphapr" in losslist:
            alphaprior = torch.mean(
                    torch.log(0.1 + rayalpha.view(rayalpha.size(0), -1)) +
                    torch.log(0.1 + 1. - rayalpha.view(rayalpha.size(0), -1)) - -2.20727, dim=-1)
            result["losses"]["alphapr"] = alphaprior

        # irgb loss
        if image is not None:
            if pixelcoords.size()[1:3] != image.size()[2:4]:
                image = F.grid_sample(image, samplecoords)

            # standardize
            rayrgb = (rayrgb - self.imagemean) / self.imagestd
            image = (image - self.imagemean) / self.imagestd

            # compute reconstruction loss weighting
            if imagevalid is not None:
                weight = imagevalid[:, None, None, None].expand_as(image) * validinput[:, None, None, None]
            else:
                weight = torch.ones_like(image) * validinput[:, None, None, None]

            irgbsqerr = weight * (image - rayrgb) ** 2

            if "irgbsqerr" in outputlist:
                result["irgbsqerr"] = rgbsqerr

            if "irgbmse" in losslist:
                irgbmse = torch.sum(irgbsqerr.view(irgbsqerr.size(0), -1), dim=-1)
                irgbmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                result["losses"]["irgbmse"] = (irgbmse, irgbmse_weight)

        return result
