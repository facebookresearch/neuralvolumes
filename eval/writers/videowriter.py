# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import multiprocessing
import os
import shutil
import subprocess

import numpy as np

import matplotlib.cm as cm

from PIL import Image

def writeimage(x):
    randid, itemnum, imgout = x

    imgout = np.clip(np.clip(imgout / 255., 0., 255.) ** (1. / 1.8) * 255., 0., 255).astype(np.uint8)

    if imgout.shape[1] % 2 != 0:
        imgout = imgout[:, :-1]

    Image.fromarray(imgout).save("/tmp/{}/{:06}.jpg".format(randid, itemnum))

class Writer():
    def __init__(self, outpath, showtarget=False, showdiff=False, bgcolor=[0., 0., 0.], colcorrect=[1.35, 1.16, 1.5], nthreads=16):
        self.outpath = outpath
        self.showtarget = showtarget
        self.showdiff = showdiff
        self.bgcolor = np.array(bgcolor, dtype=np.float32)
        self.colcorrect = np.array(colcorrect, dtype=np.float32)

        # set up temporary output
        self.randid = ''.join([str(x) for x in np.random.randint(0, 9, size=10)])
        try:
            os.makedirs("/tmp/{}".format(self.randid))
        except OSError:
            pass

        self.writepool = multiprocessing.Pool(nthreads)
        self.nitems = 0

    def batch(self, iternum, itemnum, irgbrec, ialpharec=None, image=None, irgbsqerr=None, **kwargs):
        irgbrec = irgbrec.data.to("cpu").numpy().transpose((0, 2, 3, 1))
        if ialpharec is not None:
            ialpharec = ialpharec.data.to("cpu").numpy()[:, 0, :, :, None]
        else:
            ialpharec = 1.0

        # color correction
        imgout = irgbrec * self.colcorrect[None, None, None, :]

        # composite background color
        imgout = imgout + (1. - ialpharec) * self.bgcolor[None, None, None, :]

        # concatenate ground truth image
        if self.showtarget and image is not None:
            image = image.data.to("cpu").numpy().transpose((0, 2, 3, 1))
            image = image * self.colcorrect[None, None, None, :]
            imgout = np.concatenate((imgout, image), axis=2)

        # concatenate difference image
        if self.showdiff and imagediff is not None:
            irgbsqerr = np.mean(irgbsqerr.data.to("cpu").numpy(), axis=1)
            irgbsqerr = (cm.magma(4. * irgbsqerr / 255.)[:, :, :, :3] * 255.)
            imgout = np.concatenate((imgout, irgbsqerr), axis=2)

        self.writepool.map(writeimage,
                zip([self.randid for i in range(itemnum.size(0))],
                    itemnum.data.to("cpu").numpy(),
                    imgout))
        self.nitems += itemnum.size(0)

    def finalize(self):
        # make video file
        command = (
                "ffmpeg -y -r 30 -i /tmp/{}/%06d.jpg "
                "-vframes {} "
                "-vcodec libx264 -crf 18 "
                "-pix_fmt yuv420p "
                "{}".format(self.randid, self.nitems, self.outpath)
                ).split()
        subprocess.call(command)

        shutil.rmtree("/tmp/{}".format(self.randid))
