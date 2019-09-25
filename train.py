# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import argparse
import gc
import importlib
import importlib.util
import os
import sys
import time
sys.dont_write_bytecode = True

import numpy as np

import torch
import torch.utils.data
torch.backends.cudnn.benchmark = True # gotta go fast!

class Logger(object):
    """Duplicates all stdout to a file."""
    def __init__(self, path, resume):
        if not resume and os.path.exists(path):
            print(path + " exists")
            sys.exit(0)

        iternum = 0
        if resume:
            with open(path, "r") as f:
                for line in f.readlines():
                    match = re.search("Iteration (\d+).* ", line)
                    if match is not None:
                        it = int(match.group(1))
                        if it > iternum:
                            iternum = it
        self.iternum = iternum

        self.log = open(path, "a") if resume else open(path, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.dirname(args.experconfig)
    log = Logger("{}/log.txt".format(outpath), args.resume)
    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    starttime = time.time()
    experconfig = import_module(args.experconfig, "config")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progressprof = experconfig.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - starttime))

    # build dataset & testing dataset
    starttime = time.time()
    testdataset = progressprof.get_dataset()
    dataloader = torch.utils.data.DataLoader(testdataset, batch_size=progressprof.batchsize, shuffle=False, drop_last=True, num_workers=0)
    for testbatch in dataloader:
        break
    dataset = profile.get_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=profile.batchsize, shuffle=True, drop_last=True, num_workers=16)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    writer = progressprof.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset)
    ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").train()
    if args.resume:
        ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    # build optimizer
    starttime = time.time()
    aeoptim = profile.get_optimizer(ae.module)
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    iternum = log.iternum
    prevloss = np.inf

    for epoch in range(10000):
        for data in dataloader:
            # forward
            output = ae(iternum, lossweights.keys(), **{k: x.to("cuda") for k, x in data.items()})

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in output["losses"].items()])

            # print current information
            print("Iteration {}: loss = {:.5f}, ".format(iternum, float(loss.item())) +
                    ", ".join(["{} = {:.5f}".format(k,
                        float(torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                        for k, v in output["losses"].items()]), end="")
            if iternum % 10 == 0:
                endtime = time.time()
                ips = 10. / (endtime - starttime)
                print(", iter/sec = {:.2f}".format(ips))
                starttime = time.time()
            else:
                print()

            # compute evaluation output
            if iternum in evalpoints:
                with torch.no_grad():
                    testoutput = ae(iternum, [], **{k: x.to("cuda") for k, x in testbatch.items()}, **progressprof.get_ae_args())

                b = data["campos"].size(0)
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(b), **testbatch, **testoutput)

            # update parameters
            aeoptim.zero_grad()
            loss.backward()
            aeoptim.step()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                print("Unstable loss function; resetting")

                ae.module.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = profile.get_optimizer(ae.module)

            prevloss = loss.item()

            # save intermediate results
            if iternum % 1000 == 0:
                torch.save(ae.module.state_dict(), "{}/aeparams.pt".format(outpath))

            iternum += 1

        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
