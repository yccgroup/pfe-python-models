#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import multiprocessing
import itertools
from math import inf

import config
import potentials
import sampling
import pfe
import driver
from util import *
from trajectory import *


def set_seed(cfg, it):
    seed_text = cfg.get('trajectory', 'seed')
    try:
        seed = int(seed_text)
        np.random.seed(seed + it)
    except ValueError:
        np.random.seed()


def run_once(cfg, it):

    # set output directory
    origdir = os.getcwd()
    outdir = f"out-{it:04d}"
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    os.chdir(outdir)

    # generate the samples, depending on the selected method
    method = cfg.get('trajectory', 'method')

    if method == 'read':
        fd = open("log", "a")
        log(fd, "-" * 64)
        savetraj = False
        log(fd, f"Reading samples...")
        samples = driver.run_readtraj(cfg,it)
        nsteps = cfg.getint('trajectory', 'nsteps')
        outfreq = cfg.getint('trajectory', 'outfreq')
        nsamples = nsteps // outfreq
        if nsamples < samples.nsamples:
            log(fd, f"Truncating trajectory to {nsamples} samples...")
            samples.truncate(nsamples)

    else:
        fd = open("log", "w")
        savetraj = cfg.getboolean('trajectory', 'save')
        log(fd, f"Generating samples via {method}...")
        set_seed(cfg, it)
        if method == 'monte-carlo':
            samples = driver.run_MC(cfg)
        elif method == 'replica-exchange':
            samples = driver.run_REMC(cfg)
        else:
            error(f"unknown method '{method}'")
        if savetraj:
            samples.save()

        # Statistics
        nsteps = cfg.getint('trajectory', 'nsteps')
        log(fd, f"Acceptance rate = {samples.naccept/nsteps}")
        log(fd, f"Average energy  = {samples.energy_mean()}")
        log(fd, f"Minimum energy  = {samples.energy_min()}")
        log(fd, f"0.5 * kB * T    = {0.5/cfg.beta}")
        log(fd)

    # output the energy histogram for analysis
    Emin = 0.0
    counts, Evalues = np.histogram(samples.energies, range=(Emin, Emin + 12.0/cfg.beta), bins=24)
    outdata = np.column_stack((Evalues[0:-1],counts))
    np.savetxt("histogram.dat",outdata,fmt="%.10f %d")

    # estimate Z via PFE (no threshold)
    N = samples.nsamples
    lnZ,varlnZ,tryEstar = driver.run_PFE(cfg, samples, N)
    sigmalnZ = np.sqrt(varlnZ)
    lnZs = [lnZ]
    sigmalnZs = [sigmalnZ]
    log(fd, f"ln Z_est(0.000) = {lnZ}   [ σ(lnZ) = {sigmalnZ} ]")

    # remove the sampling outliers
    for threshold in cfg.thresholds:
        trunc_traj, trunc_energies, Elower, Eupper = pfe.autotruncate(samples.traj, samples.energies, threshold)
        trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj))
        log(fd, f"Truncating samples, threshold={threshold:.3f}, Elower={Elower:.3f}, Eupper={Eupper:.3f}, kept {100*trunc_samples.nsamples/samples.nsamples:.3f}%")
        # run PFE again
        lnZ,varlnZ,tryEstar = driver.run_PFE(cfg, trunc_samples, N)
        sigmalnZ = np.sqrt(varlnZ)
        lnZs.append(lnZ)
        sigmalnZs.append(sigmalnZ)
        log(fd, f"ln Z_est({threshold:.3f}) = {lnZ}   [ σ(lnZ) = {sigmalnZ}   E* -> {tryEstar:.3f} ]")

    fd.close()
    os.chdir(origdir)
    return (lnZs,sigmalnZs)


if __name__ == '__main__':

  try:

    # parse arguments
    p = argparse.ArgumentParser(description="Generate samples and estimate Z.")
    p.add_argument('inputfile',
        help='name of the input file')
    p.add_argument('--param', '-p', metavar='P', action='append', default=[],
        help='override parameter from the input file; syntax: section.setting=value')
    args = p.parse_args()

    # read the configuration file, and apply command line settings
    cfg = config.Config(args.inputfile, args.param)

    # read meta parameters
    nproc = cfg.getint('meta', 'nproc')
    nloop = cfg.getint('meta', 'nloop')

    # generate output
    fd = sys.stdout
    log(fd, "PFE for model potentials")
    log(fd, "-" * 64)
    log(fd, f"git-commit: {get_git_revision()}")
    log(fd, f"program start: {timestamp()}")
    log(fd)
    log(fd, "-------- START INPUT --------")
    cfg.write(fd)
    log(fd, "-------- END INPUT --------")
    log(fd)

    # calculate partition function via numeric integration
    lnZint = driver.run_integral(cfg)
    log(fd, f"ln Z_int = {lnZint}")

    # get analytic partition function (where available)
    lnZexact = driver.run_exact(cfg)
    if lnZexact is None:
        log(fd, "Z_exact not available, using Z_int")
        lnZexact = lnZint
    else:
        log(fd, f"ln Z_exact = {lnZexact}")

    # run sampling in parallel
    pool = multiprocessing.Pool(nproc)
    data = pool.starmap(run_once, zip(itertools.repeat(cfg), range(nloop)), chunksize=1)

    # output PFE results
    thresholds = [0.0] + cfg.thresholds
    for i,threshold in enumerate(thresholds):
        lnZest = [ lnZs[i] for (lnZs,sigmalnZs) in data ]
        sigmalnZest = [ sigmalnZs[i] for (lnZs,sigmalnZs) in data ]
        np.savetxt(f"lnZest_{threshold:.3f}.dat", np.column_stack([lnZest,sigmalnZest]))
        log(fd, f"ln Z_est({threshold:.3f}) = {np.mean(lnZest)} ± {np.std(lnZest)} ({np.mean(sigmalnZest)})")
        log(fd, f"ln Z_est({threshold:.3f}) - ln Z_exact = {np.mean(lnZest)-lnZexact} ± {np.std(lnZest)}")

    log(fd)
    log(fd, f"program end: {timestamp()}")
    log(fd, "-" * 64)

  except AppError as e:

    sys.stderr.write(f"ERROR: {e.msg}\n")
    sys.exit(1)

