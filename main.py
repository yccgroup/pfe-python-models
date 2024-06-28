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
import rafep
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

    # estimate Z via RAFEP (no threshold)
    Z = driver.run_RAFEP(cfg, samples)
    Zs = [Z]
    log(fd, f"Z_est(inf) = {Z}")

    # remove the sampling outliers
    for threshold in cfg.thresholds:
        E_cut = threshold/cfg.beta + samples.energy_min()
        trunc_traj, trunc_energies = rafep.truncate(samples.traj, samples.energies, E_cut)
        trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj))
        log(fd, f"Truncating samples, threshold={threshold:.3f}, E_cut={E_cut:.3f}, kept {100*trunc_samples.nsamples/samples.nsamples:.3f}%")
        # run RAFEP again
        Z = driver.run_RAFEP(cfg, trunc_samples)
        Zs.append(Z)
        log(fd, f"Z_est({threshold:.3f}) = {Z}")

    # remove the sampling outliers with an automatic approach
    trunc_traj, trunc_energies, Elower_cut, Eupper_cut = rafep.autotruncate(samples.traj, samples.energies, cutoff=0.002, nbins=21)
    trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj))
    log(fd, f"Truncating samples, autotruncate, Elower_cut={Elower_cut:.3f}, Eupper_cut={Eupper_cut:.3f}, kept {100*trunc_samples.nsamples/samples.nsamples:.3f}%")
    # run RAFEP again
    Z = driver.run_RAFEP(cfg, trunc_samples)
    Zs.append(Z)
    log(fd, f"Z_est(auto) = {Z}")

    fd.close()
    os.chdir(origdir)
    return Zs


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
    log(fd, "RAFEP for model potentials")
    log(fd, "-" * 64)
    log(fd, f"git-commit: {get_git_revision()}")
    log(fd, f"program start: {timestamp()}")
    log(fd)
    log(fd, "-------- START INPUT --------")
    cfg.write(fd)
    log(fd, "-------- END INPUT --------")
    log(fd)

    # calculate partition function via numeric integration
    Zint = driver.run_integral(cfg)
    log(fd, f"Z_int = {Zint}")

    # get analytic partition function (where available)
    Zexact = driver.run_exact(cfg)
    if Zexact is None:
        log(fd, "Z_exact not available, using Z_int")
        Zexact = Zint
    else:
        log(fd, f"Z_exact = {Zexact}")

    # run sampling in parallel
    pool = multiprocessing.Pool(nproc)
    Zest_data = pool.starmap(run_once, zip(itertools.repeat(cfg), range(nloop)), chunksize=1)

    # output RAFEP results
    thresholds = [inf] + cfg.thresholds
    for i,threshold in enumerate(thresholds):
        Zest = [ Zs[i] for Zs in Zest_data ]
        np.savetxt(f"Zest_{threshold:.3f}.dat", Zest)
        log(fd, f"Z_est({threshold:.3f}) = {np.mean(Zest)} ± {np.std(Zest)}")
        log(fd, f"Z_est({threshold:.3f})/Z_exact = {np.mean(Zest)/Zexact} ± {np.std(Zest)/Zexact}")


    # output RAFEP autotruncate results
    Zest = [ Zs[i+1] for Zs in Zest_data ]
    np.savetxt(f"Zest_auto.dat", Zest)
    log(fd, f"Z_est(auto) = {np.mean(Zest)} ± {np.std(Zest)}")
    log(fd, f"Z_est(auto)/Z_exact = {np.mean(Zest)/Zexact} ± {np.std(Zest)/Zexact}")

    log(fd)
    log(fd, f"program end: {timestamp()}")
    log(fd, "-" * 64)

  except AppError as e:

    sys.stderr.write(f"ERROR: {e.msg}\n")
    sys.exit(1)

