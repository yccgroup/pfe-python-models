#!/usr/bin/env python3

import sys
import os
import argparse
import numpy as np
import multiprocessing
import itertools

import config
import potentials
import sampling
import rafep
import driver
from util import *
from trajectory import *


def run_once(cfg, it):

    # set output directory
    origdir = os.getcwd()
    outdir = f"out-{it:04d}"
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass
    os.chdir(outdir)
    fd = open("log", "w")

    # set the random seed
    try:
        seed = cfg.input.getint('trajectory', 'seed')
        np.random.seed(seed + it)
    except ValueError:
        np.random.seed()

    # generate the samples, depending on the selected method
    method = cfg.get('trajectory', 'method')
    savetraj = cfg.getboolean('trajectory', 'save')
    log(fd, f"Generating samples via {method}...")
    if method == 'monte-carlo':
        samples = driver.run_MC(cfg)
    elif method == 'replica-exchange':
        samples = driver.run_REMC(cfg)
    else:
        error(f"unknown method '{method}'")
    if savetraj:
        samples.save()

    # Statistics
    nsteps = cfg.input.getint('trajectory', 'nsteps')
    log(fd, f"Acceptance rate = {samples.naccept/nsteps}")
    log(fd, f"Average energy  = {samples.energy_mean()}")
    log(fd, f"Minimum energy  = {samples.energy_min()}")
    log(fd, f"0.5 * kB * T    = {0.5/cfg.beta}")
    log(fd)

    # estimate Z via RAFEP
    Z1 = driver.run_RAFEP(cfg, samples)
    log(fd, f"RAFEP Zest = {Z1}")

    # remove the sampling outliers
    threshold = cfg.input.getfloat('rafep', 'threshold')
    E_cut = threshold/cfg.beta + samples.energy_min()
    trunc_traj, trunc_energies = rafep.truncate(samples.traj, samples.energies, E_cut)
    trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj))
    log(fd, f"Truncated samples, threshold={threshold}, kept {100*trunc_samples.nsamples/samples.nsamples:.3f}%")

    # run RAFEP again
    Z2 = driver.run_RAFEP(cfg, trunc_samples)
    log(fd, f"Model Zest = {Z2}")
    fd.close()
 
    os.chdir(origdir)
    return Z1,Z2


if __name__ == '__main__':

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
    Zest_list = pool.starmap(run_once, zip(itertools.repeat(cfg), range(nloop)), chunksize=1)
    Zest1 = [ res[0] for res in Zest_list ]
    Zest2 = [ res[1] for res in Zest_list ]

    # output RAFEP results
    np.savetxt("RAFEP_Zest.dat", Zest1)
    np.savetxt("Model_Zest.dat", Zest2)
    log(fd, f"RAFEP Z_est = {np.mean(Zest1)} ± {np.std(Zest1)}")
    log(fd, f"RAFEP Z_est/Z_exact = {np.mean(Zest1)/Zexact} ± {np.std(Zest1)/Zexact}")
    log(fd, f"Model Z_est = {np.mean(Zest2)} ± {np.std(Zest2)}")
    log(fd, f"Model Z_est/Z_exact = {np.mean(Zest2)/Zexact} ± {np.std(Zest2)/Zexact}")

