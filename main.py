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
        elif nsamples > samples.nsamples:
            error(f"cannot read {nsamples} samples, trajectory only has {samples.nsamples}")

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

   
    # calculate partition function via a naive summation
    lnZnaive = driver.run_naive(cfg, samples)
    print("lnZ_naive:",lnZnaive) 


    # estimate Z via PFE (optimizing Estar)
    N = samples.nsamples
    lnZ,Err2lnZ,Estar,cutfrac = driver.run_PFE(cfg, samples)
    ErrlnZ = np.sqrt(Err2lnZ)
    lnZs = [lnZ]
    ErrlnZs = [ErrlnZ]
    Estars = [Estar]
    cutfracs = [cutfrac]
    log(fd, f"ln Z_est(_opt_) = {lnZ}   [ Err(lnZ) = {ErrlnZ}   E* = {Estar:.3f}  cut% = {100*cutfrac:.2f} ]")

    # remove the sampling outliers
    for threshold in cfg.thresholds:
        Estar = np.quantile(samples.energies, 1-threshold)
        # run PFE again with fixed Estar
        lnZ,Err2lnZ,Estar,cutfrac = driver.run_PFE(cfg, samples, Estar)
        ErrlnZ = np.sqrt(Err2lnZ)
        lnZs.append(lnZ)
        ErrlnZs.append(ErrlnZ)
        Estars.append(Estar)
        cutfracs.append(cutfrac)
        log(fd, f"ln Z_est({threshold:.3f}) = {lnZ}   [ Err(lnZ) = {ErrlnZ}   E* = {Estar:.3f}  cut% = {100*cutfrac:.2f} ]")

    fd.close()
    os.chdir(origdir)
    return (lnZs,ErrlnZs,Estars,cutfracs)


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

    # calculate analytic DoS and Z from it, if requested
    if cfg.getboolean('meta', 'dos'):
        lnZanados = driver.run_anados(cfg)
        if lnZanados is None:
            log(fd, f"analytic DoS not available for this potential function")
        else:
            log(fd, f"ln Z_anados = {lnZanados}")


    # run sampling in parallel
    pool = multiprocessing.Pool(nproc)
    data = pool.starmap(run_once, zip(itertools.repeat(cfg), range(nloop)), chunksize=1)

    # output PFE results
    thresholds = [None] + cfg.thresholds
    for i,threshold in enumerate(thresholds):
        lnZest    = [ lnZs[i]     for (lnZs,ErrlnZs,Estars,cutfracs) in data ]
        ErrlnZest = [ ErrlnZs[i]  for (lnZs,ErrlnZs,Estars,cutfracs) in data ]
        Estars    = [ Estars[i]   for (lnZs,ErrlnZs,Estars,cutfracs) in data ]
        cutfracs  = [ cutfracs[i] for (lnZs,ErrlnZs,Estars,cutfracs) in data ]
        if threshold is None:
            tag = '_opt_'
        else:
            tag = f"{threshold:.3f}"
        np.savetxt(f"lnZest_{tag}.dat", np.column_stack([lnZest,ErrlnZest]))
        log(fd, f"ln Z_est({tag}) = {np.mean(lnZest)} ± {np.std(lnZest)} ({np.mean(ErrlnZest)})")
        log(fd, f"ln Z_est({tag}) - ln Z_exact = {np.mean(lnZest)-lnZexact} ± {np.std(lnZest)}")
        log(fd, f"Estar({tag}) = {np.mean(Estars)} ± {np.std(Estars)}")
        log(fd, f"cut%({tag}) = {100*np.mean(cutfracs):.3f} ± {100*np.std(cutfracs):.3f}")

    log(fd)
    log(fd, f"program end: {timestamp()}")
    log(fd, "-" * 64)

  except AppError as e:

    sys.stderr.write(f"ERROR: {e.msg}\n")
    sys.exit(1)

