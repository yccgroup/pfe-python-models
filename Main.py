import sys
import numpy as np

import config
import potentials
import sampling
import rafep
import driver
from util import log, error
from trajectory import *


def usage():
    print(f"Usage: {sys.argv[0]} <inputfile>")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    # read the configuration file
    inputfile = sys.argv[1]
    cfg = config.Config(inputfile)

    # set the randoms seed
    try:
        seed = cfg.input.getint('trajectory', 'seed')
        np.random.seed(seed)
    except ValueError:
        np.random.seed()

    # generate the samples, depending on the selected method
    method = cfg.input.get('trajectory', 'method')
    log(f"Generating samples via {method}...")
    if method == 'monte-carlo':
        samples = driver.run_MC(cfg)
    elif method == 'replica-exchange':
        samples = driver.run_REMC(cfg)
    else:
        error(f"unknown method '{method}'")
    samples.save()

    # Statistics
    nsteps = cfg.input.getint('trajectory', 'nsteps')
    log(f"Acceptance rate = {samples.naccept/nsteps}")
    log(f"Average energy  = {samples.energy_mean()}")
    log(f"Minimum energy  = {samples.energy_min()}")
    log(f"0.5 * kB * T    = {0.5/cfg.beta}")
    log()

    # estimate Z via RAFEP
    Z,lnZ = driver.run_RAFEP(cfg, samples)
    log(f"RAFEP Zest = {Z}")

    # remove the sampling outliers
    threshold = cfg.input.getfloat('rafep', 'threshold')
    E_cut = threshold/cfg.beta + samples.energy_min()
    trunc_traj, trunc_energies = rafep.truncate(samples.traj, samples.energies, E_cut)
    trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj))
    log(f"Truncated samples, threshold={threshold}, kept {100*trunc_samples.nsamples/samples.nsamples:.3f}%")

    # run RAFEP again
    Z,lnZ = driver.run_RAFEP(cfg, trunc_samples)
    log(f"Model Zest = {Z}")

