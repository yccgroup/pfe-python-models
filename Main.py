import sys
import os
import subprocess
import numpy as np

import config
import potentials
import sampling
import rafep
import driver
from trajectory import *


def usage():
    print(f"Usage: {sys.argv[0]} <inputfile>")


def error(msg, rc=1):
    sys.stdout.write(f"ERROR: {msg}\n")
    sys.exit(rc)


def log(msg=''):
    print(msg)


def get_git_revision():
    try:
        return subprocess.check_output(
            ['git', 'log', '--pretty=tformat:%h (%ai)'],
            cwd = os.path.dirname(os.path.realpath(__file__))
            ).decode('ascii').strip()
    except OSError:
        return 'N/A'


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
        error("TODO")
    else:
        error(f"unknown method '{method}'")
    samples.save()

    # Statistics
    log(f"Acceptance rate = {samples.acceptratio}")
    log(f"Average energy  = {np.mean(samples.energies)}")
    log(f"0.5 * kB * T    = {0.5/cfg.beta}")
    log()

    # estimate Z via RAFEP
    Z,lnZ = driver.run_RAFEP(cfg, samples)
    log(f"RAFEP Zest = {Z}")

    # remove the sampling outliers
    threshold = cfg.input.getfloat('rafep', 'threshold')
    trunc_traj, trunc_energies = rafep.truncate(samples.traj, samples.energies, threshold)
    trunc_samples = MCTrajectory(trunc_traj, trunc_energies, len(trunc_traj)/samples.nsamples)
    log(f"Truncated samples, threshold={threshold}, kept {100*trunc_samples.acceptratio}%")

    # run RAFEP again
    Z,lnZ = driver.run_RAFEP(cfg, trunc_samples)
    log(f"Model Zest = {Z}")

