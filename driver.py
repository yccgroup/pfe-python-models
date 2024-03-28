from itertools import cycle

import potentials
import sampling
import rafep
from util import error
from trajectory import *


def run_MC(cfg):
    nsteps   = cfg.getint('trajectory', 'nsteps')
    stepsize = cfg.getfloat('trajectory', 'stepsize')
    outfreq  = cfg.getint('trajectory', 'outfreq')

    if isinstance(cfg.pot, potentials.PotentialFunction1D):
        sampling_func = sampling.MCSampling1D
    elif isinstance(cfg.pot, potentials.PotentialFunction2D):
        sampling_func = sampling.MCSampling2D
    else:
        error(f"function '{cfg.pot.__name__}' must be a PotentialFunction1D/2D")

    xinit = cfg.pot.xinit()
    samples = sampling_func(cfg.pot, xinit, nsteps, stepsize, outfreq, cfg.beta)
    return samples


def run_REMC(cfg):
    nsteps   = cfg.getint('trajectory', 'nsteps')
    stepsize = cfg.getfloat('trajectory', 'stepsize')
    outfreq  = cfg.getint('trajectory', 'outfreq')
    nreplica = cfg.getint('trajectory', 'nreplica')
    kB       = cfg.getfloat('trajectory', 'kB')
    temp_min = cfg.getfloat('trajectory', 'temp')
    temp_max = cfg.getfloat('trajectory', 'temp-max')
    exfreq   = cfg.getint('trajectory', 'exfreq')

    # current REMC can only support even number of replicas
    if nreplica % 2 != 0:
        error("nreplica must be even")
    if nsteps % exfreq != 0:
        error("nsteps must be a multiple of exfreq")

    if isinstance(cfg.pot, potentials.PotentialFunction1D):
        sampling_func = sampling.MCSampling1D
    elif isinstance(cfg.pot, potentials.PotentialFunction2D):
        sampling_func = sampling.MCSampling2D
    else:
        error(f"function '{cfg.pot.__name__}' must be a PotentialFunction1D/2D")

    # initialize the replicas
    xinit  = cfg.pot.xinit()
    nblock = nsteps // exfreq
    npair  = nreplica // 2
    s_samples = [ MCTrajectory() for i in range(nreplica) ]
    s_xinit   = [ xinit for i in range(nreplica) ]
    s_Einit   = [ cfg.pot(xinit) for i in range(nreplica) ]
    s_temp    = np.linspace(temp_min, temp_max, nreplica)
    s_beta    = 1.0 / (kB * s_temp)

    for i in range(nblock):
        # sample each replica by the length of a block
        for j in range(nreplica):
            samples = sampling_func(cfg.pot, s_xinit[j], exfreq, stepsize, outfreq, s_beta[j])
            # append data
            s_samples[j].extend(samples)
            # store final sample for exchange check
            s_xinit[j] = samples.traj[-1]
            s_Einit[j] = samples.energies[-1]

        # replica exchange
        pool = cycle(zip(s_xinit, s_Einit, s_beta))
        tmp_x = []
        if i%2 == 0:
            # exchange with the next (-->)
            for k in range(npair):
                x1,E1,beta1 = next(pool)
                x2,E2,beta2 = next(pool)
                x1,x2 = sampling.REMetropolis(x1,E1,x2,E2,beta1,beta2)
                tmp_x.append(x1)
                tmp_x.append(x2)
            # apply the new (exchanged) configurations
            s_xinit = tmp_x
        else:
            # exchange with the previous (<--)
            skipped = next(pool)  # jump over first one
            tmp_x.append(None)    # dummy value, not used
            for j in range(npair):
                x1,E1,beta1 = next(pool)
                x2,E2,beta2 = next(pool)
                x1,x2 = sampling.REMetropolis(x1,E1,x2,E2,beta1,beta2)
                tmp_x.append(x1)
                tmp_x.append(x2)
            # apply the new (exchanged) configurations
            tmp_x[0] = tmp_x[-1]
            s_xinit = tmp_x[0:-1]

    # return data from the lowest temperature
    return s_samples[0]


def run_RAFEP(cfg, samples):
    if isinstance(cfg.pot, potentials.PotentialFunction1D):
        rafep_func = rafep.partfunc_RAFEP1D
    elif isinstance(cfg.pot, potentials.PotentialFunction2D):
        rafep_func = rafep.partfunc_RAFEP2D
    Z,lnZ = rafep_func(samples.traj, samples.energies, cfg.beta)
    return Z


def run_exact(cfg):
    if hasattr(cfg.pot, 'partfunc_exact'):
        Z, lnZ = cfg.pot.partfunc_exact(cfg.beta)
        return Z
    else:
        return None


def run_integral(cfg):
    lo = cfg.getfloat('integral', 'lower')
    hi = cfg.getfloat('integral', 'upper')
    dx = cfg.getfloat('integral', 'dx')
    Zint, lnZint = cfg.pot.partfunc_integral(hi, lo, dx, cfg.beta)
    return Zint

