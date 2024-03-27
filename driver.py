import potentials
import sampling
import rafep


def run_MC(cfg):
    nsteps      = cfg.input.getint('trajectory', 'nsteps')
    stepsize    = cfg.input.getfloat('trajectory', 'stepsize')
    outfreq     = cfg.input.getint('trajectory', 'outfreq')

    if isinstance(cfg.pot, potentials.PotentialFunction1D):
        sampling_func = sampling.MCSampling1D
    elif isinstance(cfg.pot, potentials.PotentialFunction2D):
        sampling_func = sampling.MCSampling2D

    xinit = cfg.pot.xinit()
    samples = sampling_func(cfg.pot, xinit, nsteps, stepsize, outfreq, cfg.beta)
    return samples


def run_RAFEP(cfg, samples):
    if isinstance(cfg.pot, potentials.PotentialFunction1D):
        rafep_func = rafep.PartFunc_RAFEP1D
    elif isinstance(cfg.pot, potentials.PotentialFunction2D):
        rafep_func = rafep.PartFunc_RAFEP2D
    Z,lnZ = rafep_func(samples.traj, samples.energies, cfg.beta)
    return Z,lnZ

