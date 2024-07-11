import numpy as np
from util import error


class MCTrajectory:
    def __init__(self, traj=None, energies=None, naccept=0):
        if traj is None: traj=[]
        if energies is None: energies=[]
        assert len(traj)==len(energies)
        self.traj = traj
        self.energies = energies
        self.nsamples = len(traj)
        self.naccept  = naccept
    
    def save(self):
        np.savetxt("Traj.dat", self.traj)
        np.savetxt("Energy.dat", self.energies)

    def extend(self, other):
        assert len(other.traj)==len(other.energies)
        self.traj.extend(other.traj)
        self.energies.extend(other.energies)
        self.nsamples += other.nsamples
        self.naccept += other.naccept

    def truncate(self, nsamples):
        if nsamples <= self.nsamples:
            self.traj = self.traj[:nsamples]
            self.energies = self.energies[:nsamples]
            self.naccept = int(self.naccept * nsamples / self.nsamples)
            self.nsamples = nsamples
        else:
            error(f"cannot truncate trajectory with {self.nsamples} samples to {nsamples} samples")

    def energy_min(self):
        return np.min(self.energies)

    def energy_mean(self):
        return np.mean(self.energies)
