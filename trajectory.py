import numpy as np

class MCTrajectory:
    def __init__(self, traj, energies, acceptratio):
        assert len(traj)==len(energies)
        self.traj = traj
        self.energies = energies
        self.nsamples = len(traj)
        self.acceptratio = acceptratio
    
    def save(self):
        np.savetxt("Traj.dat", self.traj)
        np.savetxt("Energy.dat", self.energies)

