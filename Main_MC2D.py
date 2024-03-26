# Monte Carlo sampling
# Letitia: 2023.08.01

import numpy as np
from matplotlib import pyplot as plt 
from potentials import *
from sampling import *
from rafep import *

np.random.seed()

# Main function

# General MC parameters and initialization
nsteps = 1000000
stepsize = 0.1
outfreq = 1
Temperature = 300
beta = 1.0/(0.001987191686485529*Temperature)   # energy unit kcal/mol

# RAFEP parameters
Threshold = 4

# Potential parameters and initialization
xinit = [-0.6, 1.5]
POT = MuellerBrown()

# Monte Carlo sampling
Traj, Energy, Accept = MCSampling2D(POT, xinit, nsteps, stepsize, outfreq, beta)
np.savetxt("Traj.dat", Traj)
np.savetxt("Energy.dat", Energy)

# Statistics
Accept = Accept / nsteps
print("Acceptance rate =", Accept)
print("Average energy =",np.mean(Energy))
print("Expectation of energy =",0.5/beta)
Zest, lnZest = Partition_RAFEP2D(Traj,Energy,beta)
print("Zest:",Zest)

# Remove the sampling outlier and apply RAFEP again
Threshold = Threshold/beta + np.min(Energy)
NewTraj, NewEnergy =  Truncate(Traj,Energy,Threshold)
Zest, lnZest = Partition_RAFEP2D(NewTraj,NewEnergy,beta)
print("Model Zest:",Zest)

