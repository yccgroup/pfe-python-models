# Monte Carlo sampling
# Letitia: 2023.08.01

import numpy as np
from itertools import cycle
from matplotlib import pyplot as plt 
from potentials import *
from sampling import *
from rafep import *

np.random.seed()

def REMetropolis(X1,E1,X2,E2,beta1,beta2):
    # acceptance = min {1, Pnew/Pold}

    # calculate the effective exponent
    exponent = (beta2-beta1) * (E2-E1) 

    # accept if exponent is larger than zero
    if exponent >= 0:
        return X2, X1
    
    # othereise, accept according to the acceptance
    Pratio = np.exp(exponent)
    rand = np.random.rand()
    if rand <= Pratio:
        return X2, X1
    else:
        return X1, X2


# Main function

# General MC parameters and initialization
nsteps = 100000
stepsize = 0.1
outfreq = 1
Temperature = 300
beta = 1.0/(0.001987191686485529*Temperature)   # energy unit kcal/mol

# RAFEP parameters
Threshold = 4

# Potential parameters and initialization
xinit = 0.0
#POT = Harmonic1D(300.0, 0.0)
POT = DW1D(100,200,100)
#POT = ASymDW1D(4.0, 0.0, 3.0)  # pos/height of 2nd minimum, barrier height

# Replica Exchange parameters
nreplica = 10
Tmax = 1000
Tmin = Temperature
exfreq = 1000

# Current REMC can only support even number of replicas
if nreplica%2 != 0:
    nreplica += 1

# Initialize the replica parameters
nblock = round(nsteps/exfreq)
nsteps = exfreq
npair  = round(nreplica/2)
SuperTraj = [[] for i in range(nreplica)]
SuperEnergy = [[] for i in range(nreplica)]
SuperAccept = [0 for i in range(nreplica)]
SuperXinit = [xinit for i in range(nreplica)]
SuperEinit = [POT(xinit) for i in range(nreplica)]
SuperTemp = [(Tmin+(Tmax-Tmin)*i/(nreplica-1)) for i in range(nreplica)]
Superbeta = [1.0/(0.001987191686485529*SuperTemp[i]) for i in range(nreplica)]

print("Temperature:",SuperTemp)

# Replica Exchange Monte Carlo Sampling
for i in range(nblock):
    # sample each replica by the length of a block
    for j in range(nreplica):
        Traj, Energy, Accept = MCSampling1D(POT, SuperXinit[j], nsteps, stepsize, outfreq, Superbeta[j])
     
        # append data
        ndim = len(Traj)
        for k in range(ndim):
            SuperTraj[j].append(Traj[k])
            SuperEnergy[j].append(Energy[k])

        SuperAccept[j] += Accept
        SuperXinit[j] = Traj[-1]
        SuperEinit[j] = Energy[-1]

    # replica exchange
    if i%2 == 0:
        # exchange with the next (-->)
        PoolX = cycle(SuperXinit)
        PoolE = cycle(SuperEinit)
        PoolB = cycle(Superbeta)
        WorkX = []
        for j in range(npair):
            X1 = next(PoolX)
            X2 = next(PoolX)
            E1 = next(PoolE)
            E2 = next(PoolE)
            beta1 = next(PoolB)
            beta2 = next(PoolB)
            X1, X2 = REMetropolis(X1,E1,X2,E2,beta1,beta2)
            WorkX.append(X1)
            WorkX.append(X2)

        SuperXinit = WorkX
    else:
        # exchange with the previous (<--)
        PoolX = cycle(SuperXinit)
        PoolE = cycle(SuperEinit)
        PoolB = cycle(Superbeta)
        WorkX = []
        temp = next(PoolX)
        temp = next(PoolE)
        temp = next(PoolB)
        WorkX.append(0)
        for j in range(npair):
            X1 = next(PoolX)
            X2 = next(PoolX)
            E1 = next(PoolE)
            E2 = next(PoolE)
            beta1 = next(PoolB)
            beta2 = next(PoolB)
            X1, X2 = REMetropolis(X1,E1,X2,E2,beta1,beta2)
            WorkX.append(X1)
            WorkX.append(X2)

        SuperXinit = WorkX[0:-1]
        SuperXinit[0] = WorkX[-1]


# Statistics (data from the lowest temperature)
nsteps = nsteps * nblock
Traj = SuperTraj[0]
Energy = SuperEnergy[0]
Accept = SuperAccept[0]
beta = Superbeta[0]

np.savetxt("Traj.dat", Traj)
np.savetxt("Energy.dat", Energy)

Accept = Accept / nsteps
print("Acceptance rate =", Accept)
print("Average energy =",np.mean(Energy))
print("Expectation of energy =",0.5/beta)
Zest, lnZest = Partition_RAFEP1D(Traj,Energy,beta)
print("Zest:",Zest)

# Remove the sampling outlier and apply RAFEP again
Threshold = Threshold/beta + np.min(Energy)
NewTraj, NewEnergy =  Truncate(Traj,Energy,Threshold)
Zest, lnZest = Partition_RAFEP1D(NewTraj,NewEnergy,beta)
print("Model Zest:",Zest)


