# Monte Carlo sampling
# Letitia: 2023.08.01

import numpy as np
from copy import deepcopy
from trajectory import *


def Metropolis(Xold,Eold,Xnew,Enew,beta):
    # acceptance = min {1, Pnew/Pold}

    # Enew <= Eold, accept
    if Enew <= Eold:
        return Xnew, Enew, 1

    # otherwise, accept according to the acceptance
    Pratio = np.exp(-beta*(Enew-Eold))
    rand = np.random.rand()
    if rand <= Pratio:
        return Xnew, Enew, 1
    else:
        return Xold, Eold, 0


def REMetropolis(X1,E1,X2,E2,beta1,beta2):
    # acceptance = min {1, Pnew/Pold}

    # calculate the effective exponent
    exponent = (beta2-beta1) * (E2-E1)

    # accept if exponent is larger than zero
    if exponent >= 0:
        return X2, X1

    # otherwise, accept according to the acceptance
    Pratio = np.exp(exponent)
    rand = np.random.rand()
    if rand <= Pratio:
        return X2, X1
    else:
        return X1, X2


def Move(x,stepsize):
    # move one step (no PBC)
    step = 2*(np.random.rand()-0.5)*stepsize
    return x + step


def MCSampling1D(potfunc, xinit, nsteps, stepsize, outfreq, beta):
    # Monte Carlo Sampling
    Energy = []
    Traj = []
    Accept = 0
    Xold = xinit
    Eold = potfunc(Xold)

    # Propagation
    for i in range(nsteps):
        Xnew = Move(Xold,stepsize)
        Enew = potfunc(Xnew)
        Xnew, Enew, Acceptstep = Metropolis(Xold,Eold,Xnew,Enew,beta)
        # collect for acceptance rate
        Accept += Acceptstep
        # set Xold and Eold for next step
        Xold = Xnew
        Eold = Enew
        # collect sampling data
        if i%outfreq == 0:
            Energy.append(Enew)
            Traj.append(Xnew)
            ## set Xold and Eold for next step
            #Xold = Xnew
            #Eold = Enew

    return MCTrajectory(Traj, Energy, Accept)


def MCSampling2D(potfunc, xinit, nsteps, stepsize, outfreq, beta):
    # Monte Carlo Sampling. Note that xinit, Xold, Xnew are 2D now.
    Energy = []
    Traj = []
    Accept = 0
    Xold = xinit
    Eold = potfunc(Xold)

    # Propagation
    for i in range(nsteps):
        Xnew = np.zeros(2)   # crucial: this is to allocate a new memory space to decouple from the previous Xold
        Xnew[0] = Move(Xold[0],stepsize)
        Xnew[1] = Move(Xold[1],stepsize)
        Enew = potfunc(Xnew)
        Xnew, Enew, Acceptstep = Metropolis(Xold,Eold,Xnew,Enew,beta)
        # collect for acceptance rate
        Accept += Acceptstep
        # set Xold and Eold for next step
        Xold = Xnew
        Eold = Enew
        # collect sampling data
        if i%outfreq == 0:
            Energy.append(Enew)
            Traj.append(Xnew)
            ## set Xold and Eold for next step
            #Xold = Xnew
            #Eold = Enew

    return MCTrajectory(Traj, Energy, Accept)

