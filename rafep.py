# Monte Carlo sampling
# Letitia: 2023.08.01

import numpy as np

def truncate(Traj,Energy,Threshold):
    # truncate the trajectory and energy over the Threshold
    NewTraj = []
    NewEnergy = []
    for x, e in zip(Traj,Energy):
        if e < Threshold:
            NewTraj.append(x)
            NewEnergy.append(e)

    return NewTraj, NewEnergy


def autotruncate(Traj,Energy,Threshold,bothends=False):
    # truncate the trajectory and energy based on the probability threshold
    NewTraj = []
    NewEnergy = []

    # find energy thresholds
    Elower,Eupper = np.quantile(Energy, [Threshold,1-Threshold])
    if not bothends:
        Elower = -np.Inf

    # truncate the Threshold % of the trajectories
    for x,E in zip(Traj,Energy):
        if (E <= Eupper) and (E >= Elower):
            NewTraj.append(x)
            NewEnergy.append(E)

    return NewTraj, NewEnergy, Elower, Eupper
        

# 1D RAFEP
def partfunc_RAFEP1D(Traj,Energy,N,beta,nbins=100):
    # calculate the partition function using RAFEP theory
    Emax = np.max(Energy)
    Avg  = np.sum(np.exp(beta*(Energy-Emax)))/N * np.exp(beta*Emax)
    Avg2 = np.sum(np.exp(2*beta*(Energy-Emax)))/N * np.exp(2*beta*Emax)
    # calculate V0 via histogram (for general potentials)
    hist, bin_edges = np.histogram(Traj,bins=nbins)
    ndim = len(hist)
    V0 = 0
    for i in range(ndim):
        if hist[i] > 0:
            V0 += 1*(bin_edges[i+1]-bin_edges[i])

    Z   = V0/Avg
    lnZ = np.log(V0) - np.log(np.sum(np.exp(beta*(Energy-Emax)))/N) - beta*Emax
    varlnZ = (Avg2/Avg**2 - 1)/N
    ratio = Avg2/Avg

    return Z, lnZ, varlnZ, ratio


# 2D RAFEP
def partfunc_RAFEP2D(Traj,Energy,N,beta,nbins=100):
    # calculate the partition function using RAFEP theory
    Emax = np.max(Energy)
    Avg = np.sum(np.exp(beta*(Energy-Emax)))/N * np.exp(beta*Emax)
    Avg2 = np.sum(np.exp(2*beta*(Energy-Emax)))/N * np.exp(2*beta*Emax)
    # calculate V0 via histogram (for general potentials)
    X,Y = np.transpose(Traj)
    hist, xedges, yedges = np.histogram2d(X,Y,bins=nbins)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    V0  = np.count_nonzero(hist)*dx*dy
    Z   = V0/Avg
    lnZ = np.log(V0) - np.log(np.sum(np.exp(beta*(Energy-Emax)))/N) - beta*Emax
    varlnZ = (Avg2/Avg**2 - 1)/N
    ratio = Avg2/Avg

    return Z, lnZ, varlnZ, ratio

