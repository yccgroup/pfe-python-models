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

    # output the energy histogram for analysis 
    counts, Evalues = np.histogram(Energy,range=(int(np.min(Energy)),int(np.max(Energy))+2),bins=21)
    outdata = np.column_stack((Evalues[0:-1],counts))
    np.savetxt("histogram.dat",outdata,fmt="%.10f %d")
    
    return NewTraj, NewEnergy, Elower, Eupper
        

# 1D RAFEP
def partfunc_RAFEP1D(Traj,Energy,beta):
    # calculate the partition function using RAFEP theory
    Emax = np.max(Energy)
    Avg = np.mean(np.exp(beta*(Energy-Emax))) * np.exp(beta*Emax)
    # calculate V0 via histogram (for general potentials)
    hist, bin_edges = np.histogram(Traj,bins=100)
    ndim = len(hist)
    V0 = 0
    for i in range(ndim):
        if hist[i] > 0:
            V0 += 1*(bin_edges[i+1]-bin_edges[i])

    Z   = V0/Avg
    lnZ = np.log(V0) - np.log(np.mean(np.exp(beta*(Energy-Emax)))) - beta*Emax

    return Z, lnZ


# 2D RAFEP
def partfunc_RAFEP2D(Traj,Energy,beta):
    # calculate the partition function using RAFEP theory
    Emax = np.max(Energy)
    Avg = np.mean(np.exp(beta*(Energy-Emax))) * np.exp(beta*Emax)
    # calculate V0 via histogram (for general potentials)
    X,Y = np.transpose(Traj)
    hist, xedges, yedges = np.histogram2d(X,Y,bins=100)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    V0  = np.count_nonzero(hist)*dx*dy
    Z   = V0/Avg
    lnZ = np.log(V0) - np.log(np.mean(np.exp(beta*(Energy-Emax)))) - beta*Emax

    return Z, lnZ

