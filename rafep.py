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


def autotruncate(Traj,Energy,cutoff=0.001,nbins=21):
    # automatically truncate the trajectory and energy when probability is too small
    NewTraj = []
    NewEnergy = []

    # output the energy histogram (for later debug)
    Eupper = int(np.max(Energy))+1
    Elower = int(np.min(Energy))
    counts, Evalues = np.histogram(Energy,range=(Elower,Eupper+1),bins=nbins)
    outdata = np.column_stack((Evalues[0:-1],counts))
    np.savetxt("histogram.dat",outdata,fmt="%.10f %d")
    
    # find most probable energy (not avg) and the associated cutoff
    mpindex = np.argmax(counts)
    work = counts-cutoff*counts[mpindex]
    major = []
    for i in range(len(work)):
        if work[i] >= 0.0:
            major.append(i)
    
    Elower_cutoff = Evalues[major[0]]
    Eupper_cutoff = Evalues[major[-1]]
    
    # truncate the energy and trajectory
    for x, e in zip(Traj,Energy):
        if e >= Elower_cutoff and e <= Eupper_cutoff:
            NewTraj.append(x)
            NewEnergy.append(e)
    
    return NewTraj, NewEnergy, Elower_cutoff, Eupper_cutoff
        

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

