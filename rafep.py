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
    # Calculate the partition function using RAFEP theory:
    # ln Z = ln Ω(E*) - ln <f>
    # where
    #   Ω(E*) = ∫ dE g(E) Θ(E* - E)
    #   f(E,E*) = exp(+βE) Θ(E* - E)
    # and we estimate the expectation value <f> by the sample average,
    #   <f> ≈ Σᵢ f(Eᵢ,E*) / N
    # while the configuration space volume Ω is estimated by a histogram method.
    #
    # Also calculate quantities for estimating the error.
    # The sample average contributes
    #   var(ln<f>) = ( <f²>/<f>² - 1 ) / N
    # and this becomes minimal at E* where
    #   βE* = ln 2 + ln <f²> - ln <f>
    # This can be used by the calling function to optimize E*.
    #
    # Passed in are the list of samples [Traj] and associated energy
    # values [Energy] which have already been truncated to below E*,
    # and the original (untruncated) trajectory length [N].

    # To avoid overflow, shift all energies to negative values.
    Emax = np.max(Energy)
    # Avg  = <f> / exp(β Emax)     => <f> = Avg * exp(β Emax)
    # Avg2 = <f²> / exp(2 β Emax)  => <f²> = Avg2 * exp(2 β Emax)
    Avg  = np.sum(np.exp(beta*(Energy-Emax))) / N
    Avg2 = np.sum(np.exp(2*beta*(Energy-Emax))) / N

    # calculate Ω via histogram (for general potentials)
    hist, bin_edges = np.histogram(Traj,bins=nbins)
    ndim = len(hist)
    Omega = 0
    for i in range(ndim):
        if hist[i] > 0:
            Omega += 1*(bin_edges[i+1]-bin_edges[i])

    lnZ = np.log(Omega) - np.log(Avg) - beta*Emax
    varlnZ = (Avg2/Avg**2 - 1) / N
    tryEstar = (np.log(2) + np.log(Avg2) - np.log(Avg))/beta + Emax

    return lnZ, varlnZ, tryEstar


# 2D RAFEP
def partfunc_RAFEP2D(Traj,Energy,N,beta,nbins=100):
    # same as 1D case above, except using a 2D histogram
    Emax = np.max(Energy)
    Avg  = np.sum(np.exp(beta*(Energy-Emax))) / N
    Avg2 = np.sum(np.exp(2*beta*(Energy-Emax))) / N

    X,Y = np.transpose(Traj)
    hist, xedges, yedges = np.histogram2d(X,Y,bins=nbins)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    Omega = np.count_nonzero(hist)*dx*dy

    lnZ = np.log(Omega) - np.log(Avg) - beta*Emax
    varlnZ = (Avg2/Avg**2 - 1) / N
    tryEstar = (np.log(2) + np.log(Avg2) - np.log(Avg))/beta + Emax

    return lnZ, varlnZ, tryEstar
