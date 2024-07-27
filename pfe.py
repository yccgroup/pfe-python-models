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
        

# PFE by (optionally) iteratively solving E*
def partfunc(Traj,Energy,beta,omegafunc,nbins=100,Estar=None):
    # Calculate the partition function using PFE theory:
    # ln Z = ln Ω(E*) - ln <f>
    # where
    #   Ω(E*) = ∫ dE g(E) Θ(E* - E)
    #   f(E,E*) = exp(+βE) Θ(E* - E)
    # and we estimate the expectation value <f> by the sample average,
    #   <f> ≈ Σᵢ f(Eᵢ,E*) / N
    # while the configuration space volume Ω is estimated by a histogram method.
    #
    # Also calculate quantities for estimating the error.
    # The sample average contributes to square of the standard error
    #   Error² = var(ln<f>) / N = ( <f²>/<f>² - 1 ) / N
    # and this becomes minimal at E* where
    #   βE* = ln 2 + ln <f²> - ln <f>
    # This can be used by the calling function to optimize E*.
    #
    # Passed in are the list of samples [Traj] and associated energy
    # values [Energy].

    # To avoid overflow, shift all energies to negative values.
    N = len(Energy)
    Emax = np.max(Energy)
    Func = np.exp(beta*(Energy-Emax))
    Func2= np.exp(2*beta*(Energy-Emax))

    # Check if Estar is given or should be calculated.
    if Estar is None:

        # Iteratively solve the best value of E*
        Estar_prev = np.quantile(Energy, 0.99)
        Estar = Estar_prev
        difference = 1
        while (difference > 1e-6):
            # Generate the Heaviside function
            H = np.ones(N)
            for i in range(N):
                if Energy[i] > Estar:
                    H[i] = 0
            # Calculate the Avg and Avg2 (Energy shifted)
            Avg = np.mean(Func*H)
            Avg2 = np.mean(Func2*H)
            # Update the E*, Error2, difference
            Estar = (np.log(2) + np.log(Avg2) - np.log(Avg)) / beta + Emax
            Err2 = (Avg2/Avg**2 - 1) / N
            difference = np.abs(Estar - Estar_prev)
            # For next iteration
            Estar_prev = Estar

    # Recalculate after Estar is found.
    H = np.ones(N)
    for i in range(N):
        if Energy[i] > Estar:
            H[i] = 0
    # Calculate the Avg, Avg2 and Error2 (Energy shifted)
    Avg = np.mean(Func*H)
    Avg2 = np.mean(Func2*H)
    Err2 = (Avg2/Avg**2 - 1) / N
 
    # Truncate the trajectory based on Estar
    Trajstar = []
    for i in range(N):
        if Energy[i] <= Estar:
            Trajstar.append(Traj[i])

    # Calculate Ω via histogram
    Omega = omegafunc(Trajstar,nbins)

    # Calculate the configurational partition function
    lnZ = np.log(Omega) - np.log(Avg) - beta*Emax

    #print("lnZ:",lnZ,"Err:",np.sqrt(Err2),"E*:",Estar)

    return lnZ, Err2, Estar


# 1D Omega calculation
def omega1D(Traj,nbins):
    # Calculate Ω via histogram
    hist, bin_edges = np.histogram(Traj,bins=nbins)
    ndim = len(hist)
    Omega = 0
    for i in range(ndim):
        if hist[i] > 0:
            Omega += 1*(bin_edges[i+1]-bin_edges[i])
    return Omega


# 2D Omega calculation
def omega2D(Traj,nbins):
    # Calculate Ω via histogram
    X,Y = np.transpose(Traj)
    hist, xedges, yedges = np.histogram2d(X,Y,bins=nbins)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    Omega = np.count_nonzero(hist)*dx*dy
    return Omega
