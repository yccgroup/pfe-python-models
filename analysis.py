# Letitia: Analize the Zest data, 2023.08.02

import numpy as np
from potentials import *

# General parameters and initialization
Temperature = 300
beta = 1.0/(0.001987191686485529*Temperature)   # energy unit kcal/mol

# Potential parameters and initialization
#xinit = [-0.6, 1.5]
#POT = MuellerBrown()
POT = Harmonic1D(300.0, 0.0)
#POT = DW1D(100,200,100)
#POT = ASymDW1D(4.0, 0.0, 3.0)  # pos/height of 2nd minimum, barrier height

# Integrate the partition function numerically
Zint, lnZint = POT.partition_integral(5,-5,0.005,beta)
Zexact = Zint

# Read the output files
data = np.loadtxt("Zest_MC.dat")
modeldata = np.loadtxt("Model_Zest_MC.dat")

# Process output from Main_MC.py
print("MC Zest avg, std =",np.mean(data),np.std(data))
print("MC Zest/Zexact avg, std =",np.mean(data)/Zexact,np.std(data)/Zexact)
print("MC Model Zest avg, std =",np.mean(modeldata),np.std(modeldata))
print("MC Model Zest/Zexact avg, std =",np.mean(modeldata)/Zexact,np.std(modeldata)/Zexact)

# Read the output files
data = np.loadtxt("Zest_REMC.dat")
modeldata = np.loadtxt("Model_Zest_REMC.dat")

# Process output from Main_REMC.py
print("REMC Zest avg, std =",np.mean(data),np.std(data))
print("REMC Zest/Zexact avg, std =",np.mean(data)/Zexact,np.std(data)/Zexact)
print("REMC Model Zest avg, std =",np.mean(modeldata),np.std(modeldata))
print("REMC Model Zest/Zexact avg, std =",np.mean(modeldata)/Zexact,np.std(modeldata)/Zexact)
