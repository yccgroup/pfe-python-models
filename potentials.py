import numpy as np
from scipy import optimize

# 1D potentials from here

class PotentialFunction1D:
    # abstract base class, no __init__
    def partfunc_integral(self,upper,lower,dx,beta):
        # calculate the partition function via integration
        ndim = round((upper-lower)/dx)
        Energy = []
        for i in range(ndim):
            x = lower + i*dx
            Energy.append(self(x))

        Emin = np.min(Energy)

        Z   = np.sum(np.exp(-beta*(Energy-Emin)))*dx*np.exp(-beta*Emin)
        lnZ = np.log(np.sum(np.exp(-beta*(Energy-Emin)))*dx) - beta*Emin

        return Z, lnZ

    def calcpot(self,upper,lower,dx,beta):
        # calculate the potential curve
        ndim = round((upper-lower)/dx)
        Data = []
        for i in range(ndim):
            x = lower + i*dx
            Data.append([x,self(x)*beta])

        return Data


class Harmonic1D(PotentialFunction1D):
    # 1D harmonic oscillator potential
    def __init__(self,k,x0):
        self.k = k
        self.x0 = x0

    def __call__(self,x):
        return 0.5 * self.k * (x-self.x0)**2

    def partfunc_exact(self,beta):
        # calculate the exact partition function (available only in harmonic oscillator potential)
        Z = np.sqrt(2*np.pi/(self.k*beta))
        lnZ = np.log(Z)
        return Z, lnZ

    def xinit(self):
        return self.x0


class DW1D(PotentialFunction1D):
    # 1D double well potential
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self,x):
        return self.a * x**2 - self.b * x**3 + self.c * x**4

    def xinit(self):
        return 0.0


class SymDW1D(PotentialFunction1D):
    # symmetric 1D double well potential
    # minima are at 0 and x0, barrier height is h
    def __init__(self,x0,h):
        self.x0 = x0
        self.h = h
        self.c = 16*h/x0**4

    def __call__(self,x):
        return self.c * x**2 * (x-self.x0)**2

    def xinit(self):
        return 0.0


class ASymDW1D(PotentialFunction1D):
    # asymmetric 1D double well potential
    # minima are at x=0,y=0 and x=x0,y=v
    # maximum is at y=h
    # --> find a,b,c numerically
    def __init__(self,x0,v,h):
        self.find_params(x0,v,h)
        self.a = self.sol.x[1]
        self.b = self.sol.x[2]
        self.c = self.sol.x[3]

    def find_params(self,x0,v,h):
        # sanity checks
        if h<v:
            raise ValueError(f"barrier h={h} lower than 2nd minimum v={v}")
        if h<0:
            raise ValueError(f"barrier h={h} lower than 1st minimum (0)")
        # the potential itself and its derivative
        def pot(x,a,b,c):
            return a * x**2 - b * x**3 + c * x**4
        def pot1(x,a,b,c):
            return 2*a * x - 3*b * x**2 + 4*c * x**3
        # parameters to optimize:
        # p[0] = position of maximum
        # p[1] = a
        # p[2] = b
        # p[3] = c
        def fun(p):
            return [ pot(p[0], p[1], p[2], p[3]) - h,
                     pot1(p[0], p[1], p[2], p[3]),
                     pot(x0, p[1], p[2], p[3]) - v,
                     pot1(x0, p[1], p[2], p[3]) ]
        # initial guess
        init = [ x0/2, 1.0, 1.0, 1.0 ]
        # solve
        sol = optimize.root(fun, init)
        # check
        if not sol.success:
            raise ValueError(f"failed to find parameters: {sol.message}")
        self.sol = sol

    def __call__(self,x):
        return self.a * x**2 - self.b * x**3 + self.c * x**4

    def xinit(self):
        return 0.0


# 2D potentials from here

class PotentialFunction2D:
    # abstract base class, no __init__
    def partfunc_integral(self,upper,lower,dx,beta):
        # calculate the partition function via integration
        ndim = round((upper-lower)/dx)
        Energy = []
        for i in range(ndim):
            x = lower + i*dx
            for j in range(ndim):
                y = lower + j*dx
                Energy.append(self([x,y]))

        Emin = np.min(Energy)
        Z   = np.sum(np.exp(-beta*(Energy-Emin)))*dx*dx*np.exp(-beta*Emin)
        lnZ = np.log(np.sum(np.exp(-beta*(Energy-Emin)))*dx*dx) - beta*Emin

        return Z, lnZ

    def calcpot(self,upper,lower,dx,beta): 
        # calculate the potential surface
        ndim = round((upper-lower)/dx)
        Data = []
        for i in range(ndim):
            x = lower + i*dx
            for j in range(ndim):
                y = lower + j*dx
                Data.append([x,y,self([x,y])*beta])

        return Data 


class MuellerBrown(PotentialFunction2D):
    # 2D Mueller Brown potential
    def __init__(self):
        self.A = [-200, -100, -170, 15]
        self.a = [-1, -1, -6.5, 0.7]
        self.b = [0, 0, 11, 0.6]
        self.c = [-10, -10, -6.5, 0.7]
        self.x0 = [1, 0, -0.5, -1]
        self.y0 = [0, 0.5, 1.5, 1]

    def __call__(self,X):
        V = 146.69951720995402  # so that minimum is zero
        x = X[0]
        y = X[1]
        for i in range(4):
            V += self.A[i]*np.exp(self.a[i]*(x-self.x0[i])**2+self.b[i]*(x-self.x0[i])*(y-self.y0[i])+self.c[i]*(y-self.y0[i])**2)
        #return 0.05*V
        return V

    def xinit(self):
        return [-0.6, 1.5]


# Main program for printing out the potential
if __name__ == '__main__':

    import config
    import sys
    import os
    import argparse
    
    from util import *

    try:
        # parse arguments
        p = argparse.ArgumentParser(description="Generate samples and estimate Z.")
        p.add_argument('inputfile',
            help='name of the input file')
        p.add_argument('--param', '-p', metavar='P', action='append', default=[],
            help='override parameter from the input file; syntax: section.setting=value')
        args = p.parse_args()

        # read the configuration file, and apply command line settings
        cfg = config.Config(args.inputfile, args.param)

        # read meta parameters
        nproc = cfg.getint('meta', 'nproc')
        nloop = cfg.getint('meta', 'nloop')

        # generate output
        fd = sys.stdout
        log(fd, "Model potentials")
        log(fd, "-" * 64)
        log(fd, f"git-commit: {get_git_revision()}")
        log(fd, f"program start: {timestamp()}")
        log(fd)
        log(fd, "-------- START INPUT --------")
        cfg.write(fd)
        log(fd, "-------- END INPUT --------")
        log(fd)

        # print out the potential
        lo = cfg.getfloat('integral', 'lower')
        hi = cfg.getfloat('integral', 'upper')
        dx = cfg.getfloat('integral', 'dx')

        Data = cfg.pot.calcpot(hi, lo, dx, cfg.beta)
        np.savetxt("pot.dat",Data)
    
    except AppError as e:

        sys.stderr.write(f"ERROR: {e.msg}\n")
        sys.exit(1)

