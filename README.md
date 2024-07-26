# PFE Python Models

This code is a sample implementation of the PFE method for simple
1D or 2D model potentials.  The code performs the following:

- Collect a number of samples from the canonical distribution of
  the given potential function, via Monte-Carlo or Replica-Exchange.
- Estimate the value of the partition function Z (at the given
  temperature) from the collected sample, via the PFE method.
- To remove the effect of sampling outliers, multiple values of
  the PFE energy threshold can be specified.
- The sampling and PFE estimation is performed multiple times,
  in order to estimate the error of the PFE estimate.
- The actual value of Z is also calculated analytically (where
  possible) and numerically, so that one can verify the correctness
  of the PFE result.


## Prerequisites

The code is written in Python 3 and depends on the packages `numpy`
and `scipy`.  If your Python install does not include them, you can
create a conda environment and install them.

install:

    conda create -n py3
    conda activate py3
    conda install python=3 'numpy<2' scipy


when using:

    conda activate py3

when done:

    conda deactivate

## Running the Code

To run the code, an input file is needed (see below). Then the
code can be run via

    ./main.py input.conf

The results are printed to standard output, therefore it is recommended
to redirect it so that you have a record, e.g. via

    ./main.py input.conf | tee output.log

It is possible to override parameters from the input file by adding
the `--param` or `-p` option (which can be given multiple times).
For example, to change the number of sampling loops to 1000, use

    ./main.py --param meta.nloop=1000 input.conf


## Input File

The input file specifies all parameters and meta-parameters for the
calculations. It is organized into sections, which are documented below.

Some documented sample input files can be found in the [input](./input/)
directory.

### `[meta]` section

* `nloop` is the number of sampling iterations, i.e. how many times the
  sampling and PFE evaluation are performed.  The statistical fluctuation
  of the result is used to estimate the PFE error.
* `nproc` is the number of parallel processes to use. Each process
  performs one sampling iteration. You can set this e.g. to the number
  of CPU cores in your machine.

### `[integral]` section

This section specifies parameters for the numerical integration of
the partition function.

* `lower` is the lower bound of the coordinate range
* `upper` is the upper bound of the coordinate range
* `dx` is the integration step size

### `[potential]` section

This section specifies which potential function to use, and its parameters.
The mandatory parameter in this section is `func`, which is the name of
the function.  The available functions and their parameters are:

* `harmonic`: 1D harmonic oscillator
  * `k`: the force constant
  * `x0`: the equilibrium position
* `double-well`: double well potential `a x^2 - b x^3 + c x^4`
  * `a`, `b`, `c`: the coefficients
* `double-well-sym`: symmetric double well potential; one minimum is at position x=0 with potential V=0
  * `x0`: the position of the other minimum
  * `h`: the height of the barrier
* `double-well-asym`: asymmetric double well potential; the parameters are as for `double-well-sym`, plus:
  * `v`: the height of the other minimum
* `mueller-brown`: the 2D Mueller-Brown potential; there are no parameters.


### `[trajectory]` section

This section specifies how to collect the samples.

* `method`: either `monte-carlo` (MC) or `replica-exchange` (RE)
* `temp`: the temperature parameter
* `kB`: Boltzmann constant in your units of choice, e.g. `kB = 0.0019872` in kcal/mol/K
* `seed`: if set to an integer, fix the random seed; for each sampling iteration, it will
   be increased by one; if set to a non-integer (e.g. `seed = none`) the seed will be
   initialized randomly.
* `nsteps`: the number of MC steps (per replica, if RE is used)
* `stepsize`: the MC step size
* `outfreq`: a sample is collected every `outfreq` MC steps
* `save`: whether to save the collected samples; `yes` or `no`.

If RE is used, also the following parameters are needed:

* `nreplica`: the number of replicas
* `temp-max`: the maximum temperature. The minimum temperature is taken from the `temp`
  parameter, and the replicas are equidistantly spaced in temperature.
* `exfreq`: replicas are exchanged every `exfreq` MC steps


### `[pfe]` section

* `nbins`: the number of bins used (per dimension) for estimating the configuration
  space covered during the sampling
* `threshold`: for the PFE evaluation, a fraction of samples with the highest energies
  are discarded. `threshold` specifies this fraction, e.g. `0.01` for 1%.
  This can either be a single number, or a comma-separated list of numbers, in
  which case the PFE evaluation is performed for each of the listed thresholds in turn.


## Output

The main output (written to standard output) contains the following information:

* Hash and date of the Git commit of the running code.
* Program start and end time.
* A copy of all input file settings (taking into account any settings
  overridden with the `--param` options).
* `ln Z_int`: the value of ln(Z) obtained by numerical integration
* `ln Z_exact`: the analytic value of ln(Z) (where available)
* `ln Z_est(t)`: the PFE estimate for threshold `t`, in the form `mean ± std (sigma)`
   where the mean and standard deviation (`std`) are obtained from the multiple
   iterations.  `t=0.0` refers to the PFE estimate that is based on *all*
   samples without discarding any high-energy outliers.  `sigma` provides an estimate
   for the error in the PFE estimate of ln(Z) due to the sampling -- this can be
   calculated for each trajectory, and here the mean over all trajectories is printed.
   In general, `sigma` should be smaller than `std`, as there is an additional error
   from the volume calculation.
* `ln Z_est(t) - ln Z_exact`: the difference of the PFE estimate from the "exact" value,
  which is either the analytic value or the numeric value.

Additionally, each sampling iteration creates a subdirectory `out-0000` etc,
containing the following files:

* `log` records some basic statistics for the MC/RE sampling, as well as the PFE
  estimation of ln(Z) for this sample and the given thresholds. `ln Z_est(0.0)` is the PFE
  estimation without removing any outliers (i.e. technically with threshold = 0.0)
  while `ln Z_est(t)` is the PFE estimation for threshold value `t`.
  Further, for each threshold, an estimate for the error in ln(Z) due to the sampling
  is provided (`σ(lnZ)`), and a more optimal value for the energy cutoff `E*` is
  suggested.  (Following the `E*` suggestions iteratively should result in minimizing
  `σ(lnZ)`.)
* If `save = yes` was given in the `[trajectory]` section:
  * `Traj.dat` contains the samples, one per line (space-separated coordinates).
  * `Energy.dat` contains the potential energy for the samples, one per line.

## Reading in saved samples

It is also possible to read in existing samples, e.g. if you want to try out
additional values for the PFE threshold without having to re-do the
(potentially time-consuming) sampling.  To do so, use `method = read` in
the `[trajectory]` section. You also need to specify the `temp` and `kB` parameters.

In this mode, the per-iteration `log` file is not overwritten. Instead, the
new PFE estimates are appended to it.

For example, to run just the PFE estimate with threshold values 5.0 and 6.0
based on existing samples that had been generated from input file `input.conf`,
you can run (make sure not to overwrite your original output file!)

    ./main.py -p trajectory.method=read -p pfe.threshold=5.0,6.0 input.conf | tee output2.log

Additionally, by setting `trajectory.nsteps` to a lower value than originally,
the read-in trajectory will be truncated accordingly (namely to
`trajectory.nsteps // trajectory.outfreq` samples). This way, one can explore
the effect of the trajectory length without having to run the expensive sampling
step for all desired trajectory lengths. Example:

    ./main.py -p trajectory.method=read -p trajectory.nsteps=100000 input.conf | tee output3.log

