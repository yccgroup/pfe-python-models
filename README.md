# RAFEP Python Models

This code is a sample implementation of the RAFEP method for simple
1D or 2D model potentials.  The code performs the following:

- Collect a number of samples from the canonical distribution of
  the given potential function, via Monte-Carlo or Replica-Exchange.
- Estimate the value of the partition function Z (at the given
  temperature) from the collected sample, via the RAFEP method.
- To remove the effect of sampling outliers, multiple values of
  the RAFEP energy threshold can be specified.
- The sampling and RAFEP estimation is performed multiple times,
  in order to estimate the error of the RAFEP estimate.
- The actual value of Z is also calculated analytically (where
  possible) and numerically, so that one can verify the correctness
  of the RAFEP result.


## Prerequisites

The code is written in Python 3 and depends on the packages `numpy`
and `scipy`.  If your Python install does not include them, you can
create a virtual environment and install them via `pip`:

    python3 -m venv venv
    . venv/bin/activate
    pip3 install -r requirements.txt


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
  sampling and RAFEP evaluation are performed.  The statistical fluctuation
  of the result is used to estimate the RAFEP error.
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


### `[rafep]` section

* `threshold`: for the RAFEP evaluation, samples with energy larger than `threshold * kB * temp`
  are discarded. This can either be a single number, or a comma-separated list of numbers, in
  which case the RAFEP evaluation is performed for each of the listed thresholds in turn.


## Output

The main output (written to standard output) contains the following information:

* Hash and date of the Git commit of the running code.
* Program start and end time.
* A copy of all input file settings (taking into account any settings
  overridden with the `--param` options).
* `Z_int`: the value of Z obtained by numerical integration
* `Z_exact`: the analytic value of Z (where available)
* `Z_est(t)`: the RAFEP estimate for threshold `t`, in the form `mean ± std`
   where the mean and standard deviation (std) are obtained from the multiple
   iterations.  `t=inf` refers to the RAFEP estimate that is based on *all*
   samples without discarding any outliers.
* `Z_est(t)/Z_exact`: the ratio of the RAFEP estimate and the "exact" value,
  which is either the analytic value or the numeric value.

Additionally, each sampling iteration creates a subdirectory `out-0000` etc,
containing the following files:

* `log` records some basic statistics for the MC/RE sampling, as well as the RAFEP
  estimation of Z for this sample and the given thresholds. `Z_est(inf)` is the RAFEP
  estimation without removing any outliers (i.e. technically with threshold = infinity)
  while `Z_est(t)` is the RAFEP estimation for threshold value `t`.
* If `save = yes` was given in the `[trajectory]` section:
  * `Traj.dat` contains the samples, one per line (space-separated coordinates).
  * `Energy.dat` contains the potential energy for the samples, one per line.

## Reading in saved samples

It is also possible to read in existing samples, e.g. if you want to try out
additional values for the RAFEP threshold without having to re-do the
(potentially time-consuming) sampling.  To do so, use `method = read` in
the `[trajectory]` section. You also need to specify the `temp` and `kB` parameters.

In this mode, the per-iteration `log` file is not overwritten. Instead, the
new RAFEP estimates are appended to it.

For example, to run just the RAFEP estimate with threshold values 5.0 and 6.0
based on existing samples that had been generated from input file `input.conf`,
you can run (make sure not to overwrite your original output file!)

    ./main.py -p trajectory.method=read -p rafep.threshold=5.0,6.0 input.conf | tee output2.log

