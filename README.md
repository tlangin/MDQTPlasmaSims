# MDQTPlasmaSims
repository of code used for combined molecular dynamics &amp; quantum trajectories simulations described in PhD Thesis of Thomas Langin (Rice U., 2018)

# TABLE OF CONTENTS

I Introduction

II List of .cpp and .slurm files with brief description and list of input parameters (typically within first 100 lines of .cpp files) and instructions for running the code

III List of typical output files

# I Introduction

This repository contains code for the various Molecular Dynamics (MD) simulations of Yukawa One-Component Plasmas discussed in Thomas K. Langin's Rice University PhD Thesis (add link later).  These simulations record positions and velocities of ions interacting through the Yukawa OCP Hamiltonian; Hamilton's Eqs of motion are evaluated every timestep to calculate forces, from which changes in velocity and position are determined.  

In some simulations, we also record the wavefunction of the ions, which is evolved using a Quantum Trajectories (QT) approach, as also described in the PhD Thesis (Chapter 4).  Given a set of laser parameters (detuning and Rabi frequencies, typically) and 'real' plasma density (in 10^14 m^-3), the quantum state of each ion can be evolved using the Hamiltonian determined from the atom-light coupling (the evolution also depends on ion velocity through the Doppler Shift).  The optical force is also calcualted in these cases using the standard Heisenburg picture evolution of dp/dt = [H,\nabla] where H is the atom-light hamiltonian (e.g., contains terms like (\delta - kv)|e\rangle\langle e| + \Omega |e\rangle\langle g|, etc.)

# II List of .cpp and .slurm files with brief description and list of input parameters (typically within first 100 lines of .cpp files) and instructions for running the code.

All .cpp files presented here will have the following info

**Description**: Brief description of file

**User Inputs**: List of user inputs (plasma density, laser detunings, etc.)

**Compile Instructions** (NOTE: To SUBMIT jobs after compilation see the description for the exampleSlurmFile.slurm (the last file in this section))

**Output Directory Structure and File Type List**: Tells you where the simulation 'data' will be stored and what types of files are recorded: **for deeper decription of output file types see Section III: List of typical output files**

## A) LaserCoolingPlusExpansionMDQT.cpp

### Description: 

This code will run the laser cooling and expansion MolecularDynamics + Quantum-Trajectories (MDQT) code described in Chapter 4 of the thesis.  

### User Inputs: 

The user will provide

1) Ge: Electron \Gamma (typically 0.1)
2) density (plasma density in units 10^14 m^-3.  For simulations in thesis this was set to 2.0)
3) sig0 (size of the plasma in units mm.  In thesis this was 4.0)
4) Te (electron temperature: theoretically this should be calculated from Ge and density, but in practice we just input it.  For thesis this was 19.0)
5) fracOfSig (sets the initial position of the 'chunk' of plasma you are simulating (Chatper 4.6.2 of TKL thesis).  Thesis includes data where this was 0.0 (e.g., the plasma center), 0.5 (x=+2mm), and 1.0 (x=+4mm, for \sigma = 4 mm))
6) N0 (the number of particles, typically 3500)
7) detuning (detuning of S-> P lasers in units of \gamma_{SP}=1.41e8 s^-1.  Set to -1.0 for all thesis data)
8) detuningDP (detuning of D->P lasers in units of \gamma_{SP}.  Typically set to either +1.0 (for best cooling data) or 0.0 (for EIT supression data))
9) Om (Rabi frequency of S->P transition w/o inclusion of C-G coefficients (these are defined in 'main()' at the end of the file) in units \gamma_{SP}.  Typically 1.0 for thesis data)
10) OmDP (Rabi frequency of D->P transition w/o C-G coeffs in units \gamma_{SP}.  Typically 1.0)
11) char saveDirectory[256] : name of "head" folder where output data will be stored
12) int newRun: 1 if a new run, 0 if continuing a run.  Sometimes you will want to run the simulation for a time longer than 8 hours (for 3500 particles, 8 hours corresponds to a total simulation time of 37 time units...so choose tmax = 30 or less to be safe!).  At the end of the simulation, conditions will be recorded (conditions_timestepXXX_.dat, ions_timestepXXX.dat, wvFns_timestepXXX.dat).  If newRun=0 (and c0 is set appropriately, see next variable), these will be read in.  If newRun=1, c0 should be 0 and initial conditions are random.
13) int c0: timestep counter.  set c0=0 if newRun=1.  If newRun=0, then c0 should correspond to the timestamp of the conditions you want read (e.g., if, when the simulation ended, the output files are "conditions_timestep022500.dat" (etc.), then c0 should be 022500).
14) tmax: maximum simulation time.  Note that this is for the total simulation, including previous runs (i.e., if you finished a simulation with tmax=30 and you want to continue for another 30 time units, set newRun=0, c0 to whatever it's supposed to be, and tmax=60).

### Compile Instructions:

1) on cluster after loading into /scratch/**USERNAME**/(whatever sub directory you want to store this in): First enter "module load GCC/5.4.0" (no quotes).  Then enter "module load OpenMPI/1.10.3".  Then enter "module load Armadillo/7.600.1".   Then enter "g++ -std=c++11 -fopenmp -o **runFile** -O3 LaserCoolingPlusExpansionMDQT.cpp -lm -larmadillo" where '**runFile**' is whatever you want to name the executable to compile.  

2) on home computer (assuming you have installed armadillo package).  Type "g++ -std=c++11 -fopenmp -o **runFile** -O3 LaserCoolingPlusExpansionMDQT.cpp -lm -larmadillo" where '**runFile**' is whatever you want to name the executable

### Run Instructions On Cluster:

To run your code on cluster, you need to set up a SLURM file (I've attached one to this github).  First, open the slurm file and make sure you are running as many jobs as you want (e.g., if you want ten jobs, #SBATCH --array=1-10) and that you are running the correct file (e.g., if you have compiled into an executable named "testFile", then there should be a statement like "srun testFile $SLURM_ARRAY_TASK_ID).  Finally, if you want email updates, set #SBATCH --mail-user=blah@rice.edu (or whatever...the cluster can email gmail as well).  Once your slurm file is setup, to run type

sbatch slurmFile.slurm

This will submit your jobs to the cluster

### Output Directory Structure and File Type List: 

Output Files will be stored in subdirectories of form 

"**saveDirectory(see Above)**/Ge__Density__E+11Sig0__Te__SigFrac_DetSP__DetDP__OmSP__OmDP__NumIons__/job__/"

where the underscores indicate, in order

1) Ge X 100
2) density X 1000
3) sig0 X 10
4) Te
5) fracOfSig X 100
6) detuning X 100
7) detuningDP X 100
8) Om X 100
9) OmDP X 100
10) N0
11) Job Number (the number of jobs will be set by the slurm file).

Each job directory will contain output files of type: 

1) conditions_timestepXXX
2) energies
3) ions_timestepXXX
4) statePopulationsVsVTime__
5) vel_distX_time___, vel_DistY_time__, vel_DistZ_time__ 
6) VZERO_timestep__interval_
7) wvFns__timestepXXX.  

For description of what these mean, see **III: List of typical output files**

## B) otherFiles.cpp (placeholder)

# III: List of typical output files

## energies.dat

Probably the most important one when considering laser-cooling simulations.  First column is time (units of \omega_{E}^-1).  Second column is "x kinetic energy" in units 0.5\Gamma^-1.  Third and fourth are Y and Z kinetic energies.  Fifth is total potential energy divided by number of particles.  Sixth is total energy at time 't' minus total energy at time '0' (so, this will become increasingly negative as ions are laser-cooled).  Seventh is mean x velocity (useful when considering ions in an expanding plasma...ignore if 'fracOfSig = 0').

## statePopulationsVsVTime____ .dat

title reflects what time of the simulation the state populations are recorded (e.g., statePopulationsVsVTime000400.dat will refer to the '400th' time for which things are recorded, which is the 400th entry in the first column of 'energies.dat').  There will be N rows where N is the number of particles.  For a given row i:

Column 1 refers to the velocity of particle i.
Column 2 refers to the "s" population of particle i (e.g., |<1|\psi>|^2 + |<2|\psi>|^2) where \psi is particle "i"'s wavefunction and <1| and <2| are the two "s" eigenstates (m=\pm 1/2).
Column 3 refers to the "p" population of particle i
Column 4 refers to the "d" population of particle i.

This is primarily useful when looking for dark states.  Just bin column 3 (the P population) vs column 1 to get a P_{p}(v) plot, the population of being in a "P" state as a function of velocity: dips in this plot correspond to dark state locations (see TKL thesis 4.5)

## vel_distX_time___ .dat (and vel_distY, vel_distZ)

title again reflects the "time" for which this file corresponds to.  This is basically the velocity distribution of the ion.  Column 1 is a value of v-<v_x>.  Column 2 represents the number of particles at this velocity value.  

## VAF_interval0.dat

ignore this in laser-cooling code.

## J_interval0.dat

ignore this in laser-cooling code

## conditions_timestepXXX.dat

When simulation concludes, ion positions (columns 1 thru 3) and velocities (columns 4 thru 6) are recorded.  This are then read in whenever a simulation is continued by setting newRun=0 and c0 = XXX.

## wvFns_timestepXXX.dat

When simulation concludes, ion wavefunctions are also recorded (1 row per particle.  1st and 2nd entry are the real and imaginary componens of <\psi|1>, 3rd and 4th are real and imaginary components of <\psi|2>, etc., where |1>, |2>, etc. are the eigenstates as described in TKL thesis and |\psi> is the ion wavefunction). This are then read in whenever a simulation is continued by setting newRun=0 and c0 = XXX.

## ions_timestepXXX.dat

When simulation concludes, the total number of ions in this run is saved and read in whenever a simulation is continued by setting newRun=0 and c0 = XXX.

