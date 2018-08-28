# MDQTPlasmaSims
repository of code used for combined molecular dynamics &amp; quantum trajectories simulations described in PhD Thesis of Thomas Langin (Rice U., 2018)

## TABLE OF CONTENTS

I Introduction
II List of .cpp and .slurm files with brief description and list of input parameters (typically within first 100 lines of .cpp files) and instructions for running the code
III List of typical output files

### I Introduction

This repository contains code for the various Molecular Dynamics (MD) simulations of Yukawa One-Component Plasmas discussed in Thomas K. Langin's Rice University PhD Thesis (add link later).  These simulations record positions and velocities of ions interacting through the Yukawa OCP Hamiltonian; Hamilton's Eqs of motion are evaluated every timestep to calculate forces, from which changes in velocity and position are determined.  

In some simulations, we also record the wavefunction of the ions, which is evolved using a Quantum Trajectories (QT) approach, as also described in the PhD Thesis (Chapter 4).  Given a set of laser parameters (detuning and Rabi frequencies, typically) and 'real' plasma density (in 10^14 m^-3), the quantum state of each ion can be evolved using the Hamiltonian determined from the atom-light coupling (the evolution also depends on ion velocity through the Doppler Shift).  The optical force is also calcualted in these cases using the standard Heisenburg picture evolution of dp/dt = [H,\nabla] where H is the atom-light hamiltonian (e.g., contains terms like (\delta - kv)|e\rangle\langle e| + \Omega |e\rangle\langle g|, etc.)

### II List of .cpp and .slurm files with brief description and list of input parameters (typically within first 100 lines of .cpp files) and instructions for running the code.

All .cpp files presented here will have the following info

Description: Brief description of file
User Inputs: List of user inputs (plasma density, laser detunings, etc.)
Compile Instructions (NOTE: To SUBMIT jobs after compilation see the description for the exampleSlurmFile.slurm (the last file in this section))
Output Directory Structure and File Type List: Tells you where the simulation 'data' will be stored and what types of files are recorded: for deeper decription of file types see Section III: List of typical output files

#### A) LaserCoolingPlusExpansionMDQT.cpp

Description: This code will run the laser cooling and expansion MolecularDynamics + Quantum-Trajectories (MDQT) code described in Chapter 4 of the thesis.  

User Inputs: The user will provide

1) Ge: Electron \Gamma (typically 0.1)
2) density (plasma density in units 10^14 m^-3.  For simulations in thesis this was set to 2.0)
3) sig0 (size of the plasma in units mm.  In thesis this was 4.0)
4) Te (electron temperature: theoretically this should be calculated from Ge and density, but in practice we just input it.  For thesis this was 19.0)
5) fracOfSig (sets the initial position of the 'chunk' of plasma you are simulating (Chatper 4.6.2 of TKL thesis).  Thesis includes data where this was 0.0 (e.g., the plasma center), 0.5, and 1.0
6) N0 (the number of particles, typically 3000)
7) detuning (detuning of S-> P lasers in units of \gamma_{SP}=1.41e8 s^-1.  Set to -1.0 for all thesis data)
8) detuningDP (detuning of D->P lasers in units of \gamma_{SP}.  Typically set to either +1.0 (for best cooling data) or 0.0 (for EIT supression data))
9) Om (Rabi frequency of S->P transition w/o inclusion of C-G coefficients (these are defined in 'main()' at the end of the file) in units \gamma_{SP}.  Typically 1.0 for thesis data)
10) OmDP (Rabi frequency of D->P transition w/o C-G coeffs in units \gamma_{SP}.  Typically 1.0)
11) char saveDirectory[256] : name of "head" folder where output data will be stored

Compile Instructions:

1) on cluster after loading into /scratch/USERNAME/(whatever sub directory you want to store this in): First type "module load GCC/4.9.3" (no quotes).  Then type "g++ -std=c++11 -fopenmp -o runFile -O3 LaserCoolingPlusExpansionMDQT.cpp -lm -I/home/USERNAME/usr/include -L/users/USERNAME/user/lib64" where 'runFile' is whatever you want to name the executable and USERNAME is your username (e.g., I would use tkl1)

2) on home computer (assuming you have installed armadillo package).  Type "g++ -std=c++11 -fopenmp -o runFile -O3 LaserCoolingPlusExpansionMDQT.cpp -lm -larmadillo" where 'runFile' is whatever you want to name the executable

Output Directory Structure and File Type List: Output Files will be stored in subdirectories of form ../(directory the cpp file is in)/saveDirectory(see Above)/Ge__Density__E+11Sig0__Te__SigFrac_DetSP__DetDP__OmSP__OmDP__NumIons__/job__/

where the underscores indicate, in order

1) Ge X 100
2) density X 1000
3) sig0 X 10
4) Te
5) fracOfSig X 100
6) detuning X 100
detuningDP X 100
Om X 100
OmDP X 100
N0
Job Number (the number of jobs will be set by the slurm file).

Each job directory will contain output files of type: conditions_timestep___, energies, ions_timestep__, statePopulationsVsVTime__, vel_distX_time___, vel_DistY_time__, vel_DistZ_time__, VZERO_timestep__interval_, wvFns__.  For description of what these mean, see III: List of typical output files

