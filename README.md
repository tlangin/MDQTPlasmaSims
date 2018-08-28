# MDQTPlasmaSims
repository of code used for combined molecular dynamics &amp; quantum trajectories simulations described in PhD Thesis of Thomas Langin (Rice U., 2018)

This repository contains code for the various Molecular Dynamics (MD) simulations of Yukawa One-Component Plasmas discussed in Thomas K. Langin's Rice University PhD Thesis (add link later).  These simulations record positions and velocities of ions interacting through the Yukawa OCP Hamiltonian; Hamilton's Eqs of motion are evaluated every timestep to calculate forces, from which changes in velocity and position are determined.  

In some simulations, we also record the wavefunction of the ions, which is evolved using a Quantum Trajectories (QT) approach, as also described in the PhD Thesis (Chapter 4).  Given a set of laser parameters (detuning and Rabi frequencies, typically) and 'real' plasma density (in 10^14 m^-3), the quantum state of each ion can be evolved using the Hamiltonian determined from the atom-light coupling (the evolution also depends on ion velocity through the Doppler Shift).  The optical force is also calcualted in these cases using the standard Heisenburg picture evolution of dp/dt = [H,\nabla] where H is the atom-light hamiltonian (e.g., contains terms like (\delta - kv)|e\rangle\langle e| + \Omega |e\rangle\langle g|, etc.)

