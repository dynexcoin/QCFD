# QCFD
Implementation of QC in Computational Fluid Dynamics (CFD) as QUBO/Ising Problem using DYNEX Neuromorphic Computing, based on [An Introduction to Algorithms in Quantum Computation of Fluid Dynamics](https://www.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-AVT-377/EN-AVT-377-01.pdf), Sachin S. Bharadwaj and Katepalli R. Sreenivasan, Department of Mechanical and Aerospace Engineering, STO - Educational Notes Paper, 2022.

- [Documentation](https://github.com/dynexcoin/QCFD/blob/main/QCFD_Documentation.pdf)
- [Video](https://github.com/dynexcoin/QCFD/raw/main/IMG_7574.MP4)

This repository provides a Python class used to converting the Harrow-Hassidim-Lloyd (HHL) algorithm, typically used for solving linear systems on quantum computers, into a Quadratic Uncon- strained Binary Optimization (QUBO) model termed as ”QCFD” to be computed on DYNEX Neu- romorphic Network. This adaptation allows the use of classical and quantum-inspired solvers (a.k.a Simulated Annealing Sampler) and DYNEX Network users for finding solutions.


