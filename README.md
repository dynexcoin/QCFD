# QCFD
Implementation of QC in Computational Fluid Dynamics (CFD) as QUBO/Ising Problem using DYNEX Neuromorphic Computing, based on [An Introduction to Algorithms in Quantum Computation of Fluid Dynamics](https://www.sto.nato.int/publications/STO%20Educational%20Notes/STO-EN-AVT-377/EN-AVT-377-01.pdf), Sachin S. Bharadwaj and Katepalli R. Sreenivasan, Department of Mechanical and Aerospace Engineering, STO - Educational Notes Paper, 2022.

- [Documentation](https://github.com/dynexcoin/QCFD/blob/main/QCFD_Documentation.pdf)
- [Video](https://github.com/dynexcoin/QCFD/raw/main/IMG_7574.MP4)
- [Examples](https://github.com/dynexcoin/QCFD/blob/main/QCFD_Examples.ipynb)

![Image](https://github.com/dynexcoin/QCFD/blob/main/imgs/4.png)

This repository provides a Python class used to converting the Harrow-Hassidim-Lloyd (HHL) algorithm, typically used for solving linear systems on quantum computers, into a Quadratic Uncon- strained Binary Optimization (QUBO) model termed as ”QCFD” to be computed on DYNEX Neu- romorphic Network. This adaptation allows the use of classical and quantum-inspired solvers (a.k.a Simulated Annealing Sampler) and DYNEX Network users for finding solutions.

Several physical systems such as turbulent flows, glassy systems, protein folding and chemical reactions, are formidably hard to simulate “efficiently” on even the biggest supercomputers. However, with the growth in high performance computing (HPC) technology and equally “efficient” algorithms, computational fluid dynamics (CFD) in particular, has progressed immensely over the last several decades. Some state-of-the-art Direct Numerical Simulations (DNS) of turbulent flows governed by the Navier-Stokes equations, simulate flows at overwhelmingly large Reynolds numbers (Re) with high resolutions. These simulations not only reveal the fine details of the flow physics, but also test the limits of supercomputers they are run on. Let’s for instance refer to Figure 1, which shows how HPC, isotropic DNS simulations have evolved with time and increasing computational power, all the way from 323 to ≈ 180003 grid point simulations (a-f). We see that it roughly scales with Moore’s law. The [Dynex Neuromorphic Computing Platform](https://live.dynexcoin.org) provides a platform to overcome such limitations.




