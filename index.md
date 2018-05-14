---
title: Introduction
layout: default
---

We will use these libraries for the hands-on interactive learning exercises that complement the morning lectures:

- [hIPPYlib](https://hippylib.github.io/) (Inverse Problems with Python libraries) implements state-of-the-art scalable algorithms
for PDE-based deterministic and Bayesian inverse problems.
It builds on [FEniCS](https://fenicsproject.org/) (a parallel finite element element library) for the discretization
of the PDEs and on [PETSc](https://www.mcs.anl.gov/petsc/) for scalable and efficient linear algebra operations and solvers.

- [MUQ](http://muq.mit.edu/) (MIT Uncertainty Quantification) provides tools for exact sampling of non-Gaussian posteriors, approximating computationally intensive forward models, implementing integral covariance operators, characterizing predictive uncertainties, and defining the Bayesian models required for these tasks.

Here are a few important logistics:

- We will use cloud-based interactive tutorials that mix instruction and theory with editable and runnable code.
You can run the codes presented in the hands-on workshop through your web browser.
This will allow anyone to test our software and experiment with inverse problem algorithms quite easily,
without running into installation issues or version discrepancies.
In the first hand-on session, you will be provided with an ip address, user name and password and will be able to access
the codes via ipython notebooks. Please do not exchange the user info.

- If you are not familiar with FEniCS, the fastest way to start learning this tool is to download
and read the first chapter of the FEniCS book from [here](https://launchpadlibrarian.net/83776282/fenics-book-2011-10-27-final.pdf).
Note the updated/new FEniCS tutorial version [here](http://hplgit.github.io/fenics-tutorial/doc/pub/fenics-tutorial-4print.pdf).
For more detailed instructions, please check out the ''Getting started with FEniCS'' document available 
[here](http://faculty.ucmerced.edu/npetra/docs/).
