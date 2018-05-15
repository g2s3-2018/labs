---
title: Labs
layout: default
---

- Monday 06/18 (jupyter & FEniCS - only 1.5 hours):
  - Handout log in information
  - Introduction to FEniCS (Poisson Equation in 1D/2D and FE convergence rates);
     see [here1](https://uvilla.github.io/inverse17/02_IntroToFenics/Poisson1D.html)
     see [here2](https://uvilla.github.io/inverse17/02_IntroToFenics/ConvergenceRates.html)
     see [here3](https://uvilla.github.io/inverse17/02_IntroToFenics/Poisson2D.html)
  
- Tuesday 06/19 (hIPPYlib - only 1.5 hours): 
  - Calculus of Variation: Unconstrained Energy Minimization.
    see [here](https://uvilla.github.io/inverse17/04_UnconstrainedMinimization/UnconstrainedMinimization.html)
  
- Wednesday 06/20 (hIPPYlib):
  - Deterministic Inversion: Steepest Descent Algorithm for Inverse Permeability Poisson Problem:
    see [here](https://uvilla.github.io/inverse17/05_Poisson_SD/Poisson_SD.html)
  - Deterministic Inversion: Inexaxt Newton-CG Algorithm for Inverse Permeability Poisson Problem:
    see [here](https://uvilla.github.io/inverse17/06_Poisson_INCG/Poisson_INCG.html)
  - TODO: Extends the above codes to Advection Diffusion Problems
  
- Thursday 06/21 (hIPPYlib):
  - Explain randomized eigensolvers
  - Spectrum of the Hessian for a linear inverse problem.
    see [here](https://uvilla.github.io/inverse17/03_HessianSpectrum/HessianSpectrum.html)
   
- Friday 06/22 (MUQ):
  - Simple examples in Bayesian computation and MC methods
  
- Monday 06/25 (MUQ & hIPPYlib):
  - Linear Bayesian Inversion in hip: Inferring the Initial Condition for a Advection Diffusion Problem
    see [here](https://uvilla.github.io/inverse17/08_AddDivBayesian/AddDivBayesian.html)
  - Matt ??
  
- Tuesday 06/26 (MUQ & hIPPYlib):
  - Bayesian Inversion in hIPPYlib: Linearized Inference of Log permeability
    see [here](https://uvilla.github.io/inverse17/07_PoissonBayesian/PoissonBayesian.html)
  - Matt: extension to MCMC
  
- Wednesday 06/27:
  - Student projects
  
- Thursday 06/29:
  - Student projects
  
  ## Comments:
  - Besides the intro to FEniCS notebook, most of the other notebooks in hIPPYlib require
    some knowledge in numerical optimization and calculus of variations. It could make sense to have a full lecture on Monday afternoon
    and start with the lab sessions on Tuesday afternoon.
  - I am not sure what to do exactly with the 06/25 and 06/27 labs. As they are they seemed to fragmented.
     We need to leave some space to MUQ, but we also have to show the hippylib notebooks for the Linear Bayesian Advection Diffusion
     and the Linearized Bayesian Inference of the Log-permeability.
 
