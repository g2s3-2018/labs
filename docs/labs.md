---
title: Labs
layout: default
---

- **Ill-posedness** (Monday 06/18):
  - [Inverse problem prototype](notebooks/inverseProblemPrototype.html): An illustrative example of an ill-posed inverse problem ( [.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab1_Monday/inverseProblemPrototype.ipynb) )

- **Finite element method, calculus of variations, image denoising** (Tuesday 06/19):
  - [Poisson2D](notebooks/Poisson2D.html): Finite element solution of the Poisson equation in 2D using FEniCS ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab2_Tuesday/Poisson2D.ipynb) )
  - [Convergence rates](notebooks/ConvergeRates2D.html): Convergence rates of the finite element method for the Poisson equation in 2D ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab2_Tuesday/ConvergenceRates2D.ipynb))
  - [Unconstrained minimization](notebooks/UnconstrainedMinimization.html): This notebook illustrates the minimization of a non-quadratic energy functional using Netwon Method ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab3_Wednesday/UnconstrainedMinimization.ipynb))
  - [Image denoising](notebooks/ImageDenoising.html): This notebook illustrate the use of Tikhonov and Total Variation regularization to solve an image denoising problem ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab2_Tuesday/ImageDenoising/ImageDenoising.ipynb))
  
- **Deterministic Inversion** (Thursday 06/21):
  - [Poisson SD](notebooks/Poisson_SD.html): This notebook illustrates the use of hIPPYlib/fenics for solving a determinisitc inverse problem for the coefficient field of a Poisson equation, using the steepest descent method
  ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab4_Thursday/Poisson_SD.ipynb) ). *Note that SD is a poor choice of optimization method for this problem; it is provided here in order to compare with Newton’s method in the notebook below*
  - [Poisson INCG](notebooks/Poisson_INCG.html): This notebook illustrates the use of hIPPYlib/FEniCS for solving an inverse problem for the coefficient field of a Poisson equation, using the inexact Newton CG method ( [.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab4_Thursday/Poisson_INCG.ipynb) )
  - [Spectrum of Hessian operator](notebooks/HessianSpectrum.html): This notebook illustrates the spectral properties of the preconditioned Hessian misfit operator ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab4_Thursday/HessianSpectrum.ipynb))
   
- **Gaussian processes, Bayesian inference for linear inverse problems** (Friday 06/22):
  - Gaussian processes
  - Bayesian Linear Regression
  - Inferring loads on an Euler Bernoulli beam
  
- **Sampling methods, logistic regression** (Monday 06/25):
  - Sampling methods
  - Multinomial Logistic Regression for Sea Ice Age
  
- **Bayesian inference, Markov Chain Monte Carlo** (Tuesday 06/26):
  - Inferring material properties of a cantilevered beam
  
- **Infinite dimesional Bayesian inference, Laplace Approximation, MCMC** (Wednesday 06/27):
  - [Gaussian priors](notebooks/Gaussian_priors.html): This notebook illustrate how to construct PDE-based priors that lead to well-posed Bayesian inverse problems in infinite dimesions ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab8_Wednesday/Gaussian_priors.ipynb) )
  - [Poisson Bayesian](notebooks/SubsurfaceBayesian.html): This notebook illustrates how to solve a non-linear parameter inversion for the Poisson equation in a Bayesian setting using hIPPYlib ([.ipynb](https://github.com/g2s3-2018/labs/blob/master/Labs/Lab8_Wednesday/SubsurfaceBayesian.ipynb) )


 
