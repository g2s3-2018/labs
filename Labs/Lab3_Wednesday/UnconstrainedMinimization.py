# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 3.0 dated June 2007.

# Example: Nonlinear energy functional minimization
# 
# In this example we solve the following nonlinear minimization problem
# 
# Find u^* \in H^1_0(\Omega) such that
#    u^* = argmin_{u \in H^1_0(\Omega)} Pi(u)
# 
# Here the energy functional Pi(u) has the form
# Pi(u) = \frac{1}{2} \int_\Omega k(u) \nabla u \cdot \nabla u dx - \int_\Omega f\,u dx,
# where
# k(u) = k_1 + k_2 u^2.

# 1. Load modules

from dolfin import *

import math
import numpy as np
import logging

import matplotlib.pyplot as plt
import nb

logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
set_log_active(False)

# 2. Define the mesh and finite element spaces

nx = 32
ny = 32
mesh = UnitSquareMesh(nx,ny)
Vh = FunctionSpace(mesh, "CG", 1)

uh = Function(Vh)
u_hat = TestFunction(Vh)
u_tilde = TrialFunction(Vh)

nb.plot(mesh)
print "dim(Vh) = ", Vh.dim()

# 3. Define the energy functional

f = Constant(1.)
k1 = Constant(0.05)
k2 = Constant(1.)

Pi = Constant(.5)*(k1 + k2*uh*uh)*inner(nabla_grad(uh), nabla_grad(uh))*dx - f*uh*dx

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

u_0 = Constant(0.)    
bc = DirichletBC(Vh,u_0, Boundary() )

# 4. First variation (gradient)

grad = (k2*uh*u_hat)*inner(nabla_grad(uh), nabla_grad(uh))*dx + \
       (k1 + k2*uh*uh)*inner(nabla_grad(uh), nabla_grad(u_hat))*dx - f*u_hat*dx

u0 = interpolate(Expression("x[0]*(x[0]-1)*x[1]*(x[1]-1)"), Vh)

n_eps = 32
eps = 1e-2*np.power(2., -np.arange(n_eps))
err_grad = np.zeros(n_eps)

uh.assign(u0)
pi0 = assemble(Pi)
grad0 = assemble(grad)

dir = Function(Vh)
dir.vector().set_local(np.random.randn(Vh.dim()))
bc.apply(dir.vector())
dir_grad0 = grad0.inner(dir.vector())

for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector()) #uh = uh + eps[i]*dir
    piplus = assemble(Pi)
    err_grad[i] = abs( (piplus - pi0)/eps[i] - dir_grad0 )

plt.figure()    
plt.loglog(eps, err_grad, "-ob")
plt.loglog(eps, (.5*err_grad[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the first variation (gradient)")
plt.xlabel("eps")
plt.ylabel("Error grad")
plt.legend(["Error Grad", "First Order"], "upper left")


# 5. Second variation (Hessian)

H = k2*u_tilde*u_hat*inner(nabla_grad(uh), nabla_grad(uh))*dx + \
     Constant(2.)*(k2*uh*u_hat)*inner(nabla_grad(u_tilde), nabla_grad(uh))*dx + \
     Constant(2.)*k2*u_tilde*uh*inner(nabla_grad(uh), nabla_grad(u_hat))*dx + \
     (k1 + k2*uh*uh)*inner(nabla_grad(u_tilde), nabla_grad(u_hat))*dx

uh.assign(u0)
H_0 = assemble(H)
err_H = np.zeros(n_eps)
for i in range(n_eps):
    uh.assign(u0)
    uh.vector().axpy(eps[i], dir.vector())
    grad_plus = assemble(grad)
    diff_grad = (grad_plus - grad0)
    diff_grad *= 1/eps[i]
    H_0dir = H_0 * dir.vector()
    err_H[i] = (diff_grad - H_0dir).norm("l2")
    
plt.figure()    
plt.loglog(eps, err_H, "-ob")
plt.loglog(eps, (.5*err_H[0]/eps[0])*eps, "-.k")
plt.title("Finite difference check of the second variation (Hessian)")
plt.xlabel("eps")
plt.ylabel("Error Hessian")
plt.legend(["Error Hessian", "First Order"], "upper left")



# 6. The infinite dimensional Newton Method

uh.assign(interpolate(Constant(0.), Vh))

rtol = 1e-9
max_iter = 10

pi0 = assemble(Pi)
g0 = assemble(grad, bcs=bc)
tol = g0.norm("l2")*rtol

du = Function(Vh)

lin_it = 0
print "{0:3} {1:3} {2:15} {3:15} {4:15}".format(
      "It", "cg_it", "Energy", "(g,du)", "||g||l2")

for i in range(max_iter):
    [Hn, gn] = assemble_system(H, grad, bc)
    if gn.norm("l2") < tol:
        print "\nConverged in ", i, "Newton iterations and ", lin_it, "linear iterations."
        break
    myit = solve(Hn, du.vector(), gn, "cg", "petsc_amg")
    lin_it = lin_it + myit
    uh.vector().axpy(-1., du.vector())
    pi = assemble(Pi)
    print "{0:3d} {1:3d} {2:15e} {3:15e} {4:15e}".format(
      i, myit, pi, -gn.inner(du.vector()), gn.norm("l2"))
    
plt.figure()
nb.plot(uh, mytitle="Solution")

# 7. The built-in non-linear solver in FEniCS

uh.assign(interpolate(Constant(0.), Vh))
parameters={"symmetric": True, "newton_solver": {"relative_tolerance": 1e-9, "report": True, \
                                                 "linear_solver": "cg", "preconditioner": "petsc_amg"}}
solve(grad == 0, uh, bc, J=H, solver_parameters=parameters)
print "Built-in FEniCS non linear solver."
print "Norm of the gradient at converge", assemble(grad, bcs=bc).norm("l2")
print "Value of the energy functional at convergence", assemble(Pi)
plt.figure()
nb.plot(uh, mytitle="Build-in solver")
plt.show()

