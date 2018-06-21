import numpy as np
import random
import h5py

# MUQ Includes
import pymuqModeling as mm # Needed for Gaussian distribution
import pymuqApproximation as ma # Needed for Gaussian processes

# Include the beam model
from BeamModel import EulerBernoulli

# Discretization
numPts = 201
dim = 1
x = np.linspace(0,1,numPts)[None,:]

# Array of modulus values
E = 1e5*np.ones(numPts)

# Geometry of beam (assumes beam is cylinder with constant cross sectional area)
length = 1.0
radius = 0.1

# Create the beam model.  Note that the beam class has a member "K" holding the stiffness
# matrix.  Thus, "beam.K" will give you access to the matrix K referenced above
beam = EulerBernoulli(numPts, length, radius, E)


numObs = 60
B = np.zeros((numObs,numPts))

obsInds = random.sample(range(numPts), numObs)
for i in range(numObs):
    B[i,obsInds[i]] = 1.0

obsMat = np.linalg.solve(beam.K,B.T).T


priorVar = 10*10
priorLength = 0.5
priorNu = 3.0/2.0 # must take the form N+1/2 for zero or odd N (i.e., {0,1,3,5,...})

kern1 = ma.MaternKernel(1, 1.0, priorLength, priorNu)
kern2 = ma.ConstantKernel(1, 10*10)
kern = kern1 + kern2

mu = ma.ZeroMean(1,1)
priorGP = ma.GaussianProcess(mu,kern)
q = priorGP.Sample(x)[0,:]

A = np.linalg.solve(beam.K, B.T).T


# Open an HDF5 file for saving
fObs = h5py.File('ProblemDefinition.h5')
fTrue = h5py.File('FullProblemDefinition.h5')

fTrue['/ForwardModel/Loads'] = q
fTrue['/ForwardModel/NodeLocations'] = x
fTrue['/ForwardModel/Modulus'] = E
fTrue['/ForwardModel/SystemMatrix'] = beam.K

fTrue['/Observations/ObservationMatrix'] = B 
fTrue['/Observations/ObservationData'] = np.dot(A,q)
fTrue['/Observations/LoadSum'] = np.sum(q)*np.ones(1)
fTrue['/Observations/LoadPoint'] = np.sum(q[-1])*np.ones(1)

fObs['/ForwardModel/NodeLocations'] = x
fObs['/ForwardModel/Modulus'] = E
fObs['/ForwardModel/SystemMatrix'] = beam.K

fObs['/Observations/ObservationMatrix'] = B 
fObs['/Observations/ObservationData'] = np.dot(A,q)
fObs['/Observations/LoadSum'] = np.sum(q)*np.ones(1)
fObs['/Observations/LoadPoint'] = np.sum(q[-1])*np.ones(1)

print(q)


