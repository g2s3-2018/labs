
# Inferring material properties of a cantilevered beam

Recall the beam from last week. Previously, we fixed the material properties and inferred the applied load.  Here we flip the problem, the loads will be fixed and we will infer the material properties.

![png](BeamDrawing.png)

### Formulation:

Again, let $u(x)$ denote the vertical deflection of the beam and let $m(x)$ denote the vertial force acting on the beam at point $x$ (positive for upwards, negative for downwards).  We assume that the displacement can be well approximated using Euler-Bernoulli beam theory and thus satisfies the PDE
$$
\frac{\partial^2}{\partial x^2}\left[ \exp[m(x)] \frac{\partial^2 u}{\partial x^2}\right] = f(x),
$$
where $E(x)=\exp[m(x)]$ is an effective stiffness that depends both on the beam geometry and material properties.  Our goal is to infer $m(x)$ given a few point observations of $u(x)$ and a known load $f(x)$.

The same cantilever boundary conditions are used as before.  These take the form
$$
u(x=0) = 0,\quad \left.\frac{\partial u}{\partial x}\right|_{x=0} = 0
$$
and
$$
\left.\frac{\partial^2 u}{\partial x^2}\right|_{x=L} = 0, \quad  \left.\frac{\partial^3 u}{\partial x^3}\right|_{x=L} = 0.
$$


We assume that $m(x)$ is piecwise constant over $P$ nonoverlapping intervals on $[0,L]$.  More precisely,
$$
m(x) = \sum_{i=1}^P m_i \,I\left(x\in [a_i, a_{i+1})\right),
$$
where $I(\cdot)$ is an indicator function.  

#### Prior
For the prior, we assume each value is an independent normal random variable
$$
m_i \sim N(\mu_i, \sigma_i^2).
$$

#### Likelihood
Let $N_x$ denote the number of finite difference nodes used to discretize the Euler-Bernoulli PDE above.  For this problem, we will have observations of the solution $u(x)$ at $N_y$ of the finite difference nodes.  Let $u\in\mathbb{R}^{N_x}$ (without the $(x)$) denote a vector containing the finite difference solution and let $y\in\mathbb{R}^{N_y}$ denote the observable random variable, which is the solution $u$ at $N_y$ nodes plus some noise $\epsilon$, i.e.
$$
y = Bu + \epsilon,
$$
where $\epsilon \sim N(0, \Sigma_y)$.  The solution vector $u$ is given by
$$
u = [K(m)]^{-1}f,
$$
where $K$ represents the discretization of the Euler-Bernoulli PDE as a function of $m$.  Combining this with the definition of $y$, we have the complete forward model
$$
y = B[K(m)]^{-1} f + \epsilon
$$

The likelihood function then takes the form:
$$
p(y | m) = N\left(\, B [K(m)]^{-1} f,\,\,\Sigma_y \,\right) 
$$


#### Posterior
Evaluating the posterior, which is simply written as 
$$
p(m|y) \propto p(y|m)p(m),
$$
involves several steps.  The computational graph below highlights the steps necessary to evaluate the posterior.  This graph is also similar to the one we will construct with MUQ below.

![png](PosteriorGraphStart.png)

## Imports


```python
import sys
sys.path.insert(0,'/home/fenics/Installations/MUQ_INSTALL/lib')

import pandas as pd

from IPython.display import Image

import numpy as np
import matplotlib.pyplot as plt
import h5py

# Import forward model class
from BeamModel import EulerBernoulli

# MUQ Includes
import pymuqModeling as mm # Needed for Gaussian distribution
import pymuqApproximation as ma # Needed for Gaussian processes
import pymuqSamplingAlgorithms as ms # Needed for MCMC

```

## Load the data and finite difference model


```python
f = h5py.File('ProblemDefinition.h5','r')

x = np.array( f['/ForwardModel/NodeLocations'] )
B = np.array( f['/Observations/ObservationMatrix'] )
obsData = np.array( f['/Observations/ObservationData'] )

length = f['/ForwardModel'].attrs['BeamLength']
radius = f['/ForwardModel'].attrs['BeamRadius']

loads = np.array( f['/ForwardModel/Loads'])

numObs = obsData.shape[0]
numPts = x.shape[1]
dim = 1

```

### Define the material property intervals


```python
numIntervals = 3
endPts = np.linspace(0,1,numIntervals+1)
intervals = [(endPts[i],endPts[i+1]) for i in range(numIntervals)]
```

## Define the prior


```python
logPriorMu = 10*np.ones(numIntervals)
logPriorCov = 4.0*np.eye(numIntervals)

logPrior = mm.Gaussian(logPriorMu, logPriorCov).AsDensity()
```

## Define the forward model


```python
class LumpedToFull(mm.PyModPiece):
    
    def __init__(self, intervals, pts):
        """ 
        INPUTS:
          - Intervals is a list of tuples containing the intervals that define the 
            parameter field. 
          - pts is a vector containing the locations of the finite difference nodes
        """
        mm.PyModPiece.__init__(self, [len(intervals)], # One input containing lumped params
                                     [pts.shape[1]]) # One output containing the full E field
        
        # build a vector of indices mapping an index in the full vector to a continuous parameter
        self.vec2lump = np.zeros(pts.shape[1], dtype=np.uint)
        for i in range(len(intervals)):
            self.vec2lump[ (pts[0,:]>=intervals[i][0]) & (pts[0,:]<intervals[i][1]) ] = i
            
            
    def EvaluateImpl(self, inputs):
        """
        - inputs[0] will contain a vector with the values m_i in each interval
        - Needs to compute a vector "mField" containing m at each of the finite difference nodes
        - Should set the output using something like "self.outputs = [ mField ]"
        """
        self.outputs = [ inputs[0][self.vec2lump] ]
```


```python
mField      = LumpedToFull(intervals, x )
expmVals    = mm.ExpOperator(numIntervals)
loadPiece   = mm.ConstantVector(loads)
obsOperator = mm.DenseLinearOperator(B)
```


```python
# EulerBernoullie is a child of ModPiece with two inputs:
# 1. A vector of loads at each finite difference node
# 2. A vector containing the material property (exp(m(x))) at each finite difference node
beamModel = EulerBernoulli(numPts, length, radius)
```

## Likelihood function


```python
noiseVar = 1e-4
noiseCov = noiseVar*np.eye(obsData.shape[0])
likelihood = mm.Gaussian(obsData, noiseCov).AsDensity()
```

## Posterior


```python

posteriorPiece = mm.DensityProduct(2)
mPiece = mm.IdentityOperator(numIntervals)
```


```python
graph = mm.WorkGraph()

# Forward model nodes and edges
graph.AddNode(mPiece, "m_i")
graph.AddNode(expmVals, "exp(m_i)")
graph.AddNode(mField, "exp(m(x))")
graph.AddNode(loadPiece, "f")
graph.AddNode(obsOperator, "B")
graph.AddNode(beamModel, "u")

graph.AddEdge("m_i", 0, "exp(m_i)", 0)
graph.AddEdge("exp(m_i)", 0, "exp(m(x))", 0)
graph.AddEdge("exp(m(x))", 0, "u", 1)
graph.AddEdge("f", 0, "u", 0)
graph.AddEdge("u", 0, "B", 0)

# Other nodes and edges
graph.AddNode(likelihood, "Likelihood")
graph.AddNode(logPrior, "Prior")
graph.AddNode(posteriorPiece,"Posterior")

graph.AddEdge("B", 0, "Likelihood", 0)
graph.AddEdge("m_i", 0, "Prior", 0)
graph.AddEdge("Prior",0,"Posterior",0)
graph.AddEdge("Likelihood",0, "Posterior",1)
```


```python
graph.Visualize("PosteriorGraph.png")
Image(filename='PosteriorGraph.png') 
```




![png](EulerBernoulli_Lumped_files/EulerBernoulli_Lumped_17_0.png)




```python
problem = ms.SamplingProblem(graph.CreateModPiece("Posterior"))
```


```python
proposalOptions = dict()
proposalOptions['Method'] = 'AMProposal'
proposalOptions['ProposalVariance'] = 1e-2
proposalOptions['AdaptSteps'] = 100
proposalOptions['AdaptStart'] = 1000
proposalOptions['AdaptScale'] = 0.1

kernelOptions = dict()
kernelOptions['Method'] = 'MHKernel'
kernelOptions['Proposal'] = 'ProposalBlock'
kernelOptions['ProposalBlock'] = proposalOptions

options = dict()
options['NumSamples'] = 50000
options['ThinIncrement'] = 1
options['BurnIn'] = 10000
options['KernelList'] = 'Kernel1'
options['PrintLevel'] = 3
options['Kernel1'] = kernelOptions

mcmc = ms.SingleChainMCMC(options,problem)
```


```python
startPt = 10.0*np.ones(numIntervals)
samps = mcmc.Run(startPt)
```

    Starting single chain MCMC sampler...
      10% Complete
        Block 0:
          Acceptance Rate = 15%
      20% Complete
        Block 0:
          Acceptance Rate = 19%
      30% Complete
        Block 0:
          Acceptance Rate = 21%
      40% Complete
        Block 0:
          Acceptance Rate = 22%
      50% Complete
        Block 0:
          Acceptance Rate = 23%
      60% Complete
        Block 0:
          Acceptance Rate = 23%
      70% Complete
        Block 0:
          Acceptance Rate = 24%
      80% Complete
        Block 0:
          Acceptance Rate = 24%
      90% Complete
        Block 0:
          Acceptance Rate = 24%
      100% Complete
        Block 0:
          Acceptance Rate = 24%
    Completed in 8.98164 seconds.



```python
ess = samps.ESS()
print('Effective Sample Size = \n', ess)

sampMean = samps.Mean()
print('\nSample mean = \n', sampMean)

sampCov = samps.Covariance()
print('\nSample Covariance = \n', sampCov)

mcErr = np.sqrt( samps.Variance() / ess)
print('\nEstimated MC error in mean = \n', mcErr)
```

    Effective Sample Size = 
     [ 89.34893698  86.3766815   53.71001692]
    
    Sample mean = 
     [ 9.26056269  9.26425629  9.18006083]
    
    Sample Covariance = 
     [[ 0.00357601  0.00299626  0.00287603]
     [ 0.00299626  0.00302211  0.00264338]
     [ 0.00287603  0.00264338  0.01088799]]
    
    Estimated MC error in mean = 
     [ 0.00632637  0.00591503  0.01423791]



```python
sampMat = samps.AsMatrix()
```

## Plot the posterior samples


```python
plt.plot(sampMat.T)
plt.show()
```


![png](EulerBernoulli_Lumped_files/EulerBernoulli_Lumped_24_0.png)



```python
plt.figure(figsize=(12,12))
df = pd.DataFrame(sampMat.T, columns=['$E_%d$'%i for i in range(numIntervals) ])

pd.plotting.scatter_matrix(df, diagonal='kde', alpha=0.5)
plt.show()
```


    <matplotlib.figure.Figure at 0x7f0760fbe898>



![png](EulerBernoulli_Lumped_files/EulerBernoulli_Lumped_25_1.png)


## Plot samples of the posterior predictive


```python
predSamps = np.zeros((numPts, sampMat.shape[1]))

predModel = graph.CreateModPiece("u")

for i in range(sampMat.shape[1]):
    predSamps[:,i] = predModel.Evaluate([ sampMat[:,i] ])[0]
    
```


```python
# Plot quantiles
plt.fill_between(x[0,:], 
                 np.percentile(predSamps,1,axis=1),
                 np.percentile(predSamps,99,axis=1),
                 edgecolor='none', label='1%-99% CI')

# Plot the observations
obsInds = np.where(B>0)[1]
plt.plot(x[0,obsInds], obsData, 'xk', markerSize=10, label='Observations')

plt.legend()
plt.xlabel('Location x')
plt.ylabel('Displacement u')
plt.show()
```


![png](EulerBernoulli_Lumped_files/EulerBernoulli_Lumped_28_0.png)



```python
fig = plt.figure(figsize=(12,6))

for i in range(numIntervals):
    shiftedSamp = sampMat[i,:]-np.mean(sampMat[i,:])
    corr = np.correlate(shiftedSamp, shiftedSamp, mode='full')
    plt.plot(corr[int(corr.size/2):]/np.max(corr), label='Dimension %d'%i)
    
maxLagPlot = 1500
plt.plot([-maxLagPlot,0.0],[4.0*maxLagPlot,0.0],'--k', label='Zero')

plt.xlim([0,maxLagPlot])
plt.ylim([-0.1,1.1])
plt.legend()
plt.show()
```


![png](EulerBernoulli_Lumped_files/EulerBernoulli_Lumped_29_0.png)

