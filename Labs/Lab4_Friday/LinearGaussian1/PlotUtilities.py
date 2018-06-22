import matplotlib.pyplot as plt
import numpy as np


def PlotGaussianPDF(mean, cov):
    dim = mean.shape[0]
    
    fig, axs = plt.subplots(dim,dim, figsize=(10,10))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    
    for i in range(dim):
        for j in range(i):
            
            subMean = mean[[i,j]]
            subCov = np.array([ [cov[i,i],cov[i,j]], [cov[j,i],cov[j,j]] ])
            PlotGaussianMarginal2D(axs[i,j], subMean, subCov)
            
            if(j>0):
                axs[i,j].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
            if(i<dim-1):
                axs[i,j].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
                
            axs[i,j].tick_params(labelsize=8)
            
    for i in range(dim):
        axs[i,i].tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
        
        axs[i,i].set_title('$m_%d$'%i)
        if(i==dim-1):
            axs[i,i].tick_params(labelsize=8)
        else:
            axs[i,i].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
            
            
        PlotGaussianMarginal1D(axs[i,i], mean[i], cov[i,i])
        
        for j in range(i+1,dim):
            axs[i,j].axis('off')
    
    
def PlotGaussianMarginal1D(ax, mean, cov):
    
    numPlot = 40
    std = np.sqrt(cov)
    xs = np.linspace(mean-2.5*std, mean+2.5*std, numPlot)
    
    logPdf = -0.5*((xs-mean)/std)**2
    ax.plot(xs,np.exp(logPdf), linewidth=3)
    ax.set_ylim([0,1.1*np.max(np.exp(logPdf))])
    
def PlotGaussianMarginal2D(ax, mean, cov):
    dim = mean.shape[0]
    assert dim == 2
    std = np.sqrt(np.diag(cov))
    
    numPlot = 40
    xs = np.linspace(mean[0]-2.5*std[0], mean[0]+2.5*std[0], numPlot)
    ys = np.linspace(mean[1]-2.5*std[1], mean[1]+2.5*std[1], numPlot)
    [X,Y] = np.meshgrid(xs,ys)
    
    xs = X.reshape((1,numPlot*numPlot))
    ys = Y.reshape((1,numPlot*numPlot))
    
    locs = np.vstack((xs,ys))
    
    logPdf = np.zeros(numPlot*numPlot)
    for i in range(numPlot*numPlot):
        logPdf[i] = -0.5*np.dot( (locs[:,i]-mean).T, np.linalg.solve(cov, (locs[:,i]-mean)))
    
    logPdf = logPdf.reshape((numPlot,numPlot))
    ax.contour(X,Y,np.exp(logPdf))
    
    