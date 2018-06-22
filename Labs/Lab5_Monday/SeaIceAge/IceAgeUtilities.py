import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def NormalizeProbs(logProbs):
    """ Normalizes each column of a matrix containing log probabilities. """
    for i in range(logProbs.shape[1]):
        logProbs[:,i] = logProbs[:,i] - np.log(np.sum(np.exp(logProbs[:,i])))
    
def ReadIceAgeOneYear(year, week):
    ## USER Required Input
    ageFile = 'data/iceage.grid.week.%04d.%02d.n.v3.bin'%(year,week)
    latFile = 'data/Na12500-CF_latitude.dat'
    lonFile = 'data/Na12500-CF_longitude.dat'
    knownShape = (722,722)

    ## Read the files
    f = open(ageFile, 'r')
    ages = np.fromfile(f, dtype='<u1').reshape(knownShape)
    f = open(latFile, 'r')
    lat = np.fromfile(f, dtype='<f').reshape(knownShape)
    f = open(lonFile, 'r')
    lon = np.fromfile(f, dtype='<f').reshape(knownShape)

    ages[(ages>25)&(ages<100)] = 25
    
    return lat, lon, ages


def ReadIceAges(thinBy, numClasses):

    lat = np.array([])
    lon = np.array([])
    ages = np.array([])
    time = np.array([])

    for year in range(1984,2018):
        currLat, currLon, currAges = ReadIceAgeOneYear(year,1)

        currLat = currLat.ravel()[0::thinBy]
        currLon = currLon.ravel()[0::thinBy]
        currAges = currAges.ravel()[0::thinBy]

        keepInds = (currAges>=0)&(currAges<100)&(currLat>60)
        time = np.append(time, year*np.ones(np.sum(keepInds)))

        lat = np.append(lat, currLat[keepInds])
        lon = np.append(lon, currLon[keepInds])
        ages = np.append(ages, currAges[keepInds])

    ages /= 5

    # Squish all older ages into a single age
    ages[ages>numClasses-1] = numClasses-1
    
    return time, lat, lon, ages

def PlotAgeScatter(year, time, lat, lon, ages):

    map = Basemap(projection='npstere',boundinglat=55, lon_0=0)

    map.fillcontinents(color='gray')
    map.drawcoastlines()

    x, y = map(lon[time==year], lat[time==year])

    #map.scatter(lon[~mask].ravel()[0::thinBy],lat[~mask].ravel()[0::thinBy],latlon=True)
    plt.scatter(x,y, marker='.',c=ages[time==year],alpha=0.5,edgecolors='none')
    plt.title('Ice ages Jan 1-7, %d'%year)