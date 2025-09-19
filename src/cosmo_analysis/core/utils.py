# HELPING FUNCTIONS

# TODO: Description of file for documentation

# TODO: Description of functions for documentation

import bisect as bsct
import h5py

# Get the index of the closest but bigger element in a list than your value 
def getClosestIdx(myList, myNumber):
        return bsct.bisect_left(myList, myNumber)

# Get the idx of maximum number in iterable
def maxIdx(arr):
    maxAux = 0
    for i,el in enumerate(arr):
        if el > arr[maxAux]:
            maxAux = i
    return maxAux

# Get the n maximum numbers idx in iterable (ONLY SEND COPIES)
def findNmax(arr, N):
    minV = min(arr)
    maxes = [0]*N
    for i in range(N):
        max1 = maxIdx(arr)
        maxes[i] = max1
        arr[max1] = minV
    return maxes

# Print a hdf5 file structure
def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')
def h5print(filename):
    h5printR(filename, '  ')

# Pads a number with 3 zeros (default) and makes it a string
def padN(n,pad=3):
    return str(n).zfill(pad)

# Transformations between scale factor and redshift
def getZ(a):
    return 1/a-1
def getA(z):
    return 1/(z+1)

# Gets the idx of the snapshot with the time closest to the one inputted
def tToIdx(sim,time):
    dist = abs(sim.snap[0].time-time)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].time-time)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx

# Gets the idx of the snapshot with the redshift closest to the one inputted
def zToIdx(sim,z):
    dist = abs(sim.snap[0].z-z)
    idx = 0
    for i in range(len(sim.snap)):
        nDist = abs(sim.snap[i].z-z)
        if nDist < dist:
            dist = nDist
            idx = i
    return idx
