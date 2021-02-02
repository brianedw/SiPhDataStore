#!/usr/bin/env python
# coding: utf-8

# ### stock imports

# In[ ]:


import numpy as np


# In[ ]:


from scipy.optimize import minimize_scalar, minimize


# # Utility Functions

# ## Lin to dB Conversions

# In[ ]:


def dBToLinPower(dbData):
    return 10**(dbData/10)


# In[ ]:


dBToLinPower(-3)


# ## Array Approximation

# In[ ]:


def find_nearest(array, value):
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[ ]:


find_nearest([0,2,4,6,8], 4.2)


# In[ ]:


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# In[ ]:


find_nearest_index([0,2,4,6,8], 4.2)


# ## Random Matrix Generation

# In[ ]:


def RandomComplexGaussianMatrix(sigma, size):
    """
    Generates a matrix random complex values where each value is
    within a circle of radius `r`.  Values are evenly distributed
    by area.
    """
    reMat = np.random.normal(0, sigma, size=size)
    imMat = np.random.normal(0, sigma, size=size)
    cMat = reMat + 1j*imMat
    return cMat


# In[ ]:


RandomComplexGaussianMatrix(0.1, (2,3))


# ## Matrix Difference

# In[ ]:


m1 = RandomComplexGaussianMatrix(1, (2,2))
z = np.exp(1j*0.1)
m2 = z*m1 + RandomComplexGaussianMatrix(0.1, (2,2))


# In[ ]:


def matrixDiffMag(m1, m2):
    """
    Computes the magnitude of the difference between two complex matrices
    """
    return (abs(m1 - m2)**2).sum()


# In[ ]:


matrixDiffMag(m1, m2)


# In[ ]:


def matrixMagDiffMag(m1, m2):
    """
    Computes the magnitude of the difference between two the magnituds of two complex matrices
    """
    return (abs(np.abs(m1) - np.abs(m2))**2).sum()


# In[ ]:


matrixMagDiffMag(m1, m2)


# In[ ]:


def matrixDiffVarPh(ph, m1, m2):
    """
    Computes the magnitude of the difference between two complex matrices
    where one one is rotated by ph.
    """
    return (abs(m1 - np.exp(1j*ph)*m2)**2).sum()


# In[ ]:


matrixDiffVarPh(-0.1, m1, m2)


# In[ ]:


def matrixDiffVarPhase(m1, m2):
    """
    Computes the magnitude of the difference between two complex matrices
    where one one is rotated by ph to find the minimul distance.
    """
    soln = minimize_scalar(matrixDiffVarPh, args=(m1, m2), bounds=(-np.pi, np.pi), method='bounded')
    val = soln.fun
    return val


# In[ ]:


matrixDiffVarPhase(m1, m2)


# ## Fitted Deembedding

# In[ ]:


deg = 2*np.pi/360


# In[ ]:


round2 = lambda a: np.round(a, 2)


# In[ ]:


def makeCFUniform(theta1, n):
    """
    Makes a phase rotation matrix where all channel combinations have the same
    rotation.
    """
    cf1 = np.exp(1j*(theta1)*deg)
    CF1 = np.full((n,n), cf1)
    CF = CF1 * CF1.T
    return CF


# In[ ]:


round2(makeCFUniform(30, 3))


# In[ ]:


def makeCFSymmetrical(thetaSequence):
    """
    Makes a phase rotation matrix where all channel combinations are not equal,
    but still display symmetry.  For instance, in a 6 port system (3 in, 3 out),
    ports [1, 3, 4, 6] would be assumed to have one phase offset, while [2, 5]
    would be have another.  This provides two degrees of freedom.
    1 -|    |- 4
    2--|    |--5
    3 -|    |- 6
    """
    thetaHalfArray = np.array(thetaSequence)
    thetaArray = np.concatenate((thetaHalfArray, thetaHalfArray[-2::-1])).reshape((1,-1))
    CF1 = np.exp(1j*thetaArray*deg)
    CF = CF1 * CF1.T
    return CF


# In[ ]:


round2(makeCFSymmetrical([30, 20]))


# In[ ]:


def findSF(pMatA, pMatB):
    def f(sf):
        error = matrixMagDiffMag(pMatA*sf, pMatB)
        return error
    soln = minimize(f,[0])
    arr1D = soln.x
    sf = arr1D[0]
    return sf


# In[ ]:


m1 = RandomComplexGaussianMatrix(1, (2,2))
m2 = 0.9*m1
findSF(m1, m2)


# In[ ]:


def findSingleRotCF(matA, matB):
    def f(arr1D):
        (theta1,) = arr1D
        (n,n) = matA.shape
        CF = makeCFUniform(theta1, n)
        error = matrixDiffMag(matA*CF, matB)
        return error
    soln = minimize(f,[0])
    arr1D = soln.x
    theta1 = arr1D[0]
    print("rotation angle:", theta1)
    (n,n) = matA.shape
    return makeCFUniform(theta1, n)


# In[ ]:


m1 = RandomComplexGaussianMatrix(1, (2,2))
m2 = np.exp(1j*20*deg)*m1
findSingleRotCF(m1, m2)


# In[ ]:


a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
a2 = a1 * makeCFUniform(13., 3)
CF = findSingleRotCF(a1, a2)
matrixDiffMag(a1*CF, a2)


# In[ ]:


def findDoubleRotCF(matA, matB):
    # arr1D = [theta1, theta2]
    def f(arr1D):
        (theta1, theta2) = arr1D
        CF = makeCFSymmetrical(arr1D)
        error = matrixDiffMag(matA*CF, matB)
        return error
    soln = minimize(f,[0,0])
    (theta1, theta2) = soln.x
    print("rotation angles:", theta1, theta2)
    return makeCFSymmetrical(soln.x)


# In[ ]:


a1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
a2 = a1 * makeCFSymmetrical([13.,87.])
CF = findDoubleRotCF(a1, a2)
matrixDiffMag(a1*CF, a2)


# In[ ]:




