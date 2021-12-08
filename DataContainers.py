#!/usr/bin/env python
# coding: utf-8

# ### Stock Imports

# In[ ]:


import sys, os


# In[ ]:


import numpy as np
import pandas as pd
import scipy as sp


# In[ ]:


deg = np.pi/180


# ### Custom Imports

# In[ ]:


from UtilityMath import (find_nearest_index)


# # Data Containers
# For this project, we will have a variety of results: theoretical, simulation, 
# experimental.  These results can take the form of complex valued S-Parameters
# or power measurements.  They can be spectroscopic
# traces or they can be monochromatic.  It's a lot of possible combinations.
# 
# The goal of the functions contained herein is to create a common interface for
# all of these different types of results so that they can be compared and plotted
# together.  For instance, we might have S-Params for a simulation and want to deduce
# what would it would look like if they were interferred as a power measurement so
# it can be compared to this physical measurement.
# 
# As an additional complication, some of these results might have multiplicative offsets 
# (either complex valued or power magnitude; either spectroscopic or scalar).  These are
# considered "understandable" and can be removed by applying scale-factors (sf) 
# to the results.  These data containers will also allow us to apply these scale-factors
# to the results within the data containers and then reset them as needed.

# ## Nomenclature
# Within this code, I'm going to use the following conventions:
# * S; Complex-Valued Scattering Parameter
# * T = abs(S)^2; the power transmitted in an S-Parameter
# Within this convention, `'S41'` can be used to address S41 and `'T41'` would be used to address $abs(S_{41})^2$.
# 
# Additionally, many of our measurements are actually interferometric.  For instance, we can imagine combining
# the outputs at ports 4 and 6 when port 1 is illuminated.  Assuming ideal combiners and symmetric waveguides, this
# would be $c\ (S_{41} + S_{61})$ where $c$ is some complex scale factor that captures the unknown waveguide lengths.
# We will refer to this as `'S31_S61'`.  In the case of a power measurement, it would $c\ |S_{41} + S_{61}|^2$ and
# be represented by `'T31_T61'`.  In some case, we're comparing to a specially designed "reference waveguide" which 
# would be `'P31_R'`.

# ## Theoretical Target Values

# In[ ]:


# Old?
# class TargSParams:

#     def __init__(self, kernel, wls):
#         self.kernel = kernel
#         self.wls = np.array(wls)
#         self.rn, self.rc = kernel.shape

#     def __getitem__(self, rc):
#         if len(rc) == 2:
#             r, c = rc
#             val = self.kernel[r-1-self.rn, c-1]
#             return np.full(len(self.wls), val)
#         elif rc == 'wls':
#             return self.wls
#         else:
#             print("I don't know what you want")


# In[ ]:


K1Target = np.array(
    [[0.400*np.exp(1j*(   0)*deg), 0.311*np.exp(1j*(  40)*deg), 0.222*np.exp(1j*(  80)*deg)],
     [0.400*np.exp(1j*( 120)*deg), 0.311*np.exp(1j*( 160)*deg), 0.222*np.exp(1j*(-160)*deg)],
     [0.400*np.exp(1j*(-120)*deg), 0.311*np.exp(1j*( -80)*deg), 0.222*np.exp(1j*( -40)*deg)]
    ])


# In[ ]:


K2Target = np.array(
    [[0.467*np.exp(1j*(  26.34)*deg), 0.585*np.exp(1j*( 140.39)*deg), 0.402*np.exp(1j*( -19.92)*deg)],
     [0.614*np.exp(1j*(  34.34)*deg), 0.425*np.exp(1j*( -66.30)*deg), 0.407*np.exp(1j*( -131.93)*deg)],
     [0.358*np.exp(1j*(  14.63)*deg), 0.446*np.exp(1j*(  17.06)*deg), 0.629*np.exp(1j*(  74.63)*deg)]
    ])


# In[ ]:


K3Target = np.array(
    [[0.5494*np.exp(1j*(  26.34)*deg), 0.6882*np.exp(1j*( 140.39)*deg), 0.4729*np.exp(1j*( -19.92)*deg)],
     [0.7224*np.exp(1j*(  34.34)*deg), 0.5000*np.exp(1j*( -66.30)*deg), 0.4788*np.exp(1j*( -131.93)*deg)],
     [0.4212*np.exp(1j*(  14.63)*deg), 0.5247*np.exp(1j*(  17.06)*deg), 0.7400*np.exp(1j*(  74.63)*deg)]
    ])


# In[ ]:


K4Target = np.array(
    [[0.606*np.exp(1j*(  44.99)*deg), 0.483*np.exp(1j*( 165.15)*deg)],
     [0.483*np.exp(1j*(  26.71)*deg), 0.606*np.exp(1j*( -33.13)*deg)]
    ]).conj()


# In[ ]:


targDict = {'K1':K1Target, 'K2':K2Target, 'K3':K3Target.conj(), 'K4':K4Target}


# In[ ]:


class TargSParams:
    def __init__(self, kernel='K1', SRef=(1.+0.j), WL=1.525):
        transArray = targDict[kernel]
        zArray = np.zeros_like(transArray)
        self.SDataRaw = np.block([[zArray, transArray.T],
                          [transArray, zArray]])
        self.SData = self.SDataRaw.copy()
        self.nR, self.nC = self.SData.shape
        self.wl = WL
        self.SRef = SRef

    def getSVal(self, r, c):
        return self.SData[r-1, c-1]

    def getTVal(self, r, c):
        return np.abs(self.SData[r-1, c-1])**2

    def getSTransPart(self):
        return self.SData[(self.nR//2):, :(self.nC//2) ]

    def getTTransPart(self):
        return np.abs(self.getSTransPart())**2
    
    def applyCorrectionFactor(self, CF):
        self.SData = CF * self.SDataRaw

    def resetCorrectionFactor(self):
        self.SData = self.SDataRaw
    
    def getSRef(self):
        return self.SRef
        
    def getMeasurement(self, key, verbose=False):
        """
        key is expected to be of the form:
        S61, P61, P61_R, P41_P61.
        """
        measType = key[0] # Let's determine the measurement type from the first lettter.  'S', 'T' or 'P'
        terms = key.split('_') # Could be interferometric.  'P41_P61' -> ['P41', 'P61']
        nTerms = len(terms)
        if verbose: print(terms)
        term0 = terms[0]   # There will always be a first term.
        if len(terms) == 2:  # If there is a second term, grap it.  Otherwise let's make one for symmetry.
            term1 = terms[1]
        else: 
            term1 = 'zero'
        dataS0 = self.getSVal(eval(term0[1]), eval(term0[2])) # Grab the SParams for the first term.
        if term1 == 'zero': # Could be zero, indicating not intererometric.  Add zeros.
            dataS1 = 0.
        elif term1 == 'R':  # Could be 'R', the reference waveguide.  Get it.
            dataS1 = self.getSValR()
        else:               # Just normal data.  Grab the SParams.
            dataS1 = self.getSVal(eval(term1[1]), eval(term1[2]))
        dataS = (dataS0 + dataS1)/nTerms  # Interfer the two terms.
        if measType == 'S':
            return (self.wl, dataS)
        elif measType == 'T':
            return (self.wl, np.abs(dataS)**2)
        else:
            raise ValueError("Unrecognized measurement type.  Should be 'S' or 'T'.")
    
    def getMeasurementAt(self, key, wl, verbose=False):
        """
        key is expected to be of the form:
        S61, P61, P61_R, P41_P61.
        """
        if(wl == self.wl):
            wl, v = self.getMeasurement(key)
            return v
        else:
            print("Wavelength doesn't match")
            return 0.


# In[ ]:


targ1 = TargSParams('K1')


# In[ ]:


targ1.getSVal(4,1)


# In[ ]:


targ1.getMeasurement('S41')


# In[ ]:


wl, sTest = targ1.getMeasurement('S41_S61')


# In[ ]:


wl, tTest = targ1.getMeasurement('T41_T61')


# In[ ]:


np.abs(sTest)**2 == tTest


# ## Simulation Results

# In[ ]:


def loadSimResults(fName):
    f = open(fName, 'r')
    # Data consist of three lines
    l1 = f.readline()
    l2 = f.readline()
    l3 = f.readline()
    f.close()

    # First line gives the dimensions of the rest of the data
    nR, nC, nWLs = np.fromstring(l1, dtype=np.uint32, sep='\t')

    # second line gives the wavelengths, which we convert to microns
    wls = (10**6)*np.fromstring(l2, dtype=np.float, sep='\t')
    assert len(wls) == nWLs, 'wavelength data has unexpected length'

    # third line gives the rest of the data.
    textLine = l3
    # Python iterprets 'j' as the imaginary unit
    textLine = textLine.replace('i', 'j')
    # Values are tab separated.  Convert to list of strings.
    textList = textLine.split(sep='\t')
    # Convert to complex values.
    numberList = [complex(w) for w in textList]
    # Convert to numpy array
    npArray = np.array(numberList)
    # Check length makes sense.
    assert len(npArray) == nC*nR*nWLs, 'data not read to expected length'
    # Reshape array
    npArray = npArray.reshape(nWLs, nR, nC)
    # There is a quirk of SParams where sometimes in a 2x2, they use the transpose.  Let's undo that here.
    if nR == 2:
        sParams = npArray
    else:
        sParams = np.transpose(npArray, (0, 2, 1))
    print(fName, ':', sParams.shape)
    return wls, sParams


# In[ ]:


os.getcwd()


# In[ ]:


wls, SParams = loadSimResults("Simulations/K4_SIM4.txt")


# In[ ]:


SParams[0, 3, 0] #S41


# In[ ]:


wls, SParams = loadSimResults("Simulations/Cal4_SIM4.txt")


# In[ ]:


SParams[0, 1, 0] #S21


# In[ ]:


class SimSParams:
    def __init__(self, fName, refFName=None):
        self.wls, self.SDataRaw = loadSimResults(fName)
        if refFName:
            self.wlsRef, self.SDataRef = loadSimResults(refFName)
        else:
            self.wlsRef, self.SDataRef = None, None
        self.SData = self.SDataRaw.copy()
        self.nWLs, self.nR, self.nC = self.SData.shape

    def getSTrace(self, r, c):
        return self.SData[:, r-1, c-1]
   
    def getTTrace(self, r, c):
        return np.abs(self.SData[:, r-1, c-1])**2

    def getSVal(self, r, c, wl):
        iWL = find_nearest_index(self.wls, wl)
        return self.SData[iWL, r-1, c-1]

    def getTVal(self, r, c, wl):
        iWL = find_nearest_index(self.wls, wl)
        return np.abs(self.SData[iWL, r-1, c-1])**2

    def getSTransPartSpec(self):
        return self.SData[:, (self.nR//2):, :(self.nC//2) ]

    def getSTransPart(self, wl):
        iWL = find_nearest_index(self.wls, wl)
        return self.SData[iWL, (self.nR//2):, :(self.nC//2) ]

    def getTTransPartSpec(self):
        return np.abs(self.SData[:, (self.nR//2):, :(self.nC//2) ])**2

    def getTTransPart(self, wl):
        iWL = find_nearest_index(self.wls, wl)
        return np.abs(self.SData[iWL, (self.nR//2):, :(self.nC//2) ])**2

    def applyCorrectionFactor(self, CF):
        zArray = np.zeros_like(CF)
        CFBig = np.block([[[zArray, CF.T],
                           [CF, zArray]]])
        self.SData =  CFBig * self.SDataRaw

    def resetCorrectionFactor(self):
        self.SData = self.SDataRaw
        
    def getSValR(self):
        return self.SDataRef[:, 2-1, 1-1] #S21 or Ref
      
    def getMeasurement(self, key, verbose=False):
        """
        key is expected to be of the form:
        S61, P61, P61_R, P41_P61.
        """
        measType = key[0] # Let's determine the measurement type from the first lettter.  'S', 'T' or 'P'
        terms = key.split('_') # Could be interferometric.  'P41_P61' -> ['P41', 'P61']
        nTerms = len(terms)
        if verbose: print(terms)
        term0 = terms[0]   # There will always be a first term.
        if len(terms) == 2:  # If there is a second term, grap it.  Otherwise let's make one for symmetry.
            term1 = terms[1]
        else: 
            term1 = 'zero'
        dataS0 = self.getSTrace(eval(term0[1]), eval(term0[2])) # Grab the SParams for the first term.
        if term1 == 'zero': # Could be zero, indicating not intererometric.  Add zeros.
            dataS1 = 0.
        elif term1 == 'R':  # Could be 'R', the reference waveguide.  Get it.
            dataS1 = self.getSValR()
        else:               # Just normal data.  Grab the SParams.
            dataS1 = self.getSTrace(eval(term1[1]), eval(term1[2]))
        dataS = (dataS0 + dataS1)/nTerms  # Interfer the terms.  Note that for two terms, this is equivalent of averaging.  For 1 term, it simply takes that term.
        if measType == 'S':
            return (self.wls, dataS)
        elif measType == 'T':
            return (self.wls, np.abs(dataS)**2)
        else:
            raise ValueError("Unrecognized measurement type.  Should be 'S' or 'T'.")
            
    def getMeasurementAt(self, key, wl, verbose=False):
        wls, vals = self.getMeasurement(key)
        F = sp.interpolate.interp1d(wls, vals, kind='quadratic')
        v = F(wl).item()
        return v


# In[ ]:


k1Sim = SimSParams("Simulations/K4_SIM4.txt", "Simulations/Cal4_Sim4.txt")


# In[ ]:


k1Sim.getMeasurement('T31_T41');


# In[ ]:


k1Sim.getMeasurement('S31_R', verbose=True)


# ## Experimental Spectroscopic

# In[ ]:


# Old?
def getExpTrace(fName):
    f = open(fName, 'r')
    # Trash the first line
    f.readline()
    # Read the rest
    text = f.read()
    # free the file
    f.close()

    # Data is comma and tab deliminated, remove commas
    text1 = text.replace(',', '')
    # Split on tabs to get a 1D list that is 2*n long
    text2 = text1.split()
    # Convert string data to numbers
    numArrayDB = [float(elem) for elem in text2]
    # Convert to a numpy array and reshape to be n x 2
    npArray = np.array(numArrayDB).reshape((-1,2))
    # Convert to linear scale in power transmission
    dbData = npArray[:, 1]
    linData = dBToLinPower(dbData)
    wls = npArray[:,0]
    return wls, linData


# In[ ]:


import glob


# In[ ]:


def getExpTrace(fName):
    wlsRaw, dataRaw = np.array(pd.read_csv(fName, header=None)).T
    wls = wlsRaw/1000
    data = dataRaw
    return wls, data


# In[ ]:


def convertFNameToKey(fName):
    """
    'Experiments\\K4\\K4_T31_n15dBm_44d5K.csv' -> 'T31'
    'Experiments\\K4\\K4_T31_R_n15d5dBm_44d5K.csv' -> 'T31_R'
    'Experiments\\K4\\K4_T31_T41_n15d5dBm_44d5K.csv' -> 'T31_T41'
    """
    parts = (fName.replace('\\', '_').split('_'))[3:-2]
    key = '_'.join(parts)
    return key


# In[ ]:





# In[ ]:


def convertFNameToDetails(fPath):
    fName = fPath.split('\\')[-1]                        # 'K4_T31_T41_n15d5dBm_44d5K.csv'
    fName2 = fName.split('.')[0]                         # 'K4_T31_T41_n15d5dBm_44d5K'
    fNameParts = fName2.split("_")                       # ['K4', 'T31', 'T41', 'n15d5dBm', '44d5K']
    kernelCode = fNameParts[0]                           # 'K4'
    measCode = "_".join(fNameParts[1:-2])                # 'T31_T41'
    tiaResCode = fNameParts[-1]                          # '44d5K'
    incPowerCode = fNameParts[-2]                        # 'n15d5dBm'
    r_TIA = eval(tiaResCode.replace('d', '.').replace('K', 'e3'))                           # 44500.0
    gc_power_dB = eval(incPowerCode.replace('dBm', '').replace('d', '.').replace('n', '-'))  # -15.5
    gc_power_mW = 10**(gc_power_dB/10)                                                       # 0.0281
    outDict = {'kernelCode': kernelCode,
               'measCode': measCode,
               'gc_power_mW': gc_power_mW,
               'R_TIA': r_TIA}
    return outDict


# In[ ]:


convertFNameToDetails('Experiments\\K4\\K4_T31_T41_n15d5dBm_44d5K.csv')


# In[ ]:


class ExpResultSpect:
    
    def __init__(self, name, n, scaleFactor=1):
        self.sf = scaleFactor
        self.GCEffCurve = 1
        self.n = n
        fNames = glob.glob('Experiments/**/'+name+'_*.csv', recursive=True)
        details =[convertFNameToDetails(fName) for fName in fNames]
        keys = [d['measCode'] for d in details]
        self.detailsDict = {key:detail for key, detail in zip(keys, details)}
        self.dataDict = {key:getExpTrace(fName) for key, fName in zip(keys, fNames)}
        print("Imported Objects:", keys)
        
    def getMeasurement(self, key):
        wls, VTrace = self.dataDict[key]
        GCEffCurve = self.GCEffCurve  # mW/mW
        PDResp =  0.8/1000  # [A/mW]
        R_TIA = self.detailsDict[key]['R_TIA']  # Ohms
        P_GC = self.detailsDict[key]['gc_power_mW']  # mW
        PIn = P_GC*GCEffCurve  # mW
        POut = VTrace/(R_TIA*PDResp)  # mW
        T = self.sf*(POut/PIn)
        return (wls, T)
    
    def getMeasurementAt(self, key, wl):
        (wls, T) = self.getMeasurement(key)
        iWL = find_nearest_index(wls, wl)
        return T[iWL]
    
    def getTTrace(self, r, c):
        (wls, T) = getMeasurement('T'+str(r)+str(c))
        return (wls, T)
        
    def getTVal(self, r, c, wl):
        (wls, T) = getMeasurement('T'+str(r)+str(c))
        iWL = find_nearest_index(wls, wl)
        return T[iWL]
    
#     def getTTransPart(self, wl):
#         iWL = find_nearest_index(self.wls, wl)
#         return np.abs(self.SData[iWL, (self.nR//2):, :(self.nC//2) ])**2
   
    def getTTransPartSpec(self):       
        n = self.n // 2
        tTransKeyArray = [['T'+str(i+1+n)+str(j+1) for j in range(n)] for i in range(n)]
        TArray = np.array([[self.dataDict[k][1] for k in row] for row in tTransKeyArray])
        return(self.sf/self.calCurve)*TArray

    def applyCorrectionFactor(self, sf):
        """
        Applies a scalar or vectorial correction factor.  For instance
        """
        self.sf = sf
        
    def resetCorrectionFactor(self):
        self.sf = 1
        
    def importSimGCEffCurve(self, fname, wlSF=1, fillValue=1):       
        calCurve = np.array(pd.read_csv(fname, sep=',', header=None))
        wlsImp, TImp = calCurve.T
        f = sp.interpolate.interp1d(wlSF*wlsImp/1000, TImp, kind='quadratic', fill_value=fillValue, bounds_error=False)
        randomDataValue = list(self.dataDict.values())[0]
        wlsNew, _ = randomDataValue
        self.GCEffCurve = f(wlsNew)
        
    def importExpGCEffCurve(self, fName):       
        wls, T = getExpTrace(fName)
        self.GCEffCurve = T**(1/2)  
    
    def resetGCEffCurve(self):
        self.GCEffCurve = 1


# In[ ]:


n = 2
PKeyArray = [['T'+str(i+1+n)+str(j+1) for j in range(n)] for i in range(n)]
PKeyArray


# In[ ]:


K4ExpSp = ExpResultSpect('K4', 4)


# ## Experimental Monochromatic

# In[ ]:


class ExpResult:
    def __init__(self, transDict, portCount, units='mV', WL=1.525, R_TIA=46700, gc_power_dBm=-15.5, scaleFactor=1):
        self.wl = WL
        if units == 'mV':
            importScaleFactor = 0.001
        elif units == 'V':
            importScaleFactor = 1        
        self.dataDict = {k:v*importScaleFactor for k,v in transDict.items()}
        self.detailsDict = {'R_TIA': R_TIA, 'gc_power_mW': 10**(gc_power_dBm/10)}
        self.sf = scaleFactor
        n = int(np.sqrt(len(transDict)))
        self.portCount = portCount
        n = portCount//2
        PKeyArray = [['T'+str(i+1+n)+str(j+1) for j in range(n)] for i in range(n)]
        transArray = np.array([[transDict[key] for key in row] for row in PKeyArray])
        zArray = np.zeros_like(transArray)
        self.TData = np.block([[zArray, transArray.T],
                               [transArray, zArray]])
        
    def getMeasurement(self, key):
        value = self.dataDict[key]
        PDResp =  0.8/1000  # [A/mW]
        R_TIA = self.detailsDict['R_TIA']  # Ohms
        P_GC = self.detailsDict['gc_power_mW']  # mW
        GCEffCurve = self.GCEffCurve
        PIn = P_GC*GCEffCurve  # mW
        POut = value/(R_TIA*PDResp)  # mW
        T = self.sf*(POut/PIn)        
        return (self.wl, T)
    
    def getMeasurementAt(self, key, wl, verbose=False):
        """
        key is expected to be of the form:
        """
        if(wl == self.wl):
            wl, v = self.getMeasurement(key)
            return v
        else:
            print("Wavelength doesn't match")
            return 0.
        
    def importExpGCEffCurve(self, fName):       
        wls, T = getExpTrace(fName)
        f = sp.interpolate.interp1d(wls, T, kind='quadratic', fill_value=0, bounds_error=False)
        Tint = f(self.wl)
        self.GCEffCurve = Tint**(1/2)  
    
    def resetGCEffCurve(self):
        self.GCEffCurve = 1        

    def getTVal(self, r, c):
        return (self.sf)*self.dataDict['T'+str(r)+str(c)]

    def getTTransPart(self):
        n = self.portCount//2
        return (self.sf)*(self.TData[n:, :n])

    def applyCorrectionFactor(self, sf):
        self.sf = sf

    def resetCorrectionFactor(self):
        self.sf = 1


# In[ ]:




