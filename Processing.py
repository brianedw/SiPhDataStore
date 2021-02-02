#!/usr/bin/env python
# coding: utf-8

# ## Stock Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import shutil
import os
import sys
import re


# In[ ]:


import numpy as np
import scipy
import PIL


# In[ ]:


from scipy.optimize import minimize_scalar, minimize


# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Text
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge

print("Bokeh Version:", bokeh.__version__)

output_notebook()
bokeh.io.curdoc().theme = 'dark_minimal'


# In[ ]:


from bokeh.models import tickers


# In[ ]:


import bokeh.palettes as palettes


# In[ ]:


deg = (2*np.pi)/360


# ## Custom Imports

# In[ ]:


from DataContainers import (TargSParams, SimSParams, ExpResultSpect, ExpResult)


# In[ ]:


from UtilityMath import (findSingleRotCF,)


# In[ ]:


from UtilityPlotting import (makePolarPlot, addMatrixDiff)


# # Work

# ## Theoretical Kernel Definitions

# ## Function Library

# ### Plotting

# #### Phase Error Simulation Plot

# In[ ]:


def makePhErrorSimPlot(KName, KTarg, KSim):
    KTarget = KTarg.getSTransPart()
    p = figure(plot_width=800, plot_height=400, title=KName + " Sim Error vs Wavelength", x_range=[1.4, 1.6], y_range=[-0.1, 6.1])

    errorsComp = [matrixDiffMag(k, KTarget) for k in KSim.getSTransPart()]
    errorsMag = [matrixMagDiffMag(k, KTarget) for k in KSim.getSTransPart()]
    errorsMagVarPh = [matrixDiffVarPhase(k, KTarget) for k in KSim.getSTransPart()]

    p.line(x=KSim.wls, y=errorsComp, line_color=palettes.Category10[9][0], legend_label="Complex Error")
    p.line(x=KSim.wls, y=errorsMag, line_color=palettes.Category10[9][1], legend_label="Mag Error")
    p.line(x=KSim.wls, y=errorsMagVarPh, line_color=palettes.Category10[9][2], legend_label="Complex Error var Phase")
    return p


# In[ ]:





# #### Dispersion Plot

# In[ ]:


def makeDispersionPlot(KName, yMax):
    p = figure(plot_width=850, plot_height=400, title=KName+' Single Power Transmissions', x_range=[1.400,1.600], y_range=[0, yMax])
    # p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
    p.xaxis.axis_label = 'wavelength (um)'
    p.yaxis.axis_label = 'T'
    return p


# In[ ]:


def addSimTraces(p, KSim, rList, tList):
    colorIndex = 0
    for t in tList:
        for r in rList:
            color = palettes.Category10[9][colorIndex]
            colorIndex += 1
            trace = KSim.getPTrace(r,t)
            p.line(KSim.wls, trace, line_color=color, line_width=2, legend_label="abs(S"+str(r)+str(t)+")^2")


# In[ ]:


def addTargDots(p, KTarg, rList, tList):
    colorIndex = 0
    for t in tList:
        for r in rList:
            color = palettes.Category10[9][colorIndex]
            colorIndex += 1
            val = KTarg.getPVal(r,t)
            p.circle([1.525], [val], size=10, color=color, fill_alpha=0)


# In[ ]:


def addExpDots(p, KExp, rList, tList):
    colorIndex = 0
    for t in tList:
        for r in rList:
            color = palettes.Category10[9][colorIndex]
            colorIndex += 1
            val = KExp.getPVal(r,t)
            p.cross([1.525], [val], size=10, color=color)


# #### Interference Dispersion Plot

# In[ ]:


def makeInterDispersionPlot(KName, yMax):
    p = figure(plot_width=850, plot_height=400, title=KName+' Interferred Power Transmissions', x_range=[1.400,1.600], y_range=[0, yMax])
    p.xaxis.axis_label = 'wavelength (um)'
    p.yaxis.axis_label = 'T'
    return p


# In[ ]:


def addSimTraces1(p, KSim, rPairs, tList):
    cIndex = 0
    for t in tVals:
        for rPair in rPairs:
            color = palettes.Category10[9][cIndex]
            cIndex += 1
            r1, r2 = rPair
            trace = np.abs(KSim.getSTrace(r1,t) + KSim.getSTrace(r2,t))**2
            p.line( KSim.wls, trace, line_color=color, line_width=2, legend_label="P"+str(r1)+str(r2)+','+str(t))


# In[ ]:


def addTargDots1(p, KTarg, rPairs, tList):
    cIndex = 0
    for t in tVals:
        for rPair in rPairs:
            (r1, r2) = rPair
            color = palettes.Category10[9][cIndex]
            cIndex += 1
            val = np.abs(KTarg.getSVal(r1, t) + KTarg.getSVal(r2, t))**2
            p.circle([1.525], [val], size=10, color=color, fill_alpha=0)


# In[ ]:


def addExpDots1(p, KExpInt, rPairs, tList):
    cIndex = 0
    for t in tVals:
        for rPair in rPairs:
            color = palettes.Category10[9][cIndex]
            cIndex += 1
            val = KExpInt.getPVal(rPair, t)
            p.cross([1.525], [val], size=10, color=color)


# #### Power Bar Plot

# In[ ]:


def MakePowerBarPlot(KTarg, KSim, KExp, rVals, tVals):
    cats = ['T'+str(r)+str(t) for r in rVals for t in tVals]
    subCats = ['targ', 'sim', 'exp']
    dodges = [-0.25, 0.0, 0.25]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    targData = KTarg.getPTransPart().flatten().tolist()
    simData = KSim.getPTransPartAtWL(1.525).flatten().tolist()
    expData = KExp.getPTransPart().flatten().tolist()

    data = {'cats' : cats,
            'targ' : targData,
            'sim' : simData,
            'exp' : expData}
    source = ColumnDataSource(data=data)

    max = np.max((targData,simData,expData))
    p = figure(x_range=cats, y_range=(0, 1.2*max), plot_width=850, plot_height = 300, 
               title="Power Comparisons", toolbar_location=None, tools="")

    for i in range(len(subCats)):
        p.vbar(x=dodge('cats', dodges[i], range=p.x_range), top=subCats[i], width=0.2, source=source,
        color=colors[i], legend_label=subCats[i])

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    return p


# In[ ]:


def MakeInterPowerBarPlot(KTarg, KSim, KExpInt, rPairs, tVals):
    cats = ['T'+str(r[0])+str(r[1])+','+str(t) for r in rPairs for t in tVals]
    subCats = ['targ', 'sim', 'exp']
    dodges =  [ -0.25,   0.0,  0.25]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    targData = [np.abs(KTarg.getSVal(r[0],t) + KTarg.getSVal(r[1],t) )**2 for r in rPairs for t in tVals]
    simData = [np.abs(KSim.getSVal(r[0],t, 1.525) + KSim.getSVal(r[1],t, 1.525) )**2 for r in rPairs for t in tVals]
    expData = [KExpInt.getPVal(r, t) for r in rPairs for t in tVals]
    
    data = {'cats' : cats,
            'targ' : targData,
            'sim' : simData,
            'exp' : expData}
    source = ColumnDataSource(data=data)

    max = np.max((targData,simData,expData))
    p = figure(x_range=cats, y_range=(0, 1.2*max), plot_width=850, plot_height = 300, 
                title="Power Comparisons", toolbar_location=None, tools="")

    for i in range(len(subCats)):
        p.vbar(x=dodge('cats', dodges[i], range=p.x_range), top=subCats[i], width=0.2, source=source,
        color=colors[i], legend_label=subCats[i])

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"
    return p


# # Application

# In[ ]:


# reloadSiPhDataStore()


# ## K1 Analysis

# In[ ]:


KSim = K1Sim
KTarget = K1Target.conj()*np.exp(1j*(20)*deg)
KName = 'K1'


# In[ ]:


ind = find_nearest_index(KSim['wls'], 1.525)
STransAtWL = KSim.getTransPart()[ind]
wl = round(KSim['wls'][ind],4)


# In[ ]:


KTarget


# In[ ]:


p = makePolarPlot(KName+' '+str(wl)+'(um)')
addMatrixDiff(p, KTarget, STransAtWL)
show(p)


# In[ ]:


errorsComp = [matrixDiffMag(k, KTarget) for k in KSim.getTransPart()]
errorsMag = [matrixMagDiffMag(k, KTarget) for k in KSim.getTransPart()]
errorsMagVarPh = [matrixDiffVarPhase(k, KTarget) for k in KSim.getTransPart()]


# In[ ]:


p = figure(plot_width=800, plot_height=400, title="Sim Error vs Wavelength", x_range=[1.4, 1.6], y_range=[-0.1, 3.1])
p.line(x=KSim.wls, y=errorsComp, line_color=palettes.Category10[9][0], legend_label="Complex Error")
p.line(x=KSim.wls, y=errorsMag, line_color=palettes.Category10[9][1], legend_label="Mag Error")
p.line(x=KSim.wls, y=errorsMagVarPh, line_color=palettes.Category10[9][2], legend_label="Complex Error var Phase")
show(p)


# In[ ]:


ff = 1.0
p = figure(plot_width=850, plot_height=400, title=KName+' Single Power Transmissions', x_range=[1.400,1.600], y_range=[0, 0.2])
# p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
p.xaxis.axis_label = 'wavelength (um)'
p.yaxis.axis_label = 'T'
for t in [1,2,3]:
    color = palettes.Category10[9][t]
    for r in [4,5,6]:
        p.line( KSim['wls'], np.abs(ff*KSim[r,t])**2, line_color=color, line_width=2, legend_label="abs(S"+str(r)+str(t)+")^2")
        val = np.abs(KTarget[r-3-1,t-1])**2
        p.circle([1.525], [val], size=10, color=color)
for t in [1,2,3]:
    color = palettes.Category10[9][t]
    for r in [4,5,6]:
        dataFName = KName+'_'+'T'+str(r)+str(t)
        try:
            wlsExp, valsExp = getExpTrace(dataFName)
            p.line( wlsExp, valsExp, line_color=color, line_width=2, line_dash='dashed')
        except:
            pass
show(p)


# In[ ]:


rPairs = [(4,5), (5,6), (4,6)]
p = figure(plot_width=850, plot_height=400, title=KName + ' Interferred Transmission Powers', x_range=[1.400,1.600], y_range=[0, 0.15])
p.xaxis.axis_label = 'wavelength (um)'
p.yaxis.axis_label = 'T'
cIndex = 0
for t in [1,2,3]:
    for rPair in rPairs:
        color = palettes.Category10[9][cIndex]
        cIndex += 1
        r1, r2 = rPair
        p.line( KSim['wls'], np.abs(KSim[r1,t] + KSim[r2,t])**2, line_color=color, line_width=2, legend_label="abs(S"+str(r1)+str(t)+" + S"+str(r2)+str(t)+")^2")
for t in [1,2,3]:
    color = palettes.Category10[9][t]
    for rPair in rPairs:
        r1, r2 = rPair
        dataFName = KName+'_'+'T'+str(r1)+str(t)+'_T'+str(r1)+str(t)
        try:
            wlsExp, valsExp = getExpTrace(dataFName)
            p.line( wlsExp, valsExp, line_color=color, line_width=2, line_dash='dashed')
        except:
            pass
show(p)


# ## K2 Analysis

# In[ ]:


KSim = K2Sim
KTarget = K2Target.conj() * np.exp(1j*0*deg)
KName = 'K2'


# In[ ]:


ind = find_nearest_index(KSim['wls'], 1.525)
STransAtWL = KSim.getTransPart()[ind]
wl = round(KSim['wls'][ind],4)


# In[ ]:


p = makePolarPlot(KName+' '+str(wl)+'(um)')
addMatrixDiff(p, KTarget, STransAtWL)
show(p)


# In[ ]:


errorsComp = [matrixDiffMag(k, KTarget) for k in KSim.getTransPart()]
errorsMag = [matrixMagDiffMag(k, KTarget) for k in KSim.getTransPart()]
errorsMagVarPh = [matrixDiffVarPhase(k, KTarget) for k in KSim.getTransPart()]


# In[ ]:


p = figure(plot_width=800, plot_height=400, title="Sim Error vs Wavelength", x_range=[1.4, 1.6], y_range=[-0.1, 6.1])
p.line(x=KSim.wls, y=errorsComp, line_color=palettes.Category10[9][0], legend_label="Complex Error")
p.line(x=KSim.wls, y=errorsMag, line_color=palettes.Category10[9][1], legend_label="Mag Error")
p.line(x=KSim.wls, y=errorsMagVarPh, line_color=palettes.Category10[9][2], legend_label="Complex Error var Phase")
show(p)


# In[ ]:


ff = 1.0
p = figure(plot_width=850, plot_height=400, title=KName+' Single Power Transmissions', x_range=[1.400,1.600], y_range=[0, 0.45])
# p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
cIndex = 0
for t in [1,2,3]:
    for r in [4,5,6]:
        color = palettes.Category10[9][cIndex]
        cIndex +=1
        p.line( KSim['wls'], np.abs(ff*KSim[r,t])**2, line_color=color, line_width=2, legend_label="abs(S"+str(r)+str(t)+")^2")
        val = np.abs(KTarget[r-3-1,t-1])**2
        p.circle([1.525], [val], size=10, color=color)
p.xaxis.axis_label = 'wavelength (um)'
p.yaxis.axis_label = 'T'
show(p)


# In[ ]:


rPairs = [(4,5), (5,6), (4,6)]
p = figure(plot_width=850, plot_height=400, title=KName + ' Interferred Transmission Powers', x_range=[1.400,1.600], y_range=[0, 0.60])
# p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
cIndex = 0
for t in [1,2,3]:
    for rPair in rPairs:
        color = palettes.Category10[9][cIndex]
        cIndex += 1
        r1, r2 = rPair
        p.line( KSim['wls'], np.abs(KSim[r1,t] + KSim[r2,t])**2, line_color=color, line_width=2, legend_label="abs(S"+str(r1)+str(t)+" + S"+str(r2)+str(t)+")^2")
p.xaxis.axis_label = 'wavelength (um)'
p.yaxis.axis_label = 'T'
show(p)


# In[ ]:





# ## K3 Analysis

# In[ ]:


K3Exp1525PD = {'T41':  55, 'T42': 107, 'T43':  63,
                'T51': 126, 'T52':  39, 'T53':  59,
                'T61':  46, 'T62':  49, 'T63': 105}


# In[ ]:


K3Exp1525PDInt = {'T45,1':110, 'T45,2':125, 'T45,3': 70,
                  'T56,1':120, 'T56,2': 36, 'T56,3':135,
                  'T46,1': 99, 'T46,2': 75, 'T46,3':121}


# In[ ]:


K3Target = np.array(
    [[0.5494*np.exp(1j*(  26.34)*deg), 0.6882*np.exp(1j*( 140.39)*deg), 0.4729*np.exp(1j*( -19.92)*deg)],
     [0.7224*np.exp(1j*(  34.34)*deg), 0.5000*np.exp(1j*( -66.30)*deg), 0.4788*np.exp(1j*( -131.93)*deg)],
     [0.4212*np.exp(1j*(  14.63)*deg), 0.5247*np.exp(1j*(  17.06)*deg), 0.7400*np.exp(1j*(  74.63)*deg)]
    ])


# In[ ]:


KSim2D = SimSParams('K3_2DEIA_SIM3.txt')
KSim = SimSParams('K3_SIM3.txt')
KTarg = TargSParams(K3Target.conj())
KExp = ExpSParams(K3Exp1525PD, 1.0)
KExpInt = ExpInterParams(K3Exp1525PDInt, 1.0)
KName = 'K3'


# In[ ]:


KSim2D.resetCorrectionFactor()
CF = findSingleRotCF(KSim2D.getSTransPartAtWL(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim2D.getSTransPartAtWL(1.525))
show(p)


# In[ ]:


p = makePhErrorSimPlot(KName, KTarg, KSim2D)
show(p)


# In[ ]:


KSim.resetCorrectionFactor()
CF = findDoubleRotCF(KSim.getSTransPartAtWL(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CF)


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim.getSTransPartAtWL(1.525))
show(p)


# In[ ]:


p = makePhErrorSimPlot(KName, KTarg, KSim)
show(p)


# In[ ]:


KSim.resetCorrectionFactor()


# In[ ]:


KExp.resetCorrectionFactor()
sf = findSF(KExp.getPTransPart(), KSim.getPTransPartAtWL(1.525))
KExp.applyCorrectionFactor(sf)


# In[ ]:


p = makeDispersionPlot(KName, 0.6)
addSimTraces(p, KSim, (4,5,6), (1,2,3))
addTargDots(p, KTarg, (4,5,6), (1,2,3))
addExpDots(p, KExp, (4,5,6), (1,2,3))
show(p)


# In[ ]:


p = MakePowerBarPlot(KTarg, KSim, KExp, (4, 5, 6), (1, 2, 3))
show(p)


# In[ ]:


sf = findSF(KExp.getPTransPart(), KSim.getPTransPartAtWL(1.525))


# In[ ]:


KExpInt.applyCorrectionFactor(0.0045)


# In[ ]:


rPairs = [(4,5), (5,6), (4,6)]
tVals = [1,2,3]
p = makeInterDispersionPlot(KName, 1.5)
addSimTraces1(p, KSim, rPairs, tVals)
addTargDots1(p, KTarg, rPairs, tVals)
addExpDots1(p, KExpInt, rPairs, tVals)
show(p)


# In[ ]:


p = MakeInterPowerBarPlot(KTarg, KSim, KExpInt, ((4,5), (5,6), (4,6)), (1,2,3))
show(p)


# ## K4

# In[ ]:


K4Exp1525PD = {'T31':  55, 'T32': 107, 
               'T41': 126, 'T42':  39,
              'T31_T41':110, 'T32_T42':125}


# In[ ]:


KTarg = TargSParams('K4')


# In[ ]:


KSim2D = SimSParams('Simulations//K4_2DEIA_SIM3.txt')
KSim = SimSParams('Simulations//K4_SIM3.txt')


# In[ ]:


KExpSpect = ExpResultSpect('K4', n=4, scaleFactor=2.5)


# In[ ]:


KExp = ExpResult(K4Exp1525PD, n=4, WL=1.525, scaleFactor=2.5)


# In[ ]:


KName = 'K4'


# In[ ]:


KSim2D.getSTransPart(1.525)


# In[ ]:


KTarg.getSTransPart()


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
show(p)


# In[ ]:


KSim2D.resetCorrectionFactor()
CF = findSingleRotCF(KSim2D.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
show(p)


# In[ ]:


# p = makePhErrorSimPlot(KName, KTarg, KSim2D)
# show(p)


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim.getSTransPart(1.525))
show(p)


# In[ ]:


KSim.resetCorrectionFactor()
CF = findSingleRotCF(KSim.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CF)


# In[ ]:


p = makePolarPlot(KName)
addMatrixDiff(p, KTarg.getSTransPart(), KSim.getSTransPart(1.525))
show(p)


# In[ ]:


# p = makePhErrorSimPlot(KName, KTarg, KSim)
# show(p)


# In[ ]:


KSim.resetCorrectionFactor()


# In[ ]:


def makeDispersionPlot(title, xRange, yRange):
    p = figure(plot_width=850, plot_height=400, title=title, x_range=xRange, y_range=yRange)
    # p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
    p.xaxis.axis_label = 'wavelength (um)'
    p.yaxis.axis_label = 'T'
    return p


# In[ ]:


def addPairedTraces(p, traceData1, traceData2, name):
    color = palettes.Category10[10][hash(name)%10]
    p.line(traceData1[0], traceData1[1], line_color=color, line_width=1, legend_label=name+" goal")
    p.line(traceData2[0], traceData2[1], line_color=color, line_width=2, legend_label=name+" exp")    


# In[ ]:


def addTrace(p, traceData, name):
    color = palettes.Category10[10][hash(name)%10]
    p.line(traceData[0], traceData[1], line_color=color, line_width=2, legend_label=name)


# In[ ]:


def addPoint(p, ptData, name):
    color = palettes.Category10[10][hash(name)%10]
    p.circle(ptData[0], ptData[1], line_color=color, fill_color=color)


# In[ ]:


nPorts = 4
n = nPorts//2
tTransKeyArray = ['T'+str(i+1+n)+str(j+1) for j in range(n) for i in range(n)]
tTransKeyArray


# In[ ]:


import itertools


# In[ ]:


def genTransLabels(nPorts):
    n = nPorts//2
    inPorts = np.linspace(1, n, num=n, endpoint=True, dtype=np.int)
    outPorts = np.linspace(1 + n, n+n, num=n, endpoint=True, dtype=np.int)
    combos = list(itertools.product(outPorts, inPorts))
    labels = ['T'+str(oP)+str(iP) for oP, iP in combos]
    return labels
genTransLabels(6)


# In[ ]:


def genInterferenceLabels(nPorts):
    n = nPorts//2
    inPorts = np.linspace(1, n, num=n, endpoint=True, dtype=np.int)
    outPorts = np.linspace(1 + n, n+n, num=n, endpoint=True, dtype=np.int)
    outPortPairs = itertools.combinations(outPorts, 2)
    combos = list(itertools.product(outPortPairs, inPorts))
    labels = ['T'+str(oP1)+str(iP)+'_T'+str(oP2)+str(iP) for ((oP1, oP2), iP) in combos]
    return labels
genInterferenceLabels(6)


# In[ ]:


KExp.applyCorrectionFactor(0.2/100.)


# In[ ]:


p = makeDispersionPlot("Standard Transmission Measurements", [1.4, 1.6], [0, 0.3])
for name in genTransLabels(4):
    addPairedTraces(p, KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
    addPoint(p, KExp.getMeasurement(name), name)
show(p)


# In[ ]:


p = makeDispersionPlot("Intra Kernel Interference", [1.4, 1.6], [0, 0.3])
for name in genInterferenceLabels(4):
    addPairedTraces(p, KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
show(p)


# In[ ]:


p = makeDispersionPlot("Reference Waveguide Measurements", [1.4, 1.6], [0, 0.3])
labels = ('T31_R', 'T42_R')
for name in labels:
    addTrace(p, KExpSpect.getMeasurement(name), name)
show(p)


# In[ ]:


KExp.resetCorrectionFactor()
sf = findSF(KExp.getPTransPart(), KSim.getPTransPartAtWL(1.525))
KExp.applyCorrectionFactor(sf)


# In[ ]:


p = makeDispersionPlot(KName, 0.6)
addSimTraces(p, KSim, (3,4), (1,2))
addTargDots(p, KTarg, (3,4), (1,2))
addExpDots(p, KExp, (3,4), (1,2))
show(p)


# In[ ]:


p = MakePowerBarPlot(KTarg, KSim, KExp, (3,4), (1, 2))
show(p)


# In[ ]:


sf = findSF(KExp.getPTransPart(), KSim.getPTransPartAtWL(1.525))


# In[ ]:


KExpInt.applyCorrectionFactor(0.0045)


# In[ ]:


rPairs = [(3,4)]
tVals = [1,2]
p = makeInterDispersionPlot(KName, 1.5)
addSimTraces1(p, KSim, rPairs, tVals)
addTargDots1(p, KTarg, rPairs, tVals)
addExpDots1(p, KExpInt, rPairs, tVals)
show(p)


# In[ ]:


p = MakeInterPowerBarPlot(KTarg, KSim, KExpInt, ((3,4),), (1,2))
show(p)


# # Scrap

# In[ ]:


getExpTrace("/content/SiPhDataStore/K4_V2/K4T31_n15dBm_44d5K.csv")


# In[ ]:


import pandas as pd
df=pd.read_csv("/content/SiPhDataStore/K4_V2/K4T31_n15dBm_44d5K.csv", sep=',',header=None)


# In[ ]:


K4T31 = np.array(df)


# ## Bar Charts

# In[ ]:




