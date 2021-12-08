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


from UtilityMath import (findSingleRotCF, findDoubleRotCF, genInterferenceLabels, genTransLabels, matrixDiffMag, matrixMagDiffMag, matrixDiffVarPhase, findSF)


# In[ ]:


from UtilityPlotting import (PolarPlot, DispersionPlot, PowerBarPlot)


# # Work

# ## Function Library

# ### Plotting

# #### Phase Error Simulation Plot

# In[ ]:


def makePhErrorSimPlot(KName, KTarg, KSim):
    KTarget = KTarg.getSTransPart()
    p = figure(plot_width=800, plot_height=400, title=KName + " Sim Error vs Wavelength", x_range=[1.4, 1.6], y_range=[-0.1, 6.1])

    errorsComp = [matrixDiffMag(k, KTarget) for k in KSim.getSTransPartSpec()]
    errorsMag = [matrixMagDiffMag(k, KTarget) for k in KSim.getSTransPartSpec()]
    errorsMagVarPh = [matrixDiffVarPhase(k, KTarget) for k in KSim.getSTransPartSpec()]

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
            trace = KSim.getTTrace(r,t)
            p.line(KSim.wls, trace, line_color=color, line_width=2, legend_label="abs(S"+str(r)+str(t)+")^2")


# In[ ]:


def addTargDots(p, KTarg, rList, tList):
    colorIndex = 0
    for t in tList:
        for r in rList:
            color = palettes.Category10[9][colorIndex]
            colorIndex += 1
            val = KTarg.getTVal(r,t)
            p.circle([1.525], [val], size=10, color=color, fill_alpha=0)


# In[ ]:


def addExpDots(p, KExp, rList, tList):
    colorIndex = 0
    for t in tList:
        for r in rList:
            color = palettes.Category10[9][colorIndex]
            colorIndex += 1
            val = KExp.getTVal(r,t)
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


# def MakePowerBarPlot(KTarg, KSim, KExp, rVals, tVals):
#     cats = ['T'+str(r)+str(t) for r in rVals for t in tVals]
#     subCats = ['targ', 'sim', 'exp']
#     dodges = [-0.25, 0.0, 0.25]
#     colors = ["#c9d9d3", "#718dbf", "#e84d60"]

#     targData = KTarg.getTTransPart().flatten().tolist()
#     simData = KSim.getTTransPart(1.525).flatten().tolist()
#     expData = KExp.getTTransPart().flatten().tolist()

#     data = {'cats' : cats,
#             'targ' : targData,
#             'sim' : simData,
#             'exp' : expData}
#     source = ColumnDataSource(data=data)

#     max = np.max((targData,simData,expData))
#     p = figure(x_range=cats, y_range=(0, 1.2*max), plot_width=850, plot_height = 300, 
#                title="Power Comparisons", toolbar_location=None, tools="")

#     for i in range(len(subCats)):
#         p.vbar(x=dodge('cats', dodges[i], range=p.x_range), top=subCats[i], width=0.2, source=source,
#         color=colors[i], legend_label=subCats[i])

#     p.x_range.range_padding = 0.1
#     p.xgrid.grid_line_color = None
#     p.legend.location = "top_left"
#     p.legend.orientation = "horizontal"
#     return p


# In[ ]:


# def MakeInterPowerBarPlot(KTarg, KSim, KExp, labels):
#     cats = labels
#     subCats = ['targ', 'sim', 'exp']
#     dodges =  [ -0.25,   0.0,  0.25]
#     colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    
#     targData = [KTarg.getMeasurement(label)[1] for label in labels]
#     simData = [KSim.getMeasurementAt(label, 1.525)[1] for label in labels]
#     expData = [KExp.getMeasurement(label)[1] for label in labels]
    
#     data = {'cats' : cats,
#             'targ' : targData,
#             'sim' : simData,
#             'exp' : expData}
#     source = ColumnDataSource(data=data)
    
#     max = np.max((targData,simData,expData))
#     p = figure(x_range=cats, y_range=(0, 1.2*max), plot_width=850, plot_height = 300, 
#                 title="Power Comparisons", toolbar_location=None, tools="")

#     for i in range(len(subCats)):
#         p.vbar(x=dodge('cats', dodges[i], range=p.x_range), top=subCats[i], width=0.2, source=source,
#         color=colors[i], legend_label=subCats[i])

#     p.x_range.range_padding = 0.1
#     p.xgrid.grid_line_color = None
#     p.legend.location = "top_left"
#     p.legend.orientation = "horizontal"
#     return p


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





# ## K3 Analysis (Mono)

# In[ ]:


K3Exp1525PD = {'T41':  55, 'T42': 107, 'T43':  63,
                'T51': 126, 'T52':  39, 'T53':  59,
                'T61':  46, 'T62':  49, 'T63': 105}


# In[ ]:


K3Exp1525PDInt = {'T41_T51':110, 'T42_T52':125, 'T43_T53': 70,
                  'T51_T61':120, 'T52_T62': 36, 'T53_T63':135,
                  'T41_T61': 99, 'T42_T62': 75, 'T43_T63':121}


# In[ ]:


sorted(list(K3Exp1525PDInt.keys())) == sorted(genInterferenceLabels(6))


# In[ ]:


KName = "K3"


# In[ ]:


KSim2D = SimSParams('Simulations/K3_2DEIA_SIM3.txt')
KSim = SimSParams('Simulations/K3_SIM3.txt')


# In[ ]:


KTarg = TargSParams('K3')


# In[ ]:


p = PolarPlot(KName)
p.addMatrix(KTarg.getSTransPart())
p.show()


# In[ ]:


KExp = ExpResult(K3Exp1525PD, 6)
KExp = ExpResult({**K3Exp1525PD, **K3Exp1525PDInt}, 6)


# In[ ]:


KSim2D.resetCorrectionFactor()
CF = findSingleRotCF(KSim2D.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()


# In[ ]:


p = makePhErrorSimPlot(KName, KTarg, KSim2D)
show(p)


# In[ ]:


KSim.resetCorrectionFactor()
CF = findDoubleRotCF(KSim.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CF)


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()


# In[ ]:


p = makePhErrorSimPlot(KName, KTarg, KSim)
show(p)


# In[ ]:


KSim.resetCorrectionFactor()


# In[ ]:


KExp.resetCorrectionFactor()
sf = findSF(KExp.getTTransPart(), KSim.getTTransPart(1.525))
KExp.applyCorrectionFactor(sf)


# In[ ]:


p = makeDispersionPlot(KName, 0.6)
addSimTraces(p, KSim, (4,5,6), (1,2,3))
addTargDots(p, KTarg, (4,5,6), (1,2,3))
addExpDots(p, KExp, (4,5,6), (1,2,3))
show(p)


# In[ ]:


KExp.resetCorrectionFactor()
KExp.applyCorrectionFactor()
KExp.resetCorrectionFactor()
KExp.applyCorrectionFactor(sf)
p = MakePowerBarPlot(KTarg, KSim, KExp, (4, 5, 6), (1, 2, 3))
show(p)


# In[ ]:


KExp.applyCorrectionFactor(0.001)


# In[ ]:


labels = genInterferenceLabels(6)


# In[ ]:


p = MakeInterPowerBarPlot(KTarg, KSim, KExp, labels)
show(p)


# ## K3

# ### Imports

# In[ ]:


K3Exp1525PD = {'T41':  55, 'T42': 107, 'T43':  63,
                'T51': 126, 'T52':  39, 'T53':  59,
                'T61':  46, 'T62':  49, 'T63': 105}


# In[ ]:


K3Exp1525PDInt = {'T41_T51':110, 'T42_T52':125, 'T43_T53': 70,
                  'T51_T61':120, 'T52_T62': 36, 'T53_T63':135,
                  'T41_T61': 99, 'T42_T62': 75, 'T43_T63':121}


# In[ ]:


K3Exp1525PD = {'T41':  91, 'T42':  47, 'T43':  19,
               'T51': 158, 'T52':  92, 'T53':  42,
               'T61': 148, 'T62':  85, 'T63':  57}


# In[ ]:


K3Exp1525PDInt = {'T41_T51':127, 'T42_T52': 77, 'T43_T53': 36,
                  'T51_T61':203, 'T52_T62':106, 'T53_T63': 64,
                  'T41_T61':145, 'T42_T62': 83, 'T43_T63': 32}


# In[ ]:


K3ExpMonoDict = {}
K3ExpMonoDict.update(K3Exp1525PD)
K3ExpMonoDict.update(K3Exp1525PDInt)


# In[ ]:


KTarg = TargSParams('K3')
KName = 'K3'


# In[ ]:


print(KTarg.getSTransPart())


# In[ ]:


p = PolarPlot(KName)
p.addMatrix(KTarg.getSTransPart())
p.show()


# In[ ]:


KSim2D = SimSParams('Simulations//K3_2DEIA_SIM3.txt')
KSim = SimSParams("Simulations/K3_SIM4.txt", "Simulations/Cal3_Sim4.txt")


# In[ ]:


KExp = ExpResult(K3ExpMonoDict, portCount=6, units='mV', WL=1.525, R_TIA=46700, gc_power_dBm=-14.7)
KExp.importExpGCEffCurve('Experiments\\Calibration\\GC_GC__n12dBm_46d7K.csv')


# In[ ]:


KExp.GCEffCurve


# In[ ]:


KExp.detailsDict


# In[ ]:


value = KExp.dataDict['T51']
PDResp =  0.8/1000  # [A/mW]
R_TIA = KExp.detailsDict['R_TIA']  # Ohms
P_GC = KExp.detailsDict['gc_power_mW']  # mW
GCEffCurve = KExp.GCEffCurve
PIn = P_GC*GCEffCurve  # mW
POut = value/(R_TIA*PDResp)  # mW
T = KExp.sf*(POut/PIn)


# In[ ]:


value


# In[ ]:


KExp.getMeasurement('T51')


# In[ ]:


KExp.getMeasurementAt('T51', 1.525)


# In[ ]:


KExpSpect = ExpResultSpect('K3', n=6, scaleFactor=1)
KExpSpect.importExpGCEffCurve('Experiments\\Calibration\\GC_GC__n12dBm_46d7K.csv')


# In[ ]:


#KExp = ExpResult(K4Exp1525PD, n=4, WL=1.525, scaleFactor=2.5)


# In[ ]:


KSim2D.getSTransPart(1.525);


# In[ ]:


KTarg.getSTransPart();


# ### Matching Sim2D with Target

# In[ ]:


KSim2D.resetCorrectionFactor()
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()


# In[ ]:


CF = findSingleRotCF(KSim2D.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()
KSim2D.resetCorrectionFactor()


# In[ ]:



CF = findDoubleRotCF(KSim2D.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()
KSim.resetCorrectionFactor()


# ### Matching Sim3D with Target

# In[ ]:


KSim.resetCorrectionFactor()
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()


# In[ ]:


CFSR = findSingleRotCF(KSim.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CFSR)
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()
KSim.resetCorrectionFactor()


# In[ ]:


CFDR = findDoubleRotCF(KSim.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CFDR)
p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()
KSim.resetCorrectionFactor()


# ### Spectroscopic Results

# In[ ]:


KExpSpect.applyCorrectionFactor(1.8)
p = DispersionPlot("Standard Transmission Measurements", "T", [1.4, 1.6], [0, 0.5])
for name in genTransLabels(6):
    p.addPairedTraces(KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
    # p.addPoint(KExp.getMeasurement(name), name)
p.show()
KExpSpect.resetCorrectionFactor()


# In[ ]:


KExp.getMeasurementAt('T51', 1.525)


# In[ ]:


KSim.resetCorrectionFactor()
KExpSpect.resetCorrectionFactor()

measurements = genTransLabels(6)
pbp = PowerBarPlot(cats=measurements, 
                   #subCats=['targ', 'sim', 'exp', 'expMono'], 
                   subCats=['targ', 'sim', 'expMono'], 
                   title='Standard Transmission Measurements')
pbp.addData('targ', [KTarg.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.addData( 'sim', [KSim.getMeasurementAt(m, 1.525) for m in measurements], sf=2)
# pbp.addData( 'exp', [KExpSpect.getMeasurementAt(m, 1.525) for m in measurements], sf=2)
pbp.addData( 'expMono', [KExp.getMeasurementAt(m, 1.525) for m in measurements], sf=2)
pbp.build()


# In[ ]:


KExpSpect.applyCorrectionFactor(0.8)
p = DispersionPlot("Intra Kernel Interference", "T", [1.4, 1.6], [0, 0.3])
for name in genInterferenceLabels(6):
    p.addPairedTraces(KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
p.show()
KExpSpect.resetCorrectionFactor()


# In[ ]:


KSim.resetCorrectionFactor()
KExpSpect.resetCorrectionFactor()

measurements = genInterferenceLabels(6)
pbp = PowerBarPlot(cats=measurements, 
                   #subCats=['targ', 'sim', 'exp', 'expMono'], 
                   subCats=['targ', 'sim', 'expMono'],                    
                   title='Internal Interference Measurements')
pbp.addData('targ', [KTarg.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.addData( 'sim', [KSim.getMeasurementAt(m, 1.525) for m in measurements], sf=2)
# pbp.addData( 'exp', [KExpSpect.getMeasurementAt(m, 1.525) for m in measurements], sf=2)
pbp.addData( 'expMono', [KExp.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.build()


# ## K4

# In[ ]:


K4Exp1525PD = {'T31':  55, 'T32': 107, 
               'T41': 126, 'T42':  39,
              'T31_T41':110, 'T32_T42':125}


# In[ ]:


KExp = ExpResult(K4Exp1525PD, portCount=4)


# In[ ]:


KTarg = TargSParams('K4')
KName = 'K4'


# In[ ]:


print(KTarg.getSTransPart())


# In[ ]:


p = PolarPlot(KName)
p.addMatrix(KTarg.getSTransPart())
p.show()


# In[ ]:


KSim2D = SimSParams('Simulations//K4_2DEIA_SIM3.txt')
KSim = SimSParams("Simulations/K4_SIM4.txt", "Simulations/Cal4_Sim4.txt")


# In[ ]:


KExpSpect = ExpResultSpect('K4', n=4, scaleFactor=1)
KExpSpect.importExpGCEffCurve('Experiments\\Calibration\\GC_GC__n12dBm_46d7K.csv')


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()


# In[ ]:


KSim2D.resetCorrectionFactor()
CF = findSingleRotCF(KSim2D.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim2D.applyCorrectionFactor(CF)


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim2D.getSTransPart(1.525))
p.show()


# In[ ]:


# p = makePhErrorSimPlot(KName, KTarg, KSim2D)
# show(p)


# In[ ]:


KSim.getSTrace(3,1);


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()


# In[ ]:


KSim.resetCorrectionFactor()
CF = findSingleRotCF(KSim.getSTransPart(1.525), KTarg.getSTransPart())


# In[ ]:


KSim.applyCorrectionFactor(CF)


# In[ ]:


p = PolarPlot(KName)
p.addMatrixDiff(KTarg.getSTransPart(), KSim.getSTransPart(1.525))
p.show()


# In[ ]:


KSim.resetCorrectionFactor()


# In[ ]:


KExpSpect.applyCorrectionFactor(1)
p = DispersionPlot("Standard Transmission Measurements", "T", [1.4, 1.6], [0, 0.5])
for name in genTransLabels(4):
    p.addPairedTraces(KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
    # p.addPoint(KExp.getMeasurement(name), name)
p.show()


# In[ ]:


KSim.resetCorrectionFactor()
KExpSpect.resetCorrectionFactor()

measurements = genTransLabels(4)
pbp = PowerBarPlot(cats=measurements, 
                   subCats=['targ', 'sim', 'exp', 'expMono'], 
                   title='Standard Transmission Measurements')
pbp.addData('targ', [KTarg.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.addData( 'sim', [KSim.getMeasurementAt(m, 1.525) for m in measurements], sf=1.45)
pbp.addData( 'exp', [KExpSpect.getMeasurementAt(m, 1.525) for m in measurements], sf=1.70)
pbp.addData( 'expMono', [KExp.getMeasurementAt(m, 1.525) for m in measurements], sf=0.0033)
pbp.build()


# In[ ]:


KExpSpect.applyCorrectionFactor(0.4)
p = DispersionPlot("Intra Kernel Interference", "T", [1.4, 1.6], [0, 0.3])
for name in genInterferenceLabels(4):
    p.addPairedTraces(KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
p.show()


# In[ ]:


measurements = genInterferenceLabels(4)
pbp = PowerBarPlot(cats=measurements, 
                   subCats=['targ', 'sim', 'exp', 'expMono'], 
                   title='Internal Interference Measurements')
pbp.addData('targ', [KTarg.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.addData( 'sim', [KSim.getMeasurementAt(m, 1.525) for m in measurements], sf=1.45)
pbp.addData( 'exp', [KExpSpect.getMeasurementAt(m, 1.525) for m in measurements], sf=1.45)
pbp.addData( 'expMono', [KExp.getMeasurementAt(m, 1.525) for m in measurements], sf=0.0013)
pbp.build()


# In[ ]:


KExpSpect.applyCorrectionFactor(0.30)
p = DispersionPlot("Reference Waveguide Measurements", 'T', [1.4, 1.6], [0, 0.08])
labels = ('T31_R', 'T42_R')
for name in labels:
    p.addPairedTraces(KSim.getMeasurement(name), KExpSpect.getMeasurement(name), name)
p.show()


# In[ ]:


KSim.resetCorrectionFactor()
KExpSpect.resetCorrectionFactor()

measurements = ['T31_R', 'T42_R']
pbp = PowerBarPlot(cats=measurements, 
                   subCats=['sim', 'exp'], 
                   title='Internal Interference Measurements')
#pbp.addData('targ', [KTarg.getMeasurementAt(m, 1.525) for m in measurements], sf=1)
pbp.addData( 'sim', [KSim.getMeasurementAt(m, 1.525) for m in measurements], sf=1.5)
pbp.addData( 'exp', [KExpSpect.getMeasurementAt(m, 1.525) for m in measurements], sf=0.3)
pbp.build()


# In[ ]:


KExp.resetCorrectionFactor()
sf = findSF(KExp.getPTransPart(), KSim.getPTransPartAtWL(1.525))
KExp.applyCorrectionFactor(sf)


# In[ ]:


p = DispersionPlot(KName, 0.6)
p.addSimTraces(p, KSim, (3,4), (1,2))
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
df=pd.read_csv("./Simulations/GC_V1.csv", sep=',',header=None)


# In[ ]:


df


# In[ ]:


import pandas as pd
df=pd.read_csv"/content/SiPhDataStore/Simulations/GC_V1.csv"", sep=',',header=None)


# In[ ]:


K4T31 = np.array(df)


# ## Bar Charts

# In[ ]:


getExpTrace()


# In[ ]:




