#!/usr/bin/env python
# coding: utf-8

# ### Stock Imports

# In[ ]:


import numpy as np


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


# ### Custom Imports

# In[ ]:


from UtilityMath import RandomComplexGaussianMatrix


# In[ ]:


mainQ =(__name__ == '__main__')
mainQ


# #### Polar Error Plot

# In[ ]:


def makePolarPlot(title):
    '''
    This will create a Bokeh plot that depicts the unit circle.

    Requires import bokeh
    '''
    p = bokeh.plotting.figure(plot_width=400, plot_height=400, title=title, 
                              x_range=[-1.1, 1.1], y_range=[-1.1, 1.1])
    p.xaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25))
    p.yaxis[0].ticker=bokeh.models.tickers.FixedTicker(ticks=np.arange(-1, 2, 0.25)) 
    p.circle(x = [0,0,0,0], y = [0,0,0,0], radius = [0.25, 0.50, 0.75, 1.0], 
             fill_color = None, line_color='gray')
    p.line(x=[0,0], y=[-1,1], line_color='gray')
    p.line(x=[-1,1], y=[0,0], line_color='gray')
    xs = [0.25, 0.50, 0.75, 1.00]
    ys = [0, 0, 0, 0]
    texts = ['0.25', '0.50', '0.75', '1.00']
    source = bokeh.models.ColumnDataSource(dict(x=xs, y=ys, text=texts))
    textGlyph = bokeh.models.Text(x="x", y="y", text="text", angle=0.3, 
                                  text_color="gray", text_font_size='10px')
    p.add_glyph(source, textGlyph)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p


# In[ ]:


def addMatrixDiff(bokehPlot, m1, m2):
    """
    This will draw lines showing the difference between two 2D matrices.
    """
    p = bokehPlot
    begX = (np.real(m1)).flatten()
    begY = (np.imag(m1)).flatten()
    endX = (np.real(m2)).flatten()
    endY = (np.imag(m2)).flatten()

    xs = np.array([begX, endX]).T.tolist()
    ys = np.array([begY, endY]).T.tolist()
    p.multi_line(xs=xs, ys=ys)

    sourceTarg = bokeh.models.ColumnDataSource(dict(x=begX.tolist(), y=begY.tolist()))
    glyphTarg = bokeh.models.Circle(x="x", y="y", size=10, line_color="green", 
                                    fill_color=None, line_width=3)
    p.add_glyph(sourceTarg, glyphTarg)

    sourceSim = bokeh.models.ColumnDataSource(dict(x=endX.tolist(), y=endY.tolist()))
    glyphSim = bokeh.models.Circle(x="x", y="y", size=5, line_color=None, 
                                   fill_color='red', line_width=3)
    p.add_glyph(sourceSim, glyphSim)
    return p


# In[ ]:


m1 = RandomComplexGaussianMatrix(0.5, (5,5))
m2 = m1 + RandomComplexGaussianMatrix(0.05, (5,5))


# In[ ]:


p = makePolarPlot("Plot Title")
addMatrixDiff(p, m1, m2)
if mainQ: show(p)


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


# #### Dispersion Plot

# In[ ]:


def makeDispersionPlot(title, xRange, yRange):
    p = figure(plot_width=850, plot_height=400, title=title, x_range=xRange, y_range=yRange)
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

