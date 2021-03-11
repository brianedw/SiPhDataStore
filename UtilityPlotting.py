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
import bokeh.palettes as palettes

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


class PolarPlot:
    
    def __init__(self, title):
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
        self.p = p
    
    def addMatrixDiff(self, m1, m2):
        """
        This will draw lines showing the difference between two 2D matrices.
        """
        p = self.p
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

    def addMatrix(self, m1):
        """
        This will draw lines showing the difference between two 2D matrices.
        """
        p = self.p
        X = (np.real(m1)).flatten()
        Y = (np.imag(m1)).flatten()
        source = bokeh.models.ColumnDataSource(dict(x=X.tolist(), y=Y.tolist()))
        glyph = bokeh.models.Circle(x="x", y="y", size=5, line_color=None, 
                                       fill_color='cyan', line_width=3)
        p.add_glyph(source, glyph)
        
    def show(self):
        show(self.p)


# In[ ]:


m1 = RandomComplexGaussianMatrix(0.5, (5,5))
m2 = m1 + RandomComplexGaussianMatrix(0.05, (5,5))
m3 = RandomComplexGaussianMatrix(0.5, (5,5))


# In[ ]:


pp = PolarPlot("Plot Title")
pp.addMatrixDiff(m1, m2)
pp.addMatrix(m3)
if mainQ: pp.show()


# #### Dispersion Plot (Class)

# In[ ]:


class DispersionPlot:
    
    def __init__(self, title, yLabel, xRange, yRange):
        self.p = figure(plot_width=850, plot_height=400, title=title, x_range=xRange, y_range=yRange)
        # p.update_layout(shapes=[dict(type= 'line', yref= 'paper', y0= 0, y1= 1, xref= 'x', x0= 1.525, x1= 1.525)])
        self.p.xaxis.axis_label = 'wavelength (um)'
        self.p.yaxis.axis_label = yLabel
        self.colorIndex = 0
    
    def addPairedTraces(self, traceDataGoal, traceDataExp, name, colorIndex="auto"):
        p = self.p
        if colorIndex == "auto":
            color = palettes.Category10[10][self.colorIndex%10]
            self.colorIndex += 1
        if colorIndex == "hash":
            color = palettes.Category10[10][hash(name)%10]            
        elif type(colorIndex) is int:
            color = palettes.Category10[10][colorIndex%10]
        self.p.line(traceDataGoal[0], traceDataGoal[1], line_color=color, line_width=1, legend_label=name+" goal")
        self.p.line(traceDataExp[0],  traceDataExp[1],  line_color=color, line_width=2, legend_label=name+" exp")
        
    def addTrace(self, traceData, name):
        color = palettes.Category10[10][hash(name)%10]
        self.p.line(traceData[0], traceData[1], line_color=color, line_width=2, legend_label=name)
        
    def addPoint(self, ptData, name):
        color = palettes.Category10[10][hash(name)%10]
        self.p.circle(ptData[0], ptData[1], line_color=color, fill_color=color)
        
    def show(self):
        show(self.p)


# In[ ]:





# In[ ]:


wls = np.arange(0.6, 1.4, 0.01)
data1a = 0.3*wls**2
data1b = 0.3*wls**2+0.04
data2 = 1 - 0.3*wls
data3 = data2[::20] - 0.05
wlsDec = wls[::20]


# In[ ]:


dp = DispersionPlot("Test Plot", "|S|", [0.5, 1.5], [0, 1])
dp.addPairedTraces((wls, data1a), (wls, data1b), "Trial1")
dp.addTrace((wls, data2), "Other Data")
dp.addPoint((wlsDec, data3), "Other Data")
dp.show()


# #### MakeBarPlot (Class)

# In[ ]:


class PowerBarPlot:
    
    def __init__(self, cats, subCats, title):
        self.cats = cats        # ['T31', 'T41', 'T51', ...]
        self.subCats = subCats  # ['targ', 'sim', 'exp']
        self.title = title
        self.data = {}
        self.sf = {}
        
    def addData(self, subCat, data, sf=1):
        if subCat not in self.subCats:
            print("I don't recognize subcat.  Should be one of ", self.subCats)
        if len(data) != len(self.cats):
            print("Incorrect length.  Expecting a 1-to-1 correspondance with ", self.cats)
        self.data[subCat] = [sf*d for d in data]
        self.sf[subCat] = sf
        
    def build(self):
        self.data['cats'] = self.cats
        dodges = (np.linspace(-0.5, 0.5, endpoint=True, num=len(self.subCats)+2)[1:-1]).tolist()
        colors = ["#c9d9d3", "#718dbf", "#e84d60", "#408040"]
        colors = colors[:len(self.subCats)]
        maxV = np.max([self.data[subCat] for subCat in self.subCats])
        p = figure(x_range=self.cats, y_range=(0, 1.2*maxV), plot_width=850, plot_height = 300, 
               title=self.title, toolbar_location=None, tools="")
        source = ColumnDataSource(data=self.data)
        for i in range(len(self.subCats)):
            sf = self.sf[self.subCats[i]]
            if sf == 1:
                labelSF = ""
            else:
                labelSF = " (x "+str(self.sf[self.subCats[i]])+")"
            p.vbar(x=dodge('cats', dodges[i], range=p.x_range), top=self.subCats[i], width=0.2, source=source,
                   color=colors[i], legend_label=self.subCats[i]+labelSF)
        show(p)


# In[ ]:


pbp = PowerBarPlot(cats=['apples', 'bananas', 'pears', 'oranges'], 
                   subCats=['raw', 'ripe', 'rotten'], 
                   title='Fruit Inventory')
pbp.addData(   'raw', [  0, 80, 30, 30], sf=0.1)
pbp.addData(  'ripe', [  5,  2,  5,  4])
pbp.addData('rotten', [  2,  3,  1,  2])
pbp.build()


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
    colors = ["#c9d9d3", "#718dbf", "#e84d60", "#184d10"]

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


# In[ ]:




