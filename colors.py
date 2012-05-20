#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# create my own colormap (essentially RdBu with real white in the center)
# idea taken from 
# http://old.nabble.com/cmap-from-sepparate-color-values-td21503197.html
colors = ('blue', 'white', 'red')
ncolors = len(colors)
vals = np.linspace(0., 1., ncolors)

cdict = dict(red=[], green=[], blue=[])
for val, color in zip(vals, colors):
    r,g,b = mcolors.colorConverter.to_rgb(color)

    cdict['red'].append((val, r, r))
    cdict['green'].append((val, g, g))
    cdict['blue'].append((val, b, b))

spin = mcolors.LinearSegmentedColormap('spin', cdict)




