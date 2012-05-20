#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import os
import sys
import getopt

from multimap import *
import kno_colors as my_cm

opts, args = getopt.getopt(sys.argv[1:], 'c:m:z:', 
                           ['eps','pdf','png','xlim=','ylim=','title='])

if len(args) > 0 : dataFileName = args[0]
else : dataFileName = "scalars.out"

modify_xlim = False
modify_ylim = False
save_pdf = False
save_eps = False
save_png = False

z_col = 3

plot_options = {}
plot_options["cmap"] = cm.jet

for opt,val in opts:
    if opt == "--pdf": save_pdf = True
    if opt == "--eps": save_eps = True
    if opt == "--png": save_png = True
    if opt == "--xlim": 
        modify_xlim = True
        xlim = val
    if opt == "--ylim": 
        modify_ylim = True
        ylim = val
    if opt == "--title":
        plt.title(val)
    if opt == "-c":
        z_col = val
    if opt == "-m":
        if val == "spin":
            plot_options["cmap"] = my_cm.spin
        if val == "hot":
            plot_options["cmap"] = cm.hot_r
    if opt == "-z":
        zlimits = val.split(":")
        plot_options["vmin"] = float(zlimits[0])
        plot_options["vmax"] = float(zlimits[1])

fig = plt.figure()
ax = fig.add_subplot(111, frame_on = False)

data = MultiMap(dataFileName)

x,y,z,extent = data.retrieve_3d_plot_data("1", "2", z_col)

print "xrange: ", np.min(x), np.max(x)
print "yrange: ", np.min(y), np.max(y)
print "zrange: ", np.min(z), np.max(z)

plt.imshow(z, extent = extent, **plot_options)

if modify_xlim:
    xlimits = xlim.split( ":" )
    plt.xlim( float( xlimits[0] ), float( xlimits[1] ) )

if modify_ylim:
    ylimits = ylim.split( ":" )
    plt.ylim( float( ylimits[0] ), float( ylimits[1] ) )

# save to eps
if save_eps:
    outfileName = dataFileName.replace(".dat",".eps")
    plt.savefig( outfileName )

# save to pdf
if save_pdf:
    if debug: print "saving to pdf"
    outfileName = dataFileName.replace(".dat",".pdf")
    plt.savefig( outfileName )

# save to png
if save_png:
    outfileName = dataFileName.replace(".dat",".png")
    plt.savefig( outfileName )

ax.xaxis.set_major_locator(mticker.NullLocator())
ax.yaxis.set_major_locator(mticker.NullLocator())
plt.show()
