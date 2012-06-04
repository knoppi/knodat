#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import os
import sys
import getopt

from knodat.multimap import *
import knodat.colors as my_cm

def plotg(dataFileName, **opts):
    #ax = plt.axes([0,0,1,1])
    ax = plt.subplot(111)
    modify_xlim = False
    modify_ylim = False
    save_pdf = False
    save_eps = False
    save_png = False
    show_colorbar = False

    z_col = 2

    plot_options = {}
    plot_options["cmap"] = cm.jet
    plot_options["interpolation"] = 'nearest'

    contour_opts = {}

    plot_mode = "image"

    for opt,val in opts.items():
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
            if val == "grey":
                plot_options["cmap"] = cm.binary
            if val == "jet":
                plot_options["cmap"] = cm.jet
        if opt == "-z":
            zlimits = val.split(":")
            plot_options["vmin"] = float(zlimits[0])
            plot_options["vmax"] = float(zlimits[1])

        if opt == "-b":
            show_colorbar = True

        if opt == "--contour":
            plot_mode = "contour"

        if opt == "--levels":
            levels = val.split(":")
            levels = [float(x) for x in levels]
            contour_opts["levels"] = levels


    data = MultiMap(dataFileName)

    x,y,z,extent = data.retrieve_3d_plot_data("1", "2", z_col)

    print "xrange: ", np.min(x), np.max(x)
    print "yrange: ", np.min(y), np.max(y)
    print "zrange: ", np.min(z), np.max(z)

    #z = np.ma.masked_array(z, mask=np.isnan(z))
    result = ""
    if plot_mode is "image":
        result = plt.imshow(z, extent = extent, **plot_options)
    else:
        plot_options.update(contour_opts)
        result = plt.contour(z, extent = extent, **plot_options)

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

    if show_colorbar is True:
        cax = plt.axes([0.7,0.1,0.18,0.08])
        plt.colorbar(cax=cax, orientation = "horizontal")
        labels = cax.get_xticklabels()
        for label in labels:
                label.set_rotation(30) 
                label.set_ha("right")
                label.set_size("small")

    return result

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'c:m:z:b', 
                               ['eps','pdf','png','xlim=','ylim=','title='])

    if len(args) > 0 : dataFileName = args[0]
    else : dataFileName = "scalars.out"

    fig = plt.figure()
    ax = fig.add_subplot(111, frame_on = False)

    opts = dict(opts)
    plotg(dataFileName, **opts)

    ax.xaxis.set_major_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.NullLocator())

    plt.show()
