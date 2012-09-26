#!/usr/bin/env python

import matplotlib.cm as cm
import matplotlib.ticker as mticker
import os
import sys
import getopt

from knodat.multimap import *
import knodat.colors as my_cm

def plotcurr(dataFileName, **opts):
    ax = plt.subplot(111)
    modify_xlim = False
    modify_ylim = False
    save_pdf = False
    save_eps = False
    save_png = False
    show_colorbar = False

    u_col = 2
    v_col = 3
    c_col = 13

    plot_options = {}
    #plot_options["interpolation"] = 'nearest'
    plot_options["cmap"] = my_cm.spin
    plot_options["angles"] = "xy"
    plot_options["units"] = "height"
    #plot_options["scale_units"] = "width"
    plot_options["scale"] = 1.0 / 1

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
            u_col = float(val) - 1
            v_col = float(val)
            c_col = float(val) + 1
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

    x,y,u,v,extent = data.retrieve_quiver_plot_data("1", "2", u_col, v_col, N = 5)
    x,y,c,extent = data.retrieve_3d_plot_data("1", "2", c_col, N = 5)

    print "xrange: ", np.min(x), np.max(x)
    print "yrange: ", np.min(y), np.max(y)
    print "urange: ", np.min(u), np.max(u)
    print "vrange: ", np.min(v), np.max(v)
    print "crange: ", np.min(c), np.max(c)

    result = plt.quiver(x[:], y[:], u[:], v[:], c[:], **plot_options)
    #result = plt.quiver(x[:], y[:], u[:], v[:], **plot_options)

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
    plotcurr(dataFileName, **opts)

    plt.axis("equal")

    #ax.xaxis.set_major_locator(mticker.NullLocator())
    #ax.yaxis.set_major_locator(mticker.NullLocator())

    plt.show()
