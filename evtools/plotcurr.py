#!/usr/bin/env python

import matplotlib.cm as cm
import matplotlib.ticker as mticker
import os
import sys
import getopt
import textwrap

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

    color_coding = False
    c_col = 4
    
    N = 5

    plot_options = {}

    quiver_plot_options = {}
    quiver_plot_options["units"] = "height"
    quiver_plot_options["width"] = 0.002
    #quiver_plot_options["headlength"] = 3
    #quiver_plot_options["headaxislength"] = 2.5
    quiver_plot_options["scale"] = 1
    #quiver_plot_options["headwidth"] = 3
    quiver_plot_options["pivot"] = "middle"

    color_plot_options = {}
    color_plot_options["cmap"] = my_cm.spin

    for opt,val in opts.items():
        if opt == "--pdf": save_pdf = True
        if opt == "--eps": save_eps = True
        if opt == "--png": save_png = True
        if opt == "--color":
            color_coding = True
            c_col = int(val)

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
        if opt == "-m":
            if val == "spin":
                color_plot_options["cmap"] = my_cm.spin
            if val == "hot":
                color_plot_options["cmap"] = cm.hot_r
            if val == "grey":
                color_plot_options["cmap"] = cm.binary
            if val == "jet":
                color_plot_options["cmap"] = cm.jet
        if opt == "-z":
            zlimits = val.split(":")
            color_plot_options["vmin"] = float(zlimits[0])
            color_plot_options["vmax"] = float(zlimits[1])

        if opt == "-N":
            N = int(val)

        if opt == "-b":
            show_colorbar = True

        if opt == "-s" or opt == "--scale":
            quiver_plot_options["scale"] = 1 / float(val)

    data = MultiMap(dataFileName)

    # this is getting the data for the quiverplot
    x,y,u,v,extent = data.retrieve_quiver_plot_data("1", "2", u_col, v_col, N = N)
    print "xrange: ", np.min(x), np.max(x)
    print "yrange: ", np.min(y), np.max(y)
    print "urange: ", np.min(u), np.max(u)
    print "vrange: ", np.min(v), np.max(v)

    if color_coding:
        x,y,c,extent = data.retrieve_3d_plot_data("1", "2", c_col, N = N)
        print "crange: ", np.min(c), np.max(c)
        options = dict(plot_options, **color_plot_options)
        plt.imshow(c, extent = extent, **options)


    options = dict(plot_options, **quiver_plot_options)
    result = plt.quiver(x[:], y[:], u[:], v[:], **options)

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
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:m:z:bN:s:f:', 
                               ['col=','eps','pdf','png','xlim=',
                                   'ylim=','title=', 'color=',
                                   'scale=','figsize='])

        figsize = (20, 5)
        for opt,val in opts.items():
            if opt == "-f" or opt == "--figsize":
                tmp = val.splot(",")
                figsize = tuple([float(x) for x in tmp])

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, frame_on = False)

        opts = dict(opts)
        
        if len(args) > 0 : dataFileName = args[0]
        else : dataFileName = "scalars.out"
       
        plotcurr(dataFileName, **opts)

        plt.axis("equal")
        plt.show()
    except getopt.GetoptError:
        print textwrap.dedent("""\
        Usage: plotcurr.py [OPTIONS] ... datafilename
        Creates a vector-field plot according to x,y and u,v data found in 
        'datafilename' in particular columns

        Mandatory arguments to long options are mandatory for short options too.
        -c, --col=NUMBER    use the columns beginning from column NUMBER for u and v
            --color=COLUMN   color-code the arrows by some shading given by column COLUMN
        -N=bins                 average over a
        -b
        -s, --scale=FACTOR  changes the arrow size by FACTOR
        -m=NAME                     the NAME of the desired colormap, possible values are
                                    spin, hot, gray and jet
        -z
            --eps
            --pdf
            --png
            --xlim=LOWER:UPPER
            --ylim=LOWER:UPPER
            --title=STRING
        -f, --figsize=W,H               Width and Height of the produced figure in inches
                                        default is 20"x5"
        """)

