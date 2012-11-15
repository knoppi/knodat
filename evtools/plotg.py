#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cbook as mcbook
import os
import sys
import getopt
import textwrap
import logging

from knodat.multimap import *
import knodat.colors as my_cm

module_logger = logging.getLogger("PLOTG")
formatter = logging.Formatter(
        fmt = "%(relativeCreated)d -- %(name)s -- %(levelname)s -- %(message)s" )

ch = logging.StreamHandler()
ch.setLevel( logging.WARNING )

module_logger.addHandler( ch )
module_logger.setLevel( logging.WARNING )

def set_debug_level(level):
    possible_levels = dict(debug = logging.DEBUG, info = logging.INFO,
            warning = logging.WARNING, error = logging.ERROR,
            fatal = logging.FATAL)
    ch.setLevel(possible_levels[level])
    module_logger.setLevel(possible_levels[level])


class BiLogNorm(mcolors.Normalize):
    """
    Normalize a given value to the 0-1 range on a two-sided log scale,
    i.e. a values distance from the central value (vmax-vmin)/2
    """
    def __init__(self, vmin=None, vmax=None, clip=False, cutoff = 1.0e-20):
        """
        If *vmin* or *vmax* is not given, they are taken from the input's
        minimum and maximum value respectively.  If *clip* is *True* and
        the given value falls outside the range, the returned value
        will be 0 or 1, whichever is closer. Returns 0 if::

        vmin==vmax

        Works with scalars or arrays, including masked arrays.  If
        *clip* is *True*, masked values are set to 1; otherwise they
        remain masked.  Clipping silently defeats the purpose of setting
        the over, under, and masked colors in the colormap, so it is
        likely to lead to surprises; therefore the default is
        *clip* = *False*.
        """
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip
        self.cutoff = cutoff

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)
        module_logger.debug(result)

        #result = np.ma.masked_less_equal(result, 0, copy=False)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                val = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                mask=mask)
            
            # calculate the central point to create to masks
            c = (vmax + vmin) / 2.0
            cutoff = self.cutoff

            shift = np.log10(c - vmin) - np.log10(cutoff)
            
            # two masked arrays for keeping the datapoints below or
            # above the central point respectively
            resdat = result.data
            mask1 = result.mask
            mask2 = result.mask
            mask3 = result.mask

            if mask1 is np.ma.nomask:
                mask1 = (resdat < c + cutoff)
                mask2 = (resdat > c - cutoff)
                mask3 = np.ma.masked_outside(resdat, c-cutoff, c+cutoff).mask
            else:
                mask1 |= resdat > c + cutoff
                mask2 |= resdat < c - cutoff
                mask3 |= np.ma.masked_outside(resdat, c-cutoff, c+cutoff).mask
            values1 = np.ma.array(result, mask = mask1)
            module_logger.debug(values1)
            values2 = np.ma.array(result, mask = mask2)
            module_logger.debug(values2)
            values3 = np.ma.array(result, mask = mask3, hard_mask = True)
            module_logger.debug(values3)

            values1 -= c
            values2 *= -1
            values2 += c
            values3 -= values3
            module_logger.debug(result)
            np.log10(result, result)
            module_logger.debug(result)
            #np.log10(values1, values1)
            #np.log10(values2, values2)
            values1 -= np.log10(cutoff)
            values2 -= np.log10(cutoff)
            module_logger.debug(result)
            values2 *= -1
            module_logger.debug(result)

            # and the transformation for the lower branch
            #module_logger.debug("second masked array after rescaling = %s" % values2[0][::50])

            #values3 = np.ma.array(np.ones(values3.shape),mask=values3.mask)*0.5
            np.ma.fix_invalid(values3, copy = False, fill_value = 0.0)

            result += shift
            result /= 2
            result /= shift

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
            return vmin * ma.power((vmax/vmin), val)
        else:
            return vmin * pow((vmax/vmin), value)

    def autoscale(self, A):
        '''
        Set *vmin*, *vmax* to min, max of *A*.
        '''
        A = ma.masked_less_equal(A, 0, copy=False)
        self.vmin = ma.min(A)
        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is not None and self.vmax is not None:
            return
        A = ma.masked_less_equal(A, 0, copy=False)
        if self.vmin is None:
            self.vmin = np.ma.min(A)
        if self.vmax is None:
            self.vmax = np.ma.max(A)

def plotg(dataFileName, **opts):
    # Variables perhaps modified by **opts
    modify_xlim = False
    modify_ylim = False
    save_pdf = False
    save_eps = False
    save_png = False
    show_colorbar = False
    show_grid = False

    # the default column, note that col-numbering is
    # 0 1 2 ...
    z_col = "3"

    # the dict with the plotting options
    plot_options = {}
    plot_options["cmap"] = cm.jet
    plot_options["interpolation"] = 'nearest'

    # same for the contour plot extra options
    contour_opts = {}
    
    # range of the Gaussian convolution for numerical interpolation
    N = 1

    # by default we plot a color-coded image, contour is also possible
    plot_mode = "image"

    # processing of options that are needed to fetch the data
    for opt,val in opts.items():
        if opt == "-c":
            z_col = val
        if opt == "-N":
            N = int(val)

    # fetch the data
    data = MultiMap(dataFileName)
    x,y,z,extent = data.retrieve_3d_plot_data("1", "2", z_col, grid = 'graphenegrid', N = N)
    plot_options["extent"] = extent

    # the following data is useful in every case
    print "xrange: ", np.min(x), np.max(x)
    print "yrange: ", np.min(y), np.max(y)
    plot_options["vmin"] = np.min(z)
    plot_options["vmax"] = np.max(z)
    print "zrange: ", plot_options["vmin"], plot_options["vmax"]


    # process options for the creation of the plot
    for opt,val in opts.items():
        if opt == "--pdf": 
            save_pdf = True
        if opt == "--eps": 
            save_eps = True
        if opt == "--png": 
            save_png = True
        if opt == "--xlim": 
            modify_xlim = True
            xlim = val
        if opt == "--ylim": 
            modify_ylim = True
            ylim = val
        if opt == "--title":
            plt.title(val)
        if opt == "-m":
            if val == "spin":
                plot_options["cmap"] = my_cm.spin
            if val == "hot":
                plot_options["cmap"] = cm.hot_r
            if val == "grey":
                plot_options["cmap"] = cm.binary
            if val == "jet":
                plot_options["cmap"] = cm.jet
            if val == 'blues':
                plot_options["cmap"] = cm.Blues
            if val == 'reds':
                plot_options["cmap"] = cm.Reds
        if opt == "-z":
            zlimits = val.split(":")
            plot_options["vmin"] = float(zlimits[0])
            plot_options["vmax"] = float(zlimits[1])

        if opt == "-b":
            show_colorbar = True
        if opt == "--interpolation":
            plot_options["interpolation"] = val

        if opt == "--contour":
            plot_mode = "contour"
        if opt == "--levels":
            levels = val.split(":")
            levels = [float(t) for t in levels]
            contour_opts["levels"] = levels

        if opt == "-g" or opt == "--grid":
            show_grid = True

        if opt == "--noaspect":
            #del plot_options["extent"]
            plot_options["aspect"] = "auto"
            contour_opts['aspect'] = "auto"

    # a second run for options that might need the procession of another option
    # already before they are processed
    for opt,val in opts.items():
        if opt == "-l" or opt == "--logarithmic":
            plot_options["norm"] = mcolors.LogNorm(
                    plot_options["vmin"], plot_options["vmax"])
        if opt == "--bilogarithmic":
            if val == '':
                plot_options["norm"] = BiLogNorm(
                        plot_options["vmin"], plot_options["vmax"])
            else:
                plot_options["norm"] = BiLogNorm(
                        plot_options["vmin"], plot_options["vmax"],
                        cutoff = float(val))

    if show_grid is True:
        plt.plot(x, y, "k,")

    result = ""
    if plot_mode is "image":
        result = plt.imshow(z, 
                **plot_options)
    else:
        plot_options.update(contour_opts)
        result = plt.contour(z, **plot_options)

    if modify_xlim:
        xlimits = xlim.split( ":" )
        plt.xlim( float( xlimits[0] ), float( xlimits[1] ) )

    if modify_ylim:
        ylimits = ylim.split( ":" )
        plt.ylim( float( ylimits[0] ), float( ylimits[1] ) )

    ax = plt.gca()
    fig = plt.gcf()
    
    if show_colorbar is True:
        trans = axes.transAxes
        inv = fig.transFigure.inverted()
        
        rect = np.array([0.75, -0.1, 0.15, 0.08])
        rect[2:4] += rect[0:2]
        
        rect[0:2] = trans.transform(rect[0:2])
        rect[2:4] = trans.transform(rect[2:4])
        rect[0:2] = inv.transform(rect[0:2])
        rect[2:4] = inv.transform(rect[2:4])

        rect[2:4] -= rect[0:2]
        
        cax = plt.axes([0.7,0.1,0.28,0.08], transform = ax.transAxes)
        plt.colorbar(cax=cax, orientation = "horizontal")
        labels = cax.get_xticklabels()
        for label in labels:
                label.set_rotation(30) 
                label.set_ha("right")
                label.set_size("small")

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

    return result

def usage():
    print textwrap.dedent("""\
    Usage: plotg.py [OPTIONS] ... datafilename
    Creates a color-coded image plot according to x,y and z data found in 
    'datafilename' in particular columns

    Mandatory arguments to long options are mandatory for short options too.
    -c, --col=NUMBER            use the columns NUMBER for z
    -N=BINS                     average over BINS data points, effectively reducing
                                the number of datapoint for the plot
        --interpolation=NAME    the name of an allowed interpolation method of
                                the pyplot.imshow method, currently allowed values
                                are: 'nearest', 'bilinear', 'bicubic', 'spline16', 
                                'spline36', 'hanning', 'hamming', 'hermite', 
                                'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 
                                'mitchell', 'sinc', 'lanczos'
                                Defaults to nearest
    -b                          plot a colorbar-legend
    -g, --grid                  plot the grid behind the color-image
    -m=NAME                     the NAME of the desired colormap, possible values are
                                spin, hot, gray and jet
    -z LOWER:UPPER              limits the z-axis to a given range
        --eps                   saves the plot as an eps file
        --pdf                   saves the plot as a pdf file
        --png                   saves the plot as a png file
        --xlim=LOWER:UPPER
        --ylim=LOWER:UPPER
        --title=STRING
    -l, --logarithmic           make a logarithmic color-coding
        --bilogarithmic CUTOFF  uhh, hard to explain, logarithmic scale but to two 
                                sides, measuring the distance from a central point
                                Be careful that the data points are not to close
                                to the central point
        --interpolation
    -n, --noaspect              aspect ratio of the data is not kept
    -h, --help                  shows this explanation
        --contour               make a contour plot no color-coded one
        --levels LEVELS         use the colon-separated LEVELS for the contour plot
    """)

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:m:z:bN:glhn', 
                                   ['eps','pdf','png','xlim=','ylim=','title=',
                                       'interpolation=', 'grid', "logarithmic",
                                       "help", "noaspect",'bilogarithmic=',
                                       "contour", "levels="])

        if ("--help", '') in opts:
            raise getopt.GetoptError("")

        if len(args) > 0 : dataFileName = args[0]
        else : dataFileName = "scalars.out"

        fig = plt.figure(figsize=(20,5))
        #ax = fig.add_subplot(111, frame_on = False)
        ax = fig.add_subplot(111)

        #ax.xaxis.set_major_locator(mticker.NullLocator())
        #ax.yaxis.set_major_locator(mticker.NullLocator())

        opts = dict(opts)
        plotg(dataFileName, **opts)

        plt.show()
    except getopt.GetoptError:
        usage()

