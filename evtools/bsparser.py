#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import matplotlib
import matplotlib.pyplot as plt
import textwrap
import numpy as np

import knodat.multimap as kmm

module_logger = logging.getLogger("BSParser")
module_logger.propagate = False

formatter = logging.Formatter(
        fmt = "%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s" )

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

module_logger.addHandler(ch)
module_logger.setLevel(logging.WARNING)

def set_debug_level(level):
    possible_levels = dict(
            debug = logging.DEBUG, info = logging.INFO,
            warning = logging.WARNING, error = logging.ERROR,
            fatal = logging.FATAL)
    ch.setLevel(possible_levels[level])
    module_logger.setLevel(possible_levels[level])
    kmm.set_debug_level(level)

if __name__ == '__main__':
    # for the plots themselves
    plt.figure()
    plt.xlabel(r'$k[\frac{1}{a}]$')
    plt.ylabel(r'$E[t]$')

# organizing the data

show_figure = True
save_eps    = False
save_pdf    = False
save_png    = False
modify_xlim = False
modify_ylim = False

class BSParser:
    def __init__( self, _filename=None ):
        self.bs = kmm.MultiMap()
        self.bands = dict()

        if _filename != None:
            self.parse( _filename )

    def parse( self, _filename ):
        """parses a given input file and creates the corresponding multimap"""
        module_logger.info("parsing file %s" % _filename)
        bs_file = open(_filename, 'r')

        self.number_of_columns = len( bs_file.readline().split(" ") )-1
        columns = [ '_k' ]
        for i in range( 1, self.number_of_columns+1 ):
            columns.append( ('_b%s' % i ) )
            self.bands[i] = ('_b%s' % i )

        module_logger.info("... counting %i bands" % self.number_of_columns)

        self.bs.set_column_names(*columns)
        self.bs.read_file(_filename)

    def plot(self, color = None, fmt = "", xscaling = 1.0, yscaling = 1.0, 
            markersize = 4.0, ymin = -100, ymax = 100, shiftx = 0.0, shifty = 0.0):
        self.bs.sort('_k')
        kvals = xscaling * self.bs.get_column( '_k' ) + shiftx

        line = ""
        for iband,band in self.bands.items():
            y = yscaling * self.bs.get_column(band)

            Evals= np.array([x if (x >= ymin and x <= ymax) else np.nan for x in y]) + shifty

            if color == None: line, = plt.plot(kvals, Evals, fmt, clip_on = True)
            else: line, = plt.plot(kvals, Evals, color = color, clip_on = True)
            line.set_markersize(markersize)

    def mark( self, index ):
        plt.lines[index-1].set_marker('o')
        plt.draw()

    def unmark( self, index ):
        diag.lines[index-1].set_marker('None')
        plt.draw()

    def hide( self, index):
        unmark(index)
        diag.lines[index-1].set_linestyle('None')
        plt.draw()

    def hideAll( self ):
        for i in range(len(diag.lines) ): hide( i + 1 )

    def show( self, index ):
        diag.lines[index-1].set_linestyle('-')
        plt.draw()

    def showAll( self ):
        for i in range(len(diag.lines) ): show( i + 1 )

    def getMarkedBands( self ):
        result = []
        for i in range(len(diag.lines)): 
            if diag.lines[i].get_marker() == 'o': result.append( i+1 )
        return result

    def getHiddenBands( self ):
        result = []
        for i in range(len(diag.lines)): 
            if diag.lines[i].get_linestyle() == 'None': result.append( i+1 )
        return result
    
    def showBS( self ):
        plt.show()

    def bandgap(self, threshold = 1e-6):
        """ calculates the bandgap(s)
            We therefore distinguish between bands (as conduction or
            valuence band) and dispersions of the single modes.
            The single modes are checked for the energy for the
            energy interval they're occupying, then they are compared with
            the given bands. Three cases can occur:
            1) no overlap with any of the given bands
               -> create a new band
            2) finite overlap with one band
               -> assign the mode to this band, possibly extending its range
            3) finite overlap with several bands
               -> combine all those bands, with a possibly increased range
        """

        bands = []

        for i in range( 1, self.number_of_columns+1 ):
            column_name = ( '_b%i' % i )
            
            lower_limit = self.bs.get_minimum_value(column_name)
            upper_limit = self.bs.get_maximum_value(column_name)

            overlaps_with = []
            for iband, band in enumerate(bands):
                if lower_limit > band[0] and lower_limit < band[1]:
                    overlaps_with.append(iband)
                elif upper_limit > band[0] and upper_limit < band[1]:
                    overlaps_with.append(iband)
                elif band[0] > lower_limit and band[0] < upper_limit:
                    overlaps_with.append(iband)
                elif band[1] > lower_limit and band[1] < upper_limit:
                    overlaps_with.append(iband)
                else:
                    pass

            # helping function defined here as for limited validity
            def assign_interval_to_band(interval, band_index, bands):
                overlapping_band = bands[band_index]
                new_lower_band_limit = min(interval[0], overlapping_band[0])
                new_upper_band_limit = max(interval[1], overlapping_band[1])

                bands[band_index] = (new_lower_band_limit, new_upper_band_limit)
                return (new_lower_band_limit, new_upper_band_limit)


            current_interval = (lower_limit, upper_limit)
            if len(overlaps_with) == 0:
                bands.append(current_interval)
            elif len(overlaps_with) == 1:
                assign_interval_to_band(current_interval, overlaps_with[0], bands)
            else:
                overlaps_with.reverse()

                for iband in overlaps_with:
                    current_interval = assign_interval_to_band(current_interval, iband, bands)
                    del bands[iband]

                bands.append(current_interval)

        # now sort the bands by their minimum
        bands.sort(key = lambda interval: interval[0])

        gaps = []
        for iband, band in enumerate(bands[:-1]):
            try:
                gap = bands[iband + 1][0] - band[1]

                # we're only interested in bands above a given threshold
                if gap > threshold:
                    gaps.append(gap)
            except:
                raise
        
        return bands, gaps

def usage():
    print textwrap.dedent("""\
    Usage: bsparser.py [OPTIONS] ... datafilename
    creates a plot using the first column of datafile as the x-axis and all
    the other columns as y-values, as in a bandstructure of periodic
    quantum structures

    Mandatory arguments to long options are mandatory for short options too.
        --gaps, -g              determines the bands and gaps and plots them
        --eps                   saves the plot as an eps file
        --pdf                   saves the plot as a pdf file
        --png                   saves the plot as a png file
        --xlim=LOWER:UPPER
        --ylim=LOWER:UPPER
        --title=STRING
        --color=VALUE           By default all the lines are plotted in
                                different colors. This option defines one color
                                for all lines
        --fmt=VALUE             The VALUE defines the format of the plotted lines,
                                including color, shape, marker style etc.
        --debug                 activate debug output
        --info                  activate info output, notice that this is also
                                activated by the --debug option, if both are given
                                the last one given dominates
        --help                  prints this screen and exits
    """)


if __name__ == '__main__':
    import getopt, sys

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'g', 
                                   ['eps','pdf','png','xlim=','ylim=','title=',
                                    'color=','fmt=', "debug", "info", "help",
                                    "gaps", "rescalex=", "rescaley="])
        if len(args) > 0 : dataFileName = args[0]
        else : dataFileName = "transmission.out"

        if ("--help", '') in opts:
            raise getopt.GetoptError("")

        
        # if asked for, set debug or info output
        if ("--info", '') in opts:
                set_debug_level("info")
        if ("--debug", '') in opts:
                set_debug_level("debug")

        calculate_gaps = False

        # preprocessing
        plotting_options = {}
        for opt,val in opts:
            module_logger.debug( "%s: %s" % ( opt, val ) )
            if opt == "--color":
                plotting_options["color"] = val
            if opt == "--fmt":
                plotting_options["fmt"] = val

            if opt == "--gaps" or opt == "-g":
                calculate_gaps = True

            if opt == "--rescalex":
                plotting_options["xscaling"] = float(val)
            if opt == "--rescaley":
                plotting_options["yscaling"] = float(val)


        bs = BSParser()
        bs.parse( dataFileName )
        bs.plot(**plotting_options)

        if calculate_gaps == True:
            print bs.bandgap()

        # postprocessing
        #######################################################################
        for opt,val in opts:
            if opt == "--xlim": 
                module_logger.info("setting xlimits to %s" % val)
                xlimits = val.split( ":" )
                plt.xlim( float( xlimits[0] ), float( xlimits[1] ) )
            if opt == "--ylim": 
                module_logger.info("setting ylimits to %s" % val)
                ylimits = val.split(":")
                plt.ylim(float( ylimits[0]), float(ylimits[1]))
            if opt == "--title":
                logging.info("setting title to %s" % val)
                plt.title(val)
        
        # postprocessing, a second step
        #######################################################################
        for opt,val in opts:
            if opt == "--pdf":
                module_logger.info("saving to pdf")
                outfileName = dataFileName.replace(".dat",".pdf")
                plt.savefig(outfileName)
            if opt == "--eps":
                module_logger.info("saving to eps")
                outfileName = dataFileName.replace(".dat",".eps")
                plt.savefig(outfileName)
            if opt == "--png":
                module_logger.info("saving to png")
                outfileName = dataFileName.replace(".dat",".png")
                plt.savefig(outfileName)

        plt.show()
    
    except getopt.GetoptError:
        usage()
