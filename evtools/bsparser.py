#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import matplotlib
import matplotlib.pyplot as plt
import textwrap

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
            markersize = 4.0):
        kvals = xscaling * self.bs.get_column( '_k' )

        line = ""
        for iband,band in self.bands.items():
            Evals = yscaling * self.bs.get_column(band)
            if color == None: line, = plt.plot(kvals, Evals, fmt)
            else: line, = plt.plot( kvals, Evals, color = color )
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

    def bandgap( self ):
        """ calculates the bandgap(s) """
        gap = list()
        highest_valence_band = 0
        lowest_conduction_band = 0

        # lists containing the bands' boundaries
        lower_bounds = DataArray()
        upper_bounds = DataArray()

        # iteration over all the bands
        for i in range( 1, self.number_of_columns+1 ):
            column_name = ( '_b%i' % i )
            this_band = DataArray( self.bs.get_column( column_name ) )

            lower_bound_of_this_band = this_band.minValue()
            upper_bound_of_this_band = this_band.maxValue()

            lower_bounds.append( lower_bound_of_this_band )
            upper_bounds.append( upper_bound_of_this_band )

        # container for the results
        gaps = []
        final_bands = []

        current_lower = lower_bounds.minValue()
        current_upper = upper_bounds[lower_bounds.minKey()]

        while lower_bounds.numberOfEntries() > 0:
            this_band_index = lower_bounds.minKey()
            this_band_lower = lower_bounds.minValue()
            this_band_upper = upper_bounds[this_band_index]
            if this_band_lower > current_upper:
                # new gap found:
                gaps.append( this_band_lower - current_upper )
                final_bands.append( (current_lower, current_upper ) )
                current_lower = this_band_lower
                current_upper = this_band_upper
                lower_bounds.remove( this_band_index )
                upper_bounds.remove( this_band_index )
                continue
            elif this_band_lower <= current_upper:
                if this_band_upper <= current_upper:
                    # nothing interesting
                    lower_bounds.remove( this_band_index )
                    upper_bounds.remove( this_band_index )
                    continue
                elif this_band_upper > current_upper:
                    # shift current upper
                    current_upper = this_band_upper
                    lower_bounds.remove( this_band_index )
                    upper_bounds.remove( this_band_index )

        return final_bands, gaps

def usage():
    print textwrap.dedent("""\
    Usage: bsparser.py [OPTIONS] ... datafilename
    creates a plot using the first column of datafile as the x-axis and all
    the other columns as y-values, as in a bandstructure of periodic
    quantum structures

    Mandatory arguments to long options are mandatory for short options too.
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
        opts, args = getopt.getopt(sys.argv[1:], '', 
                                   ['eps','pdf','png','xlim=','ylim=','title=',
                                    'color=','fmt=', "debug", "info", "help"])
        if len(args) > 0 : dataFileName = args[0]
        else : dataFileName = "transmission.out"

        if ("--help", '') in opts:
            raise getopt.GetoptError("")

        
        # if asked for, set debug or info output
        if ("--info", '') in opts:
                set_debug_level("info")
        if ("--debug", '') in opts:
                set_debug_level("debug")

        # preprocessing
        plotting_options = {}
        for opt,val in opts:
            module_logger.debug( "%s: %s" % ( opt, val ) )
            if opt == "--color":
                plotting_options["color"] = val
            if opt == "--fmt":
                plotting_options["fmt"] = val


        bs = BSParser()
        bs.parse( dataFileName )
        bs.plot(**plotting_options)

        # postprocessing
        for opt,val in opts:
            module_logger.debug( "%s: %s" % ( opt, val ) )
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
            if opt == "--xlim": 
                module_logger.info("setting xlimits to %s" % val)
                xlimits = vals.split( ":" )
                plt.xlim( float( xlimits[0] ), float( xlimits[1] ) )
            if opt == "--ylim": 
                module_logger.info("setting ylimits to %s" % vals)
                ylimits = vals.split(":")
                plt.ylim(float( ylimits[0]), float(ylimits[1]))
            if opt == "--title":
                logging.info("setting title to %s" % val)
                plt.title(val)

        plt.show()
    
    except getopt.GetoptError:
        usage()
