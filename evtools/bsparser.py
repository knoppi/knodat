#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger( "BSParser" )

formatter = logging.Formatter(
            fmt = "%(relativeCreated)d -- %(name)s -- %(levelname)s -- %(message)s" )

fh = logging.FileHandler( 'eval.log' )
fh.setFormatter( formatter )
fh.setLevel( logging.DEBUG )

logger.addHandler( fh )

ch = logging.StreamHandler()
ch.setLevel( logging.DEBUG )

logger.addHandler( ch )

logger.setLevel( logging.DEBUG )

import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # for the plots themselves

    plt.figure()
    plt.xlabel(r'$k[\frac{1}{a}]$')
    plt.ylabel(r'$E[t]$')

# organizing the data
import knodat.multimap as kmm

debug = True

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
        bs_file = open( _filename, 'r' )

        self.number_of_columns = len( bs_file.readline().split(" ") )-1
        columns = [ '_k' ]
        for i in range( 1, self.number_of_columns+1 ):
            columns.append( ('_b%s' % i ) )
            self.bands[i] = ('_b%s' % i )

        self.bs.set_column_names( *columns )
        self.bs.read_file( _filename )

    def plot(self, color = None, fmt = "", xscaling = 1.0, markersize = 4.0):
        kvals = xscaling * self.bs.get_column( '_k' )

        line = ""
        for iband,band in self.bands.items():
            Evals = self.bs.get_column( band )
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

if __name__ == '__main__':
    import getopt, sys

    opts, args = getopt.getopt(sys.argv[1:], '', 
                               ['eps','pdf','png','xlim=','ylim=','title=',
                                'color=','fmt='])
    if len(args) > 0 : dataFileName = args[0]
    else : dataFileName = "transmission.out"

    logging.debug( "%s" % (opts, ) )

    plotting_options = {}

    for opt,val in opts:
        logging.debug( "%s: %s" % ( opt, val ) )
        if opt == "--pdf": save_pdf = True
        elif opt == "--eps": save_eps = True
        elif opt == "--png": save_png = True
        elif opt == "--xlim": 
            modify_xlim = True
            xlim = val
        elif opt == "--ylim": 
            modify_ylim = True
            ylim = val
        elif opt == "--title":
            print "setting title"
            plt.title( val )
        else:
            option_name = opt
            while option_name[0] == "-":
                option_name = option_name[1:]
            plotting_options[option_name] = val


    bs = BSParser()
    bs.parse( dataFileName )
    bs.plot(**plotting_options)

    if modify_xlim:
        logger.info( "setting xlimits to %s" % xlim )
        xlimits = xlim.split( ":" )
        plt.xlim( float( xlimits[0] ), float( xlimits[1] ) )

    if modify_ylim:
        logger.info( "setting ylimits to %s" % ylim )
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

    plt.show()
