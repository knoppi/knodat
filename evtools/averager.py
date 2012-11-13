#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from scipy import *
from scipy import optimize
import numpy as np
from scipy.stats import sem

import knodat.multimap as kmm

module_logger = logging.getLogger( "evaluation.averaging" )
# constants of a transmission curve (like number of channels) can be 
# stored here
###############################################################################
constants = {}

# assuming simple 1/L-diffusive behaviour
###############################################################################
diffT = lambda p, x: constants['c1'] / (1.0 + x/p[0] )
diffusion_parameters = [1000]

# if we assume a localized transport regime an exponential decay
# describes the length dependence
###############################################################################
expT = lambda p, x: p[0] * exp(2 * x / p[1])
localization_parameters = [10,10000]

# For the spin transport we also asume an exponential decay
###############################################################################
spinT = lambda p, x: p[0] * exp( - x / p[1])
spinT_parameters = [0.5, 1000]

# specialized function to do all the fancy averaging stuff with the particular
# goal to get insight into spin transport of a given system
###############################################################################
def calculate_spin_transmission(dataFileName, 
        columns = ['L', 'T', 'Tuu', 'Tdu', 'Tud', 'Tdd', 'c1' ],
        colsToAverage = ["T", 'polarization', 'relaxationrate', "Ttemp"],
        colsToFit = ["T","Tlog","polarization"],
        xCol = "L",
        shapeTransmission = [diffT, expT, spinT], 
        shapeParameters = [diffusion_parameters, localization_parameters, 
            spinT_parameters],
        generalInformation = ['c1']):
    module_logger.debug("calculating charge and "
            "spin transmission from the input file %s" % dataFileName)
    
    # the input data
    ###########################################################################
    T = kmm.MultiMap()
    T.set_column_names( *columns )
    T.read_file( dataFileName )

    # add new columns for spin transmission, spin relaxation and
    # potentially localized charge transport
    ###########################################################################
    spinTransmission = lambda uu,du,t : ( uu - du ) / t
    neededColumns = ['Tuu', 'Tdu', 'T']
    newName = "polarization"
    T.add_column( newName, origin = neededColumns,
            connection = spinTransmission )

    relaxationrate = lambda uu,du : du / uu
    neededColumns = ['Tuu', 'Tdu']
    newName = "relaxationrate"
    T.add_column( newName, origin = neededColumns,
            connection = relaxationrate )

    # and the logarithm of the transmission
    T.add_column('Ttemp', origin = ['T'], connection = log)

    # do the averaging
    ###########################################################################
    outfileName = dataFileName.replace(".out",".dat")
    calculateFromObject(T, outfileName,
            colsToAverage = colsToAverage,
            colsToFit = [], xCol = xCol,
            generalInformation = generalInformation)

    # load the file with the averaged data
    ###########################################################################
    O = kmm.MultiMap(outfileName)
    
    # add one new row for the localized regime
    O.add_column('Tlog', origin = ["Ttemp"], connection = exp)

    # do fitting
    result = []
    xvals = O.get_possible_values( xCol )
    
    for i, column in enumerate(colsToFit):
        yvals = O.get_column(column)
        
        module_logger.debug("xvals.shape: %s" % (xvals.shape,))
        module_logger.debug("yvals.shape: %s" % (yvals.shape,))

        err = lambda p, x, y: y - shapeTransmission[i](p, x)
        par = shapeParameters[i]
        args = (xvals, yvals)
        
        output, success = optimize.leastsq(err, par, args = args)
        result.append(output)

    result.append(constants)

    return result

def calculateFromObject(T, outfileName, colsToAverage = ["_T"], 
                        colsToFit = ["_T"], xCol = "_L", 
                        shapeTransmission = [], 
                        shapeParameters = [], 
                        generalInformation = []):
                        # war vorher:
                        #shapeTransmission = [err_diffT], 
                        #shapeParameters = [diffusion_parameters], 
                        #generalInformation = ['_c1']):
    O = kmm.MultiMap()  # output data

    # output columns are
    output_columns = [xCol]
    for y in colsToAverage:
        output_columns.append( y )
        output_columns.append( "error%s" % y )
        output_columns.append( "rms%s" % y )

    module_logger.debug( ", ".join( output_columns ) )
    O.set_column_names( *output_columns )

    xvals = T.get_possible_values( xCol )

    for value in generalInformation:
        constants[value] = T.get_possible_values( value )[0]

    for x0 in xvals:
        current_restrictions={ xCol : x0 }

        outputline = [ x0 ]
        for y in colsToAverage:
            yvals = T.get_column_hard_restriction(y, **current_restrictions)
            yaverage = np.average(yvals)
            #yerror   = np.std( yvals )
            yerror   = sem(yvals)
            rms = np.sqrt(np.mean((yvals - yaverage)**2))
            outputline.extend([yaverage, yerror, rms])

        O.append_row( outputline )
        #O.appendRow( np.array( tuple( outputline ), ndmin=2 ) )

    # write averaged data
    O.write_file( outfileName )

    result = []

    # do fitting
    for i, column in enumerate(colsToFit):
        module_logger.debug("xvals.shape: %s" % (xvals.shape,))
        yvals = O.get_column(column)
        module_logger.debug("yvals.shape: %s" % (yvals.shape,))
        output, success = optimize.leastsq(shapeTransmission[i],
                shapeParameters[i], args = (xvals, yvals))
        result.append(output)

    #result.append( constants )
    #print result
    return result

def calculate( dataFileName, cols = ['_L','_T','_c1'], 
        colsToAverage = ["_T"],
        colsToFit = ["_T"],
        xCol = "_L",
        shapeTransmission = [diffT], 
        shapeParameters = [diffusion_parameters],
        generalInformation = ['_c1']
        ):
    # filename for evaluated output
    outfileName = dataFileName.replace(".out",".dat")

    module_logger.debug( "%s --> %s" % ( dataFileName,  outfileName ) )
    T = MultiMap()  # input data

    T.set_column_names( *cols )
    T.read_file( dataFileName )

    return calculateFromObject( T, outfileName, colsToAverage = colsToAverage, 
            colsToFit = colsToFit, shapeTransmission = shapeTransmission, 
            shapeParameters = shapeParameters, xCol = xCol,
            generalInformation = generalInformation )

if __name__ == '__main__':
    import getopt, sys
    opts, args = getopt.getopt( sys.argv[1:], '', [] )
    if len(args) > 0 : dataFileName = args[0]
    else : dataFileName = "transmission.out"

    calculate( dataFileName )
