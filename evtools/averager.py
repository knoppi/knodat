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
constants = {}

# assuming simple 1/L-diffusive behaviour
fit_diffT = lambda p, x: constants['_c1'] / (1.0 + x/p[0] )
err_diffT = lambda p, x, y: fit_diffT(p, x) - y
diffusion_parameters = [1000]

# standard function for fitting spin transport is exponentially damped cosine
# with the damping as parameter 0 and the frequency as parameter 1
# we need limits for the validity of the function they are set to some thing
constants['limit_exp_spin_lower'] = 10
constants['limit_exp_spin_upper'] = 20000
fit_spinT = lambda x, p: p[1] * np.exp(-x/p[0])
err_spinT = lambda p, x, y: fit_spinT(p, x) - y
err_spinT = lambda p, x, y: np.piecewise( x, 
    [ x < constants['limit_exp_spin_lower'], x > constants['limit_exp_spin_upper'] ],
    [ y, y, fit_spinT ], p ) - y
spinTransport_default_parameters = [1000, 1.0]

expDecay = lambda l_decay, x: constants['_c1'] + exp( -x / l_decay )
expDecay_dev = lambda p, x, y: y - expDecay( p[0], x )
expDecay_params = [100]

def calculateSpinTransmission( dataFileName, 
        columns = ['_L', '_T', '_Tuu', '_Tdu', '_Tud', '_Tdd', '_c1' ],
        colsToAverage = ["_T", '_Tspin', '_Tspin2'],
        colsToFit = ["_T","_Tspin"],
        xCol = "_L",
        shapeTransmission = [err_diffT, err_spinT], 
        shapeParameters = [diffusion_parameters, spinTransport_default_parameters],
        generalInformation = ['_c1']):
    module_logger.debug( "calculateSpin function" )
    # filename for evaluated output
    outfileName = dataFileName.replace(".out",".dat")
    T = MultiMap()
    T.set_column_names( *columns )
    T.read_file( dataFileName )

    spinTransmission = lambda uu,du,t : ( uu - du ) / t
    neededColumns = ['_Tuu', '_Tdu', '_T']
    newName = "_Tspin"
    T.add_column( newName, origin = neededColumns,
            connection = spinTransmission )

    spinTransmission = lambda uu,du : du / uu
    neededColumns = ['_Tuu', '_Tdu']
    newName = "_Tspin2"
    T.add_column( newName, origin = neededColumns,
            connection = spinTransmission )

    return calculateFromObject( T, outfileName,
            colsToAverage = colsToAverage,
            colsToFit = colsToFit, xCol = xCol,
            shapeTransmission = shapeTransmission,
            shapeParameters = shapeParameters,
            generalInformation = generalInformation )

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
        shapeTransmission = [err_diffT], 
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
