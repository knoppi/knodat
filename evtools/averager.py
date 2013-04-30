#!/usr/bin/env python
# -*- coding: utf-8 -*-

# averager.py
# 2013

import logging
from scipy import *
from scipy import optimize
import numpy as np
from scipy.stats import sem

import knodat.multimap as kmm

# setup of logging, level can be adjusted by set_debug_level
module_logger = logging.getLogger("averager")
formatter = logging.Formatter(
    fmt = "%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s" )

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

module_logger.addHandler(ch)
module_logger.setLevel(logging.WARNING)

# define a global zero
ZERO = 1E-6

def set_debug_level(level):
    possible_levels = dict(debug = logging.DEBUG, info = logging.INFO,
            warning = logging.WARNING, error = logging.ERROR,
            fatal = logging.FATAL)
    ch.setLevel(possible_levels[level])
    module_logger.setLevel(possible_levels[level])
    kmm.set_debug_level(level)


# constants of a transmission curve (like number of channels) can be 
# stored here
###############################################################################
constants = {}

# assuming simple 1/L-diffusive behaviour
###############################################################################
diffT = lambda p, x, N: N / (1.0 + x/p[0] )
diffusion_parameters = [1000]

# if we assume a localized transport regime an exponential decay
# describes the length dependence
###############################################################################
expT = lambda p, x: p[0] * exp(2 * x / p[1])
localization_parameters = [10,10000]

# For the spin transport we also asume an exponential decay
###############################################################################
spinT = lambda p, x, D, N: p[0] * N * p[1] / 2.0 / D * exp( - x / p[1])
#spinT = lambda p, x, D: p[0] * exp( - x / p[1])
spinT_parameters = [1, 100]

# specialized function to do all the fancy averaging stuff with the particular
# goal to get insight into spin transport of a given system
###############################################################################
def calculate_spin_transmission(
        dataFileName, 
        columns = ['L', 'T', 'Tuu', 'Tud', 'Tdu', 'Tdd', 'c1']
        ):
    # assign some names to interesting values
    colsToAverage = ["T", 'Tsu', 'Tsd']
    colsToFit = ["T","Tsu","Tsd"]
    xCol = "L"

    # the input data
    ###########################################################################
    T = kmm.MultiMap()
    T.set_column_names(*columns)
    T.read_file( dataFileName )

    # add new columns for spin transmission, spin relaxation and
    # potentially localized charge transport
    ###########################################################################
    spinTransmission = lambda sc,sf: sc - sf
    
    neededColumns = ['Tuu', 'Tdu']
    newName = "Tsu"
    T.add_column(newName, origin = neededColumns, connection = spinTransmission)

    neededColumns = ['Tdd', 'Tud']
    newName = "Tsd"
    T.add_column(newName, origin = neededColumns, connection = spinTransmission)

    # do the averaging
    ###########################################################################
    outfileName = dataFileName.replace(".out",".dat")
    calculateFromObject(T, outfileName, colsToAverage = colsToAverage,
        colsToFit = [], xCol = xCol)

    # load the file with the averaged data
    ###########################################################################
    O = kmm.MultiMap(outfileName)

    # fetch the number of open channels
    array_of_Ns = T.get_column("c1")
    average_N = array_of_Ns.mean()
    variance_N = np.sqrt(np.mean((array_of_Ns - average_N)**2))

    if not variance_N == 0:
        module_logger.warning("number of channels not constant")

    
    # do fitting
    ###########################################################################
    xvals = O.get_possible_values(xCol)

    # charge transport for mean-free path and diffusion constant
    yvals = O.get_column("T")
    err = lambda p, x, y: y - diffT(p, x, average_N)
    start_parameters = diffusion_parameters

    final_parameters, success = optimize.leastsq(
            err, start_parameters, args = (xvals, yvals))

    # l is the RMT-transport length in units of a
    l = final_parameters[0]
    # rescaling according to scaling theory gives the transport theory mean
    # free path, again in units of a
    ltr = 2 * l / np.pi
    # the diffusion constant can be calculated as D = v_F * ltr / 2
    # to stay conform with the units we use v_F = 1
    # so D is given in units of v_F * a
    D = ltr / 2.0

    # spin transport
    err = lambda p, x, y: y - spinT(p, x, D, average_N)
    start_parameters = spinT_parameters

    # spin up
    yvals = O.get_column("Tsu")
    final_parameters, success = optimize.leastsq(
            err, start_parameters, args = (xvals, yvals))
    Cu = final_parameters[0]
    LSu = final_parameters[1]

    # spin down
    yvals = O.get_column("Tsd")
    final_parameters, success = optimize.leastsq(
            err, start_parameters, args = (xvals, yvals))
    Cd = final_parameters[0]
    LSd = final_parameters[1]

    # Now the fit parameters of the two spin components are compared
    # If the deviation is larger than 5%, a warning is plotted:
    deviationC = np.abs(Cu / Cd - 1)
    deviationLS = np.abs(LSu / LSd - 1)
    module_logger.info("deviation between spin up and down channel: %g / %g" % (deviationC, deviationLS))
    if deviationC > 0.05 or deviationLS > 0.05:
        module_logger.warning("results for spin up and spin differ by" 
                "DC =  %i %% and DLS = %i %%, respectively" % 
                (deviationC * 100, deviationLS * 100))

    C = (Cu + Cd) / 2.0
    LS = (LSu + LSd) / 2.0

    # LS is given in units of a, we also calculate tau_s which according 
    # to our definition of D then is given in a / v_F
    tauS = LS**2 / D

    result = (average_N, ltr, D, C, LS, tauS)
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
        output_columns.append(y)
        output_columns.append("error%s" % y)
        output_columns.append("rms%s" % y)
    
    output_columns.append("ensemble_size")

    module_logger.debug(", ".join(output_columns))
    O.set_column_names(*output_columns)

    xvals = T.get_possible_values(xCol)

    for value in generalInformation:
        constants[value] = T.get_possible_values(value)[0]

    N = 0
    for x0 in xvals:
        current_restrictions={xCol : x0}

        outputline = [ x0 ]
        for y in colsToAverage:
            yvals = T.get_column_hard_restriction(y, **current_restrictions)
            N = yvals.shape[0]
            yaverage = np.average(yvals)
            yerror = np.std(yvals) / np.sqrt(N)
            rms = np.sqrt(np.mean((yvals - yaverage)**2))
            outputline.extend([yaverage, yerror, rms])

        outputline.append(N)


        O.append_row(outputline)

    module_logger.info("ensemble size: %i" % N)
    
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
    T = kmm.MultiMap()  # input data

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
