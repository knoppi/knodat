#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as ssignal
import matplotlib.pyplot as plt
import logging

# I set the logging to both logging to std out and to a file with
# two different logging levels.
module_logger = logging.getLogger("multimap")
formatter = logging.Formatter(
    fmt = "%(relativeCreated)d -- %(name)s -- %(levelname)s -- %(message)s" )

#fh = logging.FileHandler('multimap.log')
#fh.setFormatter(formatter)
#fh.setLevel(logging.DEBUG)
#module_logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)
#ch.setLevel(logging.DEBUG)

module_logger.addHandler(ch)

module_logger.setLevel(logging.DEBUG)

# define a global zero
ZERO = 1E-6

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    module_logger.debug("called gauss_kern with size = %i" % size)
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


class MultiMap:
    def __init__(self, _fileName=None, _cols=None):
        """initialization of the Multimap"""
        self.commentIndicators = ['#']
        self.columns = []
        self.dataType = []
        if not _cols == None:
            self.set_column_names(*_cols)
        if _fileName is not None:
            self.read_file( _fileName )

        module_logger.debug("MultiMap initialization:")
        module_logger.debug("filename: %s" % _fileName)
        module_logger.debug("columns: %s" % (self.columns, ))
        module_logger.debug("data type: %s" % self.dataType)

    def __del__(self):
        module_logger.info("multimap deleted")

    def __getitem__( self, i ):
        """ retrieve a single row of the MultiMap as a dict """
        tmp = {}
        for j in range(len(self.columns)):
            tmp[self.columns[j]] = self.data[i,j]
        return tmp

    def getitem( self, i ):
        """ retrieve a single row of the MultiMap as a dict """
        self.__getitem__( i )

    def set_column_names(self, *keyw, **options):
        """sets the names of the columns and - if needed - the
        corresponding dtype, this will also clear the data,
        a new filling is required"""
        # easy thing is setting the array of the column names
        self.columns = [x for x in keyw]

        # Now define the data type.
        new_data_type = []

        if "dType" in options.keys():
            standardType = options["dType"]
        else:
            standardType = "|f8"

        for i in keyw:
            new_data_type.append((i, standardType))

        module_logger.debug(new_data_type)

        self.set_data_type(new_data_type)

    def set_data_type(self, new_dataType):
        """sets the data type of the internal data storage and
        creates a new empty np-array"""
        self.dataType = new_dataType
        self.data = np.zeros( 
                (0, len(new_dataType)), 
                dtype = new_dataType)

    def append_row(self, row):
        """row is some iterable object which somehow has to fit to dtype"""
        new_row = np.array(tuple(row), dtype=self.dataType)
        self.data = np.append( self.data, new_row )

    def read_file(self, filename, **options):
        """reads data from ascii file"""
        for (key,val) in options:
            if key == "delimiter": 
                self.separator = val
            if key == "fieldnames": 
                self.set_column_names(val)
            if key == "skipinitialspaces": 
                skipInitialSpaces = val


        # now three cases can be thought of:
        # 1) we already know the column names
        # 2) we do not know the but in the first line of the file
        #    there is a hint line (beginning with ##)
        # 3) neither of the cases
        # In case 1 len(self.columns) > 1 and we can directly go to reading
        # the file. In case 2 we read the magic line to get the column names.
        # In the last case we look for the first line not beginning with
        # a comment indicator to count the number of columns and name them
        # just with numbers
        if len(self.columns) == 0:
            file_id = open(filename,'rb')
            first_row = file_id.readline()
            if first_row[0:2] == "##":
                column_names = first_row[3:(len(first_row) - 1)].split(" ")
                self.set_column_names(*column_names)
            else:
                while first_row[0:1] in self.commentIndicators:
                    first_row = file_id.readline()
                number_of_cols = len(first_row.strip().split(" "))
                module_logger.debug(
                        "content of first row: %s" %
                        first_row)
                module_logger.info("number of columns: %i" % number_of_cols)
                column_names = [str(x) for x in range(1, number_of_cols + 1)]
                self.set_column_names(*column_names)

        self.data = np.loadtxt(filename, dtype = self.dataType)

        module_logger.debug("finished reading file")

    def write_file(self, filename, **options):
        """writes the data to a file called 'filename'"""
        outfile = open(filename, 'w')

        # head line containing the description of the file, i.e. the
        # dolumn names
        column_names = ' '.join( ["%s" % k for k in self.columns])
        line = "".join((self.commentIndicators[0], self.commentIndicators[0], 
                        " ", column_names, "\n"))
        outfile.write( line )

        # write data line by line
        for iline in range( self.data.shape[0] ):
            #line = " ".join(np.str_(self.data[iline]))
            line = ""
            for x in self.data[iline]:
                line += np.str_( x ) + " "
                #line += ( "%.10f " % x )
            line = line+"\n"
            outfile.write( line )

        module_logger.debug("finished writing file")

    def write_file_numpy_style(self, filename):
        '''writes the content of the MultiMap in numpy-style .npy file'''
        np.save(filename, self.data)

    def read_file_numpy_style(self, filename):
        '''loads content into the MultiMap which was formerly saved as a
        numpy style .npy file'''
        self.data = np.load(filename)

    def get_minimum_value(self, column, absolute = False):
        '''finds the element in column with the least (absolute) value'''
        if absolute is True:
            return np.amin(np.abs(self.data[:][column]))
        else:
            return np.amin(self.data[:][column])

    def get_x_of_minimum_value(self, x_column, y_column, absolute = False):
        '''returns value of x_column where (absolute) of y_column has its
           minimum'''
        y_value = self.get_minimum_value(y_column, absolute)
        restriction = {y_column: y_value}
        x_values = self.get_column_hard_restriction(x_column, **restriction)

        return (x_values, y_value)

    def get_column_by_index(
            self, _col_index, _col2_indices = [], _lambda = None, 
            _deletion = False):
        """returns array containing content of _col_index where the 
        columns in _col2_index fulfill the corresponding condition in 
        _lambda _col_index is a single integer, _col2_index is an 
        array of integers, _lambda is a function taking two arguments: 
        the index of the current restriction and the corresponding 
        restriction.  Example:
        #def restrictions(n, x):
            #if n == 0: return (x>0)
            #if n == 1: return (x<0)
        desired_values = example_multi_map.getColumnGeneral(
            1, [2,3], restrictions) 
           
        this piece of code fetches all values in column 1, where column 
        2 contains a positive value and column 3 contains a negative value
        """
        col_name = self.columns[_col_index]
        col2_names = []
        for index in _col2_indices:
            col2_names.append( self.columns[index] )
        return self.get_column_general(col_name, col2_names, _lambda, _deletion)

    def get_column_general(self, _col_name, _col2_names = [], _lambda = None, 
                           _deletion = False):
        result = self.data[:]
        indices = np.array( range( self.data.shape[0] ) )

        i = 0
        for rest_name in _col2_names:
            result = result[ np.where( _lambda( i, result[:][rest_name] ) ) ]
            indices = indices[ np.where( _lambda( i, result[:][rest_name] ) ) ]
            i += 1

        if _deletion: 
                self.data = np.delete( self.data, indices, 0 )
        return result[:][_col_name]

    def get_column_hard_restriction(self, desire, **restrictions):
        """ gets the content of column desire under hard (==) restrictions 
        Returns an array according to some hard restriction, i.e. 
        requesting column "desire" where all the key of "restrictions" 
        exactly have the value given by the values of "restrictions"
        """
        restriction_lhs = []
        restriction_rhs = []
        for key in restrictions:
            restriction_lhs.append( key )
            restriction_rhs.append( restrictions[key] )

        restriction_function = lambda n,x: ( x == restriction_rhs[n] )
        return self.get_column_general( 
                desire, restriction_lhs, restriction_function )

    def get_column( self, desire ):
        """ returns column desire without applying any restriction
        """
        return self.get_column_general( desire )

    def get_subset(self, restrictions = {}, _deletion = False):
        """ returns a subset of data with hard restrictions
        """
        module_logger.info(restrictions)
        restriction_lhs = []
        restriction_rhs = []
        for key in restrictions:
            restriction_lhs.append( key )
            restriction_rhs.append( restrictions[key] )

        _lambda = lambda n,x: ( x == restriction_rhs[n] )

        result = self.data[:]

        indices = np.array(range(self.data.shape[0]))

        i = 0
        for rest_name in restriction_lhs:
            result = result[ np.where( _lambda( i, result[:][rest_name] ) ) ]
            indices = indices[ np.where( _lambda( i, result[:][rest_name] ) ) ]
            i += 1

        if _deletion: 
                self.data = np.delete(self.data, indices, 0)
        return result[:]

    def add_column(self, name, dataType = "|f8", 
                   origin = [], connection = None):
        '''adds a new column called "name" with is either just
           zero or is constructed out of the columns listed in
           origin, connected by some function "connection"'''
        new_datatype = self.dataType
        new_datatype.append((name, dataType))

        if connection == None:
            newCol = [np.zeros((self.data.size,))]
        else:
            arguments = []
            for colname in origin:
                arguments.append(self.data[:][colname])
            newCol = connection(*arguments)
            newCol = [newCol]

        # do some strange transformation which should be faster than
        # iterating over the rows
        self.data = np.array(self.data.tolist())
        self.data = self.data.transpose()
        self.data = np.append(self.data, newCol, axis = 0)
        self.data = self.data.transpose()
        self.data = np.rec.fromrecords(self.data, dtype = new_datatype)
        self.data = np.array(self.data)

        self.dataType = new_datatype
        self.columns.append(name)


    def get_possible_values(self, colName, **restrictions):
        """ similar as get_column_general, but duplicates are deleted
        """
        temp = np.unique(
                self.get_column_hard_restriction(colName, **restrictions))
        return temp

    def pull_rows(self, desire, **restrictions):
        """ 
        Returns an array according to some hard restriction, 
        i.e. requesting column "desire" where all the key of 
        "restrictions" exactly have the value given by the values 
        of "restrictions", the rows are cancelled out of the data-array
        """
        idesire = self.columns.index( desire )
        restriction_lhs = []
        restriction_rhs = []
        for key in restrictions:
            restriction_lhs.append( self.columns.index( key ) )
            restriction_rhs.append( restrictions[key] )

        restriction_function = lambda n,x: ( x == restriction_rhs[n] )
        return self.getColumnGeneral( 
                idesire, restriction_lhs, restriction_function, 
                _deletion = True )

    def sort( self, column ):
        """ sorts the MultiMap along column
        """
        self.data = np.sort( self.data, order = column )

    def retrieve_2d_plot_data(self, _colx, _coly, errx = None, erry = None, 
                              restrictions = {}):
        """ TODO has to be rewritten """
        _colx = str(_colx)
        _coly = str(_coly)
        module_logger.debug( 
                "retrieving plot data %s vs %s" % ( _colx, _coly ) )
        module_logger.debug( "errorbars are %s, %s" % ( errx, erry ) )
        self.sort( _colx )
        xvals = self.get_column_hard_restriction( _colx, **restrictions )
        yvals = self.get_column_hard_restriction( _coly, **restrictions )
        xerrs = ( 0 if errx == None else 
                self.get_column_hard_restriction( errx, **restrictions ) )
        yerrs = ( 0 if erry == None else
                self.get_column_hard_restriction( erry, **restrictions ) )

        if errx == None and erry == None:
            return ( xvals, yvals )
        elif errx == None and not erry == None:
            return ( xvals, yvals, yerrs )
        elif not errx == None and erry == None:
            return ( xvals, yvals, xerrs )
        else:
            return ( xvals, yvals, xerrs, yerrs )

    def plot_2d_data(self, _colx, _coly, errx = None, erry = None, 
                     restrictions = {}, label = "", fmt = "", **options):
        '''This method directly plots _colx and _coly with matplotlib

        _colx and _coly are the two columns you wish to plot, additionally
        you can define columns for errorbars setting errx and / or erry.

        With restrictictions you limit the rows to a given set (see 
        retrieve_2d_plot_data for further details on this).

        The plot can de modified by the following arguments:
        - label: gives a label to the drawn line; make sure that the
                 legend of the figure is activated
        - fmt: matplotlib drawing style, directly passed to the function
               call of plot()
        '''
        _colx = str(_colx)
        _coly = str(_coly)
        if errx == None and erry == None:
            xvals, yvals = self.retrieve_2d_plot_data( 
                    _colx, _coly, restrictions = restrictions )
            line = plt.plot( xvals, yvals, fmt, label = label )
        elif errx == None and not erry == None:
            xvals, yvals, yerrs = self.retrieve_2d_plot_data( 
                    _colx, _coly, erry = erry, restrictions = restrictions )
            line = plt.errorbar( xvals, yvals, yerr = yerrs, 
                    label = label, fmt = fmt )
        elif not errx == None and erry == None:
            xvals, yvals, xerrs = self.retrieve_2d_plot_data( 
                    _colx, _coly, errx = errx, restrictions = restrictions )
            line = plt.errorbar( xvals, yvals, xerr = xerrs, 
                    label = label, fmt = fmt )
        else:
            xvals, yvals, xerrs, yerrs = self.retrieve_2d_plot_data( 
                    _colx, _coly, errx = errx, erry = erry, 
                    restrictions = restrictions )
            line = plt.errorbar( xvals, yvals, xerr = xerrs, 
                    yerr = yerrs, label = label, fmt = fmt )
        return line

    def retrieve_3d_plot_data( self, _x, _y, _z, **kwargs ):
        """ returns the needed matrices for creating a matplotlib-like 3d-plot
        """
        restrictions = {}
        if 'restrictions' in kwargs.keys():
            restrictions = kwargs["restrictions"]

        deletion = False;

        data = self.get_subset(restrictions = restrictions)
        data = np.sort(data, order=[_y, _x])

        x = np.unique(data[:][_x])
        y = np.unique(data[::-1][_y])

        # for graphene we have to reduce the x-dimension by a factor of 2
        # TODO: this could be solved in a better way automatically, 
        # but at the moment only graphene is interesting for me
        T,Y = np.meshgrid(x[::2], y)
        X = np.zeros(T.shape)
        X[0::4] = T[0::4]
        X[1::4] = T[1::4] + 0.5
        X[2::4] = T[2::4] + 0.5
        X[3::4] = T[3::4] + 0.5

        extent = ( x.min(), x.max(), y.min(), y.max() )

        #X,Y = np.meshgrid( x, y )
        Z = np.zeros(X.shape)
        #Z *= np.nan
        ##X = np.zeros(Z.shape)
        ##Y = np.zeros(Z.shape)
        # NOTE missing values rot this reshaping, an additional method for that
        # case is the one commented out below, but this is by far less fast
        if len(data[:][_z]) == X.shape[0]*X.shape[1]:
            Z = data[:][_z].reshape(Y.shape)
        else:
            for row in data:
                xi = int(np.where(x == row[_x])[0][0] / 2)
                yi, = np.where(y == row[_y])[0]

                X[yi,xi] = row[_x]
                Y[yi,xi] = row[_y]
                Z[yi,xi] = row[_z]

                xi += 1
                
        g = gauss_kern(2)
        module_logger.debug(g)
        Z = ssignal.convolve(Z, g, 'same')

        return ( X, Y, Z, extent )

    def retrieveQuiverPlotData( self, _x, _y, _u, _v, **kwargs ):
        """ returns the needed matrices for creating a matplotlib-like 3d-plot 
        """
        #if debug: before = time()
        #bin_distance = 5
        #averaging_range_a = 10
        #for i in kwargs:
            #if i == "bin_distance": bin_distance = kwargs[i]
            #if i == "averaging": averaging_range_a = kwargs[i]
        #averaging_range_i = max( int( round( averaging_range_a / bin_distance ) ), 1 )

        #xs = DataArray( self.getPossibleValues( _x ) )
        #ys = DataArray( self.getPossibleValues( _y ) )

        #x = np.arange( xs.minValue() - 1, xs.maxValue() + 1, bin_distance )
        #y = np.arange( ys.minValue() - 1, ys.maxValue() + 1, bin_distance )
        #X,Y = np.meshgrid( x, y )

        #bins_U = np.zeros( X.shape )
        #Ns_U   = np.zeros( X.shape )
        #bins_V = np.zeros( X.shape )
        #Ns_V   = np.zeros( X.shape )

        # fill the bins
        #for row in self:
            #relative_x = row[ _x ] - x[0]
            #relative_y = row[ _y ] - y[0]

            #x_central_index = int( round( relative_x / bin_distance ) )
            #y_central_index = int( round( relative_y / bin_distance ) )
            #x_min_index = max( 0, x_central_index - averaging_range_i )
            #y_min_index = max( 0, y_central_index - averaging_range_i )
            #x_max_index = min( len( x ), x_central_index + averaging_range_i )
            #y_max_index = min( len( y ), y_central_index + averaging_range_i )

            #current_u = row[ _u ]
            #current_v = row[ _v ]

            #for x_index in range( x_min_index, x_max_index ):
                #for y_index in range( y_min_index, y_max_index ):
                    #bins_V[y_index, x_index] = bins_V[y_index, x_index] + current_v
                    #Ns_V[y_index, x_index] = Ns_V[y_index, x_index] + 1
                    #bins_U[y_index, x_index] = bins_U[y_index, x_index] + current_u
                    #Ns_U[y_index, x_index] = Ns_U[y_index, x_index] + 1

        # finish the U and V calculations
        #V = bins_V / Ns_V
        #U = bins_U / Ns_U
        #if debug: print "time for retreiving vector plot data: ", ( time() - before ), "s"

        #return ( X, Y, U, V )

    # Old functions kept due to backwards comapbility; will give a warning
    # about deprecation
    ##########################################################################
    def setColumnNames(self, *keyw, **options):
        '''Deprecated; use set_column_names instead'''
        module_logger.warning( "using deprecated function setColumnNames!" )
        self.set_column_names( *keyw, **options )

    def setDataType(self, new_dataType):
        '''Deprecated; use set_data_type instead'''
        module_logger.warning( "using deprecated function setDataType!" )
        self.set_data_type(new_dataType)

    def readFile(self, filename, **options):
        '''Deprecated; use read_file instead'''
        module_logger.warning( "using deprecated function readFile!" )
        self.read_file(filename, **options)

    def appendRow(self, row):
        '''Deprecated; use append_row instead'''
        module_logger.warning("using deprecated function appendRow!")
        self.append_row(row)

    def addColumn(self, name, dataType = "|f8", origin = [], connection = None):
        '''Deprecated; use add_column instead'''
        module_logger.warning("using deprecated function addColumn!")
        self.add_column(name, dataType, origin, connection)

    def writeFile(self, filename, **options):
        '''Deprecated; use write_file instead'''
        module_logger.warning("using deprecated function writeFile!")
        self.write_file(filename, **options)

    def getIndexedColumnGeneral(self, _col_index, _col2_indices = [],
                                _lambda = None, _deletion = False ):
        '''Deprecated; use get_column_by_index instead'''
        module_logger.warning(
                "using deprecated function getIndexedColumnGeneral!")
        return self.get_column_by_index(_col_index, _col2_indices, 
                                        _lambda, _deletion)

    def getColumnHardRestriction(self, desire, **restrictions):
        '''Deprecated; use get_column_hard_restriction instead'''
        module_logger.warning(
                "using deprecated function getColumnHardRestriction!")
        return self.get_column_hard_restriction(desire, **restrictions)

    def pullRows(self, desire, **restrictions):
        '''Deprecated; use pull_rows instead'''
        module_logger.warning("using deprecated function pullRows!")
        return self.pull_rows(desire, **restrictions)

    def plot2dData(self, _colx, _coly, errx = None, erry = None, 
                   restrictions = {}, label = "", fmt = ""):
        '''Deprecated; use plot_2d_data instead'''
        module_logger.warning("using deprecated function plot2dData!")
        self.plot_2d_data(_colx, _coly, errx = errx, erry = errx, 
                          restrictions = restrictions, label = label, 
                          fmt = fmt)

    def retrieve2dPlotData( self, _colx, _coly, errx = None, erry = None, 
                            restrictions = {} ):
        '''Deprecated; use retrieve_2d_data instead'''
        module_logger.warning("using deprecated function retrieve2dPlotData!")
        return self.retrieve_2d_plot_data(_colx, _coly, 
                                          errx = errx, erry = erry, 
                                          restrictions = restrictions )

    def retrieve3dPlotData( self, _x, _y, _z, **kwargs ):
        '''Deprecated; use retrieve_3d_plot_data instead'''
        module_logger.warning("using deprecated function retrieve3dPlotData!")
        return self.retrieve_3d_plot_data(_x, _y, _z, **kwargs)

    def getColumnGeneral(self, _col_name, _col2_names = [], 
                         _lambda = None, _deletion = False):
        '''Deprecated; use get_column_general instead'''
        module_logger.warning("using deprecated function getColumnGeneral!")
        return self.get_column_general(_col_name, _col2_names, 
                                       _lambda, _deletion)

    def getColumn(self, desire):
        '''Deprecated; use get_column instead'''
        module_logger.warning("using deprecated function getColumn!")
        return self.get_column(desire)

    def getPossibleValues(self, colName, **restrictions):
        '''Deprecated; use get_possible_values instead'''
        module_logger.warning("using deprecated function getPossibleValues!")
        return self.get_possible_values(colName, **restrictions)


if __name__ == "__main__":
    module_logger.info("create multimap with column a, b and c")

    cols = ["a", "b", "c"]
    test_object = MultiMap(_cols = cols)

    module_logger.info("multimap created")

    module_logger.info("fill test object with content")
    test_object.append_row([1, 2, 3])
    test_object.append_row([4, 5, 6])
    test_object.append_row([7, 8, 9])

    module_logger.info("test object filled with: %s" % test_object.data)

    module_logger.info("adding column 'd' filled with zeros")
    test_object.add_column("d")
    module_logger.info("test object in new state: %s" % test_object.data)

    module_logger.info("adding column 'e' filled with the sum of columns a and b")
    add = lambda x, y: x + y
    test_object.add_column("e", origin = ["a", "b"], connection = add)
    module_logger.info("test object in new state: %s" % test_object.data)
