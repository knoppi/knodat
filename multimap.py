#!/usr/bin/env python
# -*- coding: utf-8 -*-

# developers notes:
# - http://packages.python.org/joblib/memory.html might speed up

import numpy as np
import scipy.signal as ssignal
import matplotlib.pyplot as plt
import logging

# I set the logging to both logging to std out and to a file with
# two different logging levels.
module_logger = logging.getLogger("multimap")
formatter = logging.Formatter(
    fmt = "%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s" )

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

module_logger.addHandler(ch)
module_logger.setLevel(logging.WARNING)

def set_debug_level(level):
    possible_levels = dict(debug = logging.DEBUG, info = logging.INFO,
            warning = logging.WARNING, error = logging.ERROR,
            fatal = logging.FATAL)
    ch.setLevel(possible_levels[level])
    module_logger.setLevel(possible_levels[level])


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

def gauss_kern_1d(size):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    module_logger.debug("called gauss_kern with size = %i" % size)
    size = int(size)
    
    x = np.arange(-size, size+1, 1)
    g = np.exp(-(x**2/float(size)))
    return g / g.sum()


class MultiMap:
    def __init__(self, _fileName=None, _cols=None):
        """
        Initialization of the Multimap
        Optional parameters are _fileName and _cols.
        """
        module_logger.info("MultiMap initialization:")
        module_logger.debug("-- filename: %s" % _fileName)
        module_logger.debug("-- columns: %s" % _cols)

        self.commentIndicators = ['#']
        self.columns = []
        self.dataType = []

        if not _cols == None:
            self.set_column_names(*_cols)
        if _fileName is not None:
            if _fileName[-4:] == '.npy':
                self.read_file_numpy_style(_fileName)
            else:
                self.read_file( _fileName )

        # As long as we do not define a column to be the indexing column
        # getitem_by_index should be chosen to retrieve a single row
        self.getitem_method = self.getitem_by_index

        # we need a numerical zero
        self.zero = 1e-10

        module_logger.debug("-- MultiMap initialized")

    def __setitem__(self, key, value):
        # possible error checking:
        # - key valid (type, index valid)
        # - value is dict and has correct dType
        module_logger.debug("__setitem__(%s, %s)" % (key, value))
        try:
            module_logger.debug("trying to find the key in the key list")
            i = self.keys.index(key)
        except AttributeError as e:
            module_logger.debug("Exception %s" % e)
            module_logger.debug("assuming no key list been given")
        except ValueError as e:
            module_logger.warning("key %s not in key list" % key)

        for j, j2 in enumerate(self.columns):
            module_logger.debug("... setting element '%s' (%i) in row %i "
                    "to %s" % (j2, j, i, value[j2]))
            self.data[i][j] = value[j2]

    def __iter__(self):
        return iter(self.data)

    # methods for modifying and reading out the structure of the multimap
    ###########################################################################
    section_structure = True
    def describe(self):
        """ Prints column names and data types, followed by the number of 
            entries. This method is based on the SQL method to decribe
            tables.
        """
        try:
            print "data loaded from: %s" % self.filename
        except AttributeError:
            # data not read from file but freshly created
            pass
        except:
            raise

        for item in self.dataType:
            print ("%20s: %s" % (item[0], item[1]))
        #print self.dataType.descr

        print "MultiMap contains %i entries" % self.length()
        print "MultiMap shape: %s" % (self.data.shape, )

    def length(self):
        return self.data.shape[0]

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

        self.set_data_type(new_data_type)

    def set_data_type(self, new_dataType):
        """sets the data type of the internal data storage and
        creates a new empty np-array"""
        module_logger.debug("... data type: %s" % new_dataType)
        self.dataType = new_dataType
        self.data = np.zeros( 
                (0, len(new_dataType)), 
                dtype = new_dataType)

        self.columns = [x[0] for x in new_dataType]

    def change_data_type_for_column(self, column, new_data_type):
        """ modifies the datatype of a single column """
        module_logger.debug("changing data type for column %s to %s" % (column, new_data_type))
        icolumn = self.columns.index(column)
        new_dType = self.dataType[:]
        new_dType[icolumn] = (column, new_data_type)

        self.set_data_type(new_dType)

    def select_indexing_column(self, name):
        """choose one column containing the indices which then can be called by
           getitem"""
        self.named_indices = True
        try:
            self.keys = self.get_column(name).tolist()
            self.index_column = name
        except None:
            module_logger.fatal("chosen invalid key-column")

    def set_x_column(self, x_name):
        """
        This method declares a certain column of the MultiMap to act
        as an index column, so __getitem__ takes a value of that column
        as the index.
        """
        self.x_column_name = x_name
        try:
            self.x_column = self.get_column(x_name).tolist()
        except Exception:
            raise
        else:
            self.getitem_method = self.getitem_by_x

    # methods for read access to the data in the multimap
    ###########################################################################
    section_read_access_simple = True

    def __getitem__(self, i):
        """ 
        Retrieve a single row of the MultiMap as a dict, i can either be an 
        integer or a variable type key
        """
        return self.getitem_method(i)

    def getitem_by_index(self, i):
        """
        This method is intended for internal use only, its purpose is to return
        row i of the multimap as a dictionary.
        """
        tmp = {}

        try:
            module_logger.debug("trying to find the key in the key list")
            i = self.keys.index(i)
            module_logger.debug("the key is now %s" % i)
        except AttributeError as e:
            module_logger.debug("Exception %s" % e)
            module_logger.debug("assuming no key list been given")
        except ValueError as e:
            module_logger.warning("key %s not in key list" % i)
        
        # with the key finally found, we cann create the result
        try:
            for j in range(len(self.columns)):
                tmp[self.columns[j]] = self.data[i][j]
        except IndexError as e:
            module_logger.warning("Calling no-existent key!")
            module_logger.warning(e)
        return tmp

    def getitem_by_x(self, i):
        """
        This method is for internal use only. It returns a single row of the
        multimap as a dictionary defined by col_x = i
        """
        try:
            j = self.x_column.index(i)
            return self.getitem_by_index(j)
        except Exception:
            raise

    def getitem(self, i):
        """ retrieve a single row of the MultiMap as a dict """
        self.__getitem__(i)

    def get_subset(self, restrictions = {}, deletion = False):
        """ returns a subset of data with hard restrictions
        """
        module_logger.debug('get_subset: restrictions: %s' % (restrictions,))
        restriction_lhs = []
        restriction_rhs = []
        for key in restrictions:
            restriction_lhs.append( key )
            restriction_rhs.append( restrictions[key] )

        _lambda = lambda n,x: np.abs(x - restriction_rhs[n]) < self.zero
        _lambda_general = lambda n,x: (x == restriction_rhs[n])

        result = self.data[:]

        indices = np.array(range(self.data.shape[0]))

        i = 0
        for rest_name in restriction_lhs:
            try:
                result = result[ np.where( _lambda( i, result[:][rest_name] ) ) ]
                indices = indices[ np.where( _lambda( i, result[:][rest_name] ) ) ]
            except TypeError:
                # most probably the variable is no float
                result = result[ np.where( _lambda_general( i, result[:][rest_name] ) ) ]
                indices = indices[ np.where( _lambda_general( i, result[:][rest_name] ) ) ]
            except:
                raise
            i += 1

        if deletion: 
            module_logger.debug("deletion!")
            self.data = np.delete(self.data, indices, 0)
        return result[:]

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
        result = self.get_subset(restrictions)

        return result[desire]

    def get_column(self, desire):
        """ returns column desire without applying any restriction
        """
        return self.data[:][desire]

    # methods for write access to the data in the multimap
    ###########################################################################
    section_write_access = True

    def add_column(self, new_name, dataType = "|f8", 
                   origin = [], connection = None):
        '''adds a new column called "name" with is either just
           zero or is constructed out of the columns listed in
           origin, connected by some function "connection"'''
        new_datatype = self.dataType
        new_datatype.append((new_name, dataType))

        if connection == None:
            newCol = np.zeros((self.data.size,))
        else:
            arguments = []
            for colname in origin:
                arguments.append(self.data[:][colname])
            newCol = connection(*arguments)
            newCol = newCol

        module_logger.info("new column: %s" % newCol)

        tmp = np.empty(self.data.shape, dtype = new_datatype)
        for name in self.data.dtype.names:
            tmp[name] = self.data[name]

        tmp[new_name] = newCol

        self.data = tmp

        self.dataType = new_datatype
        self.columns.append(new_name)

    def append_row(self, row):
        """row is some iterable object which somehow has to fit to dtype"""
        module_logger.debug("append_row(%s)" % (row,))
        new_row = np.array(tuple(row), dtype=self.data.dtype, ndmin = 2)
        self.data = np.append(self.data, new_row)

        #if self.named_indices:
            #self.select_indexing_column(self.index_column)

        #new_row = self.__getitem__(self.keys[-1])
        #module_logger.debug("new row as ndarray is %s" % (new_row,))
        #module_logger.debug("new row as ndarray is %s" % (self.data,))

        # TODO Do this with functional programming
        try:
            self.set_x_column(self.x_column_name)
        except AttributeError, e:
            module_logger.info(e)
        except:
            raise

    def append_data(self, other):
        """ concatenate MultiMap with a second one """
        try:
            self.data = np.append(self.data, other.data)
        except:
            raise

    def add_to_column(self, col, values):
        """ modifies column by adding values """
        try:
            self.data[:,col] = self.data.column[:,col] + values.transpose()
        except:
            raise
    
    def multiply_column(self, col, factor):
        """ modifies column by multiplying it with factor """
        try:
            self.data[:,col] = self.data.column[:,col] * factor
        except:
            raise

    # section for interactions with the filesystem
    ###########################################################################
    section_filesystem_interactions = True
    
    def read_file(self, filename, **options):
        """reads data from ascii file"""
        self.filename = filename
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
                module_logger.debug('taking column names from first line')
                column_names = first_row[3:(len(first_row) - 1)].split(" ")
                self.set_column_names(*column_names)
            else:
                module_logger.debug("column names not given on first line")
                while first_row[0:1] in self.commentIndicators:
                    first_row = file_id.readline()
                number_of_cols = len(first_row.strip().split(" "))
                module_logger.debug("... number of columns: %i" % number_of_cols)
                column_names = [str(x) for x in range(1, number_of_cols + 1)]
                self.set_column_names(*column_names)

        self.data = np.loadtxt(filename, dtype = self.dataType)

        module_logger.debug("finished reading file")

    def read_file_numpy_style(self, filename):
        '''loads content into the MultiMap which was formerly saved as a
        numpy style .npy file'''
        self.filename = filename
        self.data = np.load(filename)
        self.dataType = self.data.dtype.descr
        self.columns = []
        for item in self.dataType:
            self.columns.append(item[0])

    def set_zero(self, value):
        self.zero = value

    def sort( self, column ):
        """ sorts the MultiMap along column
        """
        self.data = np.sort( self.data, order = column )

    def write_file(self, filename, **options):
        """writes the data to a file called 'filename' in ASCII format """
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

    def write_file_for_gnuplot_splot(self, _x, _y, _filename):
        """ when doing surface plots with gnuplot there have to be empty lines 
            in the data file after each isoline which can be achieved by this
            method
        """
        self.sort(np.str(_x))
        self.sort(np.str(_y))
        
        outfile = open(_filename, 'w')

        # head line containing the description of the file, i.e. the
        # dolumn names
        column_names = ' '.join( ["%s" % k for k in self.columns])
        line = "".join((self.commentIndicators[0], self.commentIndicators[0], 
                        " ", column_names, "\n"))
        outfile.write( line )

        # write data line by line and check if x has been increased
        current_x = self.data[0][_x]
        for iline in range( self.data.shape[0] ):
            current_line = self.data[iline]

            if not current_line[_x] == current_x:
                outfile.write("\n")
                current_x = current_line[_x]

            line = ""
            for x in current_line:
                line += np.str_( x ) + " "
            line = line+"\n"
            outfile.write( line )

        module_logger.debug("finished writing file in gnuplot style")

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

    def get_maximum_value(self, column, absolute = False):
        '''finds the element in column with the least (absolute) value'''
        if absolute is True:
            return np.amax(np.abs(self.data[:][column]))
        else:
            return np.amax(self.data[:][column])

    def get_x_of_minimum_value(self, x_column, y_column, absolute = False):
        '''returns value of x_column where (absolute) of y_column has its
           minimum'''
        y_value = self.get_minimum_value(y_column, absolute)
        restriction = {y_column: y_value}
        x_values = self.get_column_hard_restriction(x_column, **restriction)

        return (x_values, y_value)

    def get_x_of_maximum_value(self, x_column, y_column, absolute = False):
        '''returns value of x_column where (absolute) of y_column has its
           minimum'''
        y_value = self.get_maximum_value(y_column, absolute)
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
        result = self.get_subset(restrictions)

        return result[desire]

    def get_column(self, desire):
        """ returns column desire without applying any restriction
        """
        return self.data[:][desire]

    # methods giving non-trivial read access to data, usually involving some 
    # maths or statistical operations
    ###########################################################################
    section_read_access_complex = True
    def mean(self, column):
        tmp = self.get_column(column)
        return self.mean

    def get_possible_values(self, colName, **restrictions):
        """ similar as get_column_general, but duplicates are deleted
        """
        temp = np.unique(
                self.get_column_hard_restriction(colName, **restrictions))
        return temp

    def get_histogram(self, col, restrictions = {}, **kwargs):
        """get a histogram for the values in column col
           """
        kwargs = dict(kwargs)
        x = self.get_column(col)

        if 'bins' not in kwargs.keys():
            kwargs['bins'] = 20

        hist, bin_edges = np.histogram(x, **kwargs)

        x = np.zeros(hist.shape)
        x = 0.5 * (bin_edges[0:-1] + bin_edges[1:])

        return (x, hist)

    def retrieve_2d_plot_data(self, _colx, _coly, errx = None, erry = None, N = 1,
                              restrictions = {}):
        """ TODO has to be rewritten """
        _colx = str(_colx)
        _coly = str(_coly)
        module_logger.debug( 
                "retrieving plot data %s vs %s" % ( _colx, _coly ) )
        module_logger.debug( "errorbars are %s, %s" % ( errx, erry ) )
        self.sort(_colx)
        xvals = self.get_column_hard_restriction( _colx, **restrictions )
        yvals = self.get_column_hard_restriction( _coly, **restrictions )
        xerrs = ( 0 if errx == None else 
                self.get_column_hard_restriction( errx, **restrictions ) )
        yerrs = ( 0 if erry == None else
                self.get_column_hard_restriction( erry, **restrictions ) )
        
        if N > 1:
            g = gauss_kern_1d(N)
            module_logger.debug(g)
            #xvals = ssignal.convolve(xvals, g, 'same')
            yvals = ssignal.convolve(yvals, g, 'same')

 
        if errx == None and erry == None:
            return ( xvals, yvals )
        elif errx == None and not erry == None:
            return ( xvals, yvals, yerrs )
        elif not errx == None and erry == None:
            return ( xvals, yvals, xerrs )
        else:
            return ( xvals, yvals, xerrs, yerrs )

    def retrieve_3d_plot_data(self, _x, _y, _z, N = 2, data_is_complete = True, 
            *args, **kwargs):
        """ returns the needed matrices for creating a matplotlib-like 3d-plot
        """
        try:
            restrictions = kwargs["restrictions"]
        except KeyError:
            restrictions = {}
        except:
            raise

        try:
            grid = kwargs['grid']
        except KeyError:
            grid = 'square'
        except:
            raise

        deletion = False;
        if "deletion" in kwargs.keys():
            deletion = kwargs['deletion']

        # retrieve the subset according to restrictions
        data = self.get_subset(restrictions = restrictions, deletion = deletion)
        data = np.sort(data, order=[_y, _x])

        x = np.unique(data[:][_x])
        y = np.unique(data[::-1][_y])


        # for graphene we have to reduce the x-dimension by a factor of 2
        X = np.zeros((1,1))
        Y = np.zeros((1,1))
        if grid == 'graphenegrid':
            module_logger.debug('assuming graphene grid')
            T,Y = np.meshgrid(x[::2], y)
            X = np.zeros(T.shape)
            module_logger.debug('grid dimensions: %s' % (X.shape, ))

            # let the x-shift depend on the y-coordinates:
            # if the change in y is 1 / sqrt(3), there is
            # no x-shift, otherwise, it is 0.5
            x1 = 0.0
            if y[1] - y[0] == 1 / np.sqrt(3):
                x2 = 0.0
                x3 = 0.5
                x4 = 0.5
            else:
                x2 = 0.5
                x3 = 0.5
                x4 = 0.0

            X[0::4] = T[0::4] + x1
            X[1::4] = T[1::4] + x2
            X[2::4] = T[2::4] + x3
            X[3::4] = T[3::4] + x4
        else:
            module_logger.debug('assuming square grid')
            X,Y = np.meshgrid(x, y)

        extent = ( x.min(), x.max(), y.min(), y.max() )

        Z = np.zeros(X.shape)
        
        if data_is_complete == False:
            for row in data:
                xi = 0
                xi = int(np.where(x == row[_x])[0][0] / 2) 
                yi, = np.where(y == row[_y])[0]
                Z[yi,xi] = row[_z]
        else:
            # NOTE missing values rot this reshaping, an additional method for that
            # case is the one commented out below, but this is by far less fast
            difference = Y.shape[0]*Y.shape[1] - len(data[:][_z])
            module_logger.debug('datapoints %i' % len(data[:][_z]))
            module_logger.debug('grid size %i' % (Y.shape[0]*Y.shape[1]))
            module_logger.debug('difference to optimum entry number %i' % difference)
            if difference > 0:
                data = np.sort(data, order=[_x, _y])
                data = np.append(data[:], data[-difference:])
                data = np.sort(data, order=[_y, _x])
            
            module_logger.debug('datapoints %i' % len(data[:][_z]))
            Z = data[:][_z].reshape(Y.shape)
                
        xoffset = 0
        yoffset = 0
        if N > 1:
            g = gauss_kern(N)
            xoffset = (Y.shape[1] % N) / 2
            yoffset = (Y.shape[0] % N) / 2
            Z = ssignal.convolve(Z, g, 'same')

            #X = X[yoffset::N,xoffset::N]
            #Y = Y[yoffset::N,xoffset::N]
            Z = Z[yoffset::N,xoffset::N]
        
        return (X, Y, Z, extent )

    def retrieve_quiver_plot_data( self, _x, _y, _u, _v, N = 5, **kwargs ):
        """ for detailed information what a quiver plot is, please read 
            matplotlib documentation; this method basically returns the data
            in a format as the matplotlib.pyplot.quiver method understand
        """
        module_logger.info("retrieve_quiver_plot_data, x-col: %s" % (_x))
        module_logger.info("retrieve_quiver_plot_data, y-col: %s" % (_y))
        module_logger.info("retrieve_quiver_plot_data, u-col: %s" % (_u))
        module_logger.info("retrieve_quiver_plot_data, v-col: %s" % (_v))

        restrictions = {}
        if 'restrictions' in kwargs.keys():
            module_logger.debug("retrieve_quiver_plot_data restrictions: %s" % (kwargs["restrictions"]))
            restrictions = kwargs["restrictions"]

        grid = 'square'
        if 'grid' in kwargs.keys():
            grid = kwargs['grid']

        deletion = False;
        if "deletion" in kwargs.keys():
            deletion = kwargs['deletion']

        data = self.get_subset(restrictions = restrictions, deletion = deletion)
        data = np.sort(data, order=[_y, _x])

        x = np.unique(data[::][_x])
        y = np.unique(data[::-1][_y])
        u = (data[::][_u])

        # for graphene we have to reduce the x-dimension by a factor of 2
        X = np.zeros((1,1))
        Y = np.zeros((1,1))
        if grid == 'graphenegrid':
            module_logger.debug('assuming graphene grid')
            x0 = x[0]
            x1 = x[0] - x0
            x2 = x[1] - x0
            x3 = x[2] - x0
            x4 = x[3] - x0
            T,Y = np.meshgrid(x[::2], y)
            X = np.zeros(T.shape)
            X[0::4] = T[0::4] + x1
            X[1::4] = T[1::4] + x2
            X[2::4] = T[2::4] + x3
            X[3::4] = T[3::4] + x4
        else:
            module_logger.debug('assuming square grid')
            X,Y = np.meshgrid(x, y)

        extent = ( x.min(), x.max(), y.min(), y.max() )

        #U = np.zeros(X.shape)
        #V = np.zeros(X.shape)
        
        # NOTE missing values rot this reshaping, an additional method for that
        # case is the one commented out below, but this is by far less fast
        difference = X.shape[0]*X.shape[1] - len(data[:][_u])
        module_logger.debug('difference to optimum entry number %i' % difference)
        data = np.sort(data, order=[_x, _y])
        data = np.append(data[:], data[-difference:])
        data = np.sort(data, order=[_y, _x])
        
        U = data[:][_u].reshape(Y.shape)
        V = data[:][_v].reshape(Y.shape)
        #if len(data[:][_u]) == X.shape[0]*X.shape[1]:
            #module_logger.debug("retrieve_quiver_plot_data: using fast reshaping method for the data")
            #U = data[:][_u].reshape(Y.shape)
            #V = data[:][_v].reshape(Y.shape)
        #else:
            #module_logger.debug("retrieve_quiver_plot_data: using slow element-wise ordering")
            #for row in data:
                #xi = int(np.where(x == row[_x])[0][0] / 2)
                #yi, = np.where(y == row[_y])[0]

                #X[yi,xi] = row[_x]
                #Y[yi,xi] = row[_y]
                #U[yi,xi] = row[_u]
                #V[yi,xi] = row[_v]

                #xi += 1
                
        g = gauss_kern(N, N)
        xoffset = (Y.shape[1] % N) / 2
        yoffset = (Y.shape[0] % N) / 2
        #module_logger.debug(g)
        U = ssignal.convolve(U, g, 'same')
        V = ssignal.convolve(V, g, 'same')

        X = X[yoffset::N,xoffset::N]
        Y = Y[yoffset::N,xoffset::N]
        U = U[yoffset::N,xoffset::N]
        V = V[yoffset::N,xoffset::N]

        return (X, Y, U, V, extent)

    # Old functions kept due to backwards comapbility; will give a warning
    # about deprecation
    ##########################################################################
    section_old = True
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

    def plot_2d_data(self, _colx, _coly, errx = None, erry = None, 
                     restrictions = {}, label = "", fmt = "", **options):
        print "function does not exist anymore"

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

    def retrieveQuiverPlotData( self, _x, _y, _u, _v, **kwargs ):
        """ returns the needed matrices for creating a matplotlib-like 3d-plot 
        """
        return self.retrieve_quiver_plot_data(_x, _y, _u, _v, **kwargs)

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
    set_debug_level("debug")
    module_logger.info("create multimap with column a, b and c")

    test_object = MultiMap(_cols = ["a", "b", "c"])

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
    test_object.add_column("e", origin = ["a", "b"], connection = np.add)
    module_logger.info("test object in new state: %s" % test_object.data)

    module_logger.info("row 1 contains: %s" % test_object[0])

    module_logger.info("choose column 'b' for indexing")
    test_object.select_indexing_column("b")
    module_logger.info("row with key %s contains: %s" % (5.0, test_object[5.0]))
    module_logger.info("calling non-existent key %s yields: %s" % (3.0, test_object[3.0]))
    module_logger.info("calling non-existent key %s yields: %s" % (1.0, test_object[1.0]))

