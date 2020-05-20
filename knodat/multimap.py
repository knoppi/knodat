#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the core element of the project. It provides the class
:py:class:`MultiMap` which is where the evaluation tools
evtools are based upon
"""

import sys
import itertools

import numpy as np
import scipy.signal as ssignal
import logging

# I set the logging to both logging to std out and to a file with
# two different logging levels.
module_logger = logging.getLogger("multimap")
formatter = logging.Formatter(
    fmt="%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.WARNING)

module_logger.addHandler(ch)
module_logger.setLevel(logging.WARNING)


def set_debug_level(level):
    possible_levels = dict(
        debug=logging.DEBUG, info=logging.INFO, warning=logging.WARNING,
        error=logging.ERROR, fatal=logging.FATAL)
    ch.setLevel(possible_levels[level])
    module_logger.setLevel(possible_levels[level])


def gauss_kern(size, sizey=None):
    """
    Return a normalized 2D gauss kernel array for convolutions.

    :params int size: Number of points the Gaussian should extend over.
    :params int sizey: Similar as :py:obj:`size`. Only set a finite value here
        if the Gaussian shall not be isotropic.
    """

    module_logger.debug("called gauss_kern with size = %i" % size)
    size = int(size)

    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)

    if size == 1 and sizey == 1:
        return [[1]]

    x, y = np.mgrid[-size:size + 1, -sizey:sizey + 1]
    g = np.exp(- (x ** 2 / float(size) + y ** 2 / float(sizey)))
    print(g)
    return g / g.sum()


def gauss_kern_1d(size):
    """
    Returns a normalized 1D gauss kernel array for convolutions.

    :params int size: Number of points the Gaussian should extend over.
    """
    module_logger.debug("called gauss_kern with size = %i" % size)
    size = int(size)

    x = np.arange(-size, size + 1, 1)
    g = np.exp(-(x ** 2 / float(size)))
    return g / g.sum()


def create_ordering_column(*cols):
    result = np.zeros(cols[0].shape[0], "|S200")
    for column in cols:
        result = np.core.defchararray.add(
            result, column.astype("|S20"))
    return result


def reduce_subset(current_subset, static, averaging_cols, statistics, method):
    import numpy as np

    new_row = []
    ensemble_size = current_subset.shape[0]
    for static_col in static:
        new_row.append(current_subset[static_col][0])
    for averaged_col in averaging_cols:
        if statistics is True:
            average = method(current_subset[averaged_col])
            values = current_subset[averaged_col]
            std = values.std()
            variance = np.sqrt(np.mean((values - average) ** 2))
            new_row.append(average)
            new_row.append(std / np.sqrt(ensemble_size))
            new_row.append(variance)
        else:
            average = method(current_subset[averaged_col])
            new_row.append(average)
    if statistics is True:
        new_row.append(ensemble_size)

    return new_row


class MultiMap:
    """
    Class for organizing large datasets.

    :param str filename: Name of a file to load the data from.
    :param list columns: List of column names

    :py:class:`MultiMap` acts as a kind of small in-memory database with special
    methods for fast operations on columns required for the operation on large
    amounts of data from, e.g., numerical simulations.

    It can be used to load data from an existing file or to collect data.
    When loading data from an existing file this can be done with a plain text
    file where data is organised in whitespace separated columns or in the
    binary format used by the numpy.save function for ndarrays (assuming a
    filename suffix .npy).

    Other ways of organizing data are possible as long as numpy.loadtxt can
    understand it.

    If a list of column names and no filename ending with ".npy" is given,
    columns of the file can be adressed by these names. .npy-files can store
    column names within themselves. If no column names are given they are
    labeled by integer numbers.

    By default the column content is assumed to be floats. In other cases first
    define the data type before loading the data since otherwise inconsistencies
    might appear. See :py:meth:`set_column_names`, :py:meth:`set_data_type`,
    :py:meth:`change_data_type_for_column`, :py:meth:`read_file` and
    :py:meth:`read_file_numpy_style` for further instructions.
    """

    def __init__(self, filename=None, columns=None, **kwargs):
        # renaming function parameters requires caring for backwards
        # compatibility:
        if "_fileName" in kwargs.keys():
            filename = kwargs["_fileName"]
        if "_cols" in kwargs.keys():
            columns = kwargs["_cols"]

        module_logger.info("MultiMap initialization:")
        module_logger.debug("-- filename: %s" % filename)
        module_logger.debug("-- columns: %s" % columns)

        self.commentIndicators = ['#']
        self.columns = []
        self.dtype = []

        if columns is not None:
            self.set_column_names(*columns)
        if filename is not None:
            if filename[-4:] == '.npy':
                self.read_file_numpy_style(filename)
            else:
                self.read_file(filename)

        # we need a numerical zero
        self.zero = 1e-10

        # we need a default running average size
        self.running_average_N = 1

        # we need a default grid
        self.chosen_grid = "squaregrid"

        # we need to set the completeness of our data
        self.complete = True

        module_logger.debug("-- MultiMap initialized")

    def __getitem__(self, index):
        """
        Retrieve a single row of the MultiMap as a dictionary object
        """
        tmp = {}

        try:
            module_logger.debug("trying to find the key in the key list")
            index = self.keys.index(index)
            module_logger.debug("the key is now %s" % index)
        except AttributeError as e:
            module_logger.debug("Exception %s" % e)
            module_logger.debug("assuming no key list been given")
        except ValueError as e:
            module_logger.warning("key %s not in key list" % index)

        # with the key finally found, we cann create the result
        try:
            for j in range(len(self.columns)):
                tmp[self.columns[j]] = self.data[index][j]
        except IndexError as e:
            module_logger.warning("Calling no-existent key!")
            module_logger.warning(e)
        return tmp

    def getitem(self, index):
        """
        Return row ``index`` of the MultiMap as ``dict``-object.
        """
        self.__getitem__(index)

    def __setitem__(self, row, value):
        # possible error checking:
        # - key valid (type, index valid)
        # - value is dict and has correct dType
        module_logger.debug("__setitem__(%s, %s)" % (row, value))

        for j, j2 in enumerate(self.columns):
            self.data[row][j] = value[j2]

    def __iter__(self):
        return iter(self.data)

    # methods for modifying and reading out the structure of the multimap
    #
    section_structure = True

    def __eq__(self, other):
        """
        Compare to MultiMaps if they contain the same data and have the same
        datatype.
        """
        return np.all((self.data == other.data))
    """
    Filesystem access
    """
    def read_file(self, filename, **options):
        """
        Read data from ascii file.

        :param str filename: Name of the file data is loaded from.

        This method reads data from an ascii file using the numpy function
        loadtxt which is in particular suited for reading csv-datafile.
        Further parameters are required for the control of the loadtxt function.
        """
        self.filename = filename
        for (key, val) in dict(options):
            if key == "delimiter":
                self.delimiter = val
            if key == "fieldnames":
                self.set_column_names(val)
            # if key == "skipinitialspaces":
                #skipInitialSpaces = val

        # now three cases can be thought of:
        # 1) we already know the column names
        # 2) we do not know the but in the first line of the file
        # there is a hint line (beginning with ##)
        # 3) neither of the cases
        # In case 1 len(self.columns) > 1 and we can directly go to reading
        # the file. In case 2 we read the magic line to get the column names.
        # In the last case we look for the first line not beginning with
        # a comment indicator to count the number of columns and name them
        # just with numbers
        if len(self.columns) == 0:
            file_id = open(filename, 'rb')
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
                module_logger.debug("... number of columns: %i" %
                                    number_of_cols)
                column_names = [str(x) for x in range(1, number_of_cols + 1)]
                self.set_column_names(*column_names)

        try:
            self.data = np.loadtxt(filename, dtype=self.dtype, delimiter=self.delimiter)
        except AttributeError:
            self.data = np.loadtxt(filename, dtype=self.dtype)

        module_logger.debug("finished reading file")

    def read_file_numpy_style(self, filename):
        '''
        Load content of a compressed numpy ND-array

        :param str filename: file to load
        '''
        self.filename = filename
        self.data = np.load(filename)
        self.dtype = self.data.dtype.descr
        self.columns = []
        for item in self.dtype:
            self.columns.append(item[0])

    def write_file(self, filename, **options):
        """writes the data to a file called 'filename' in ASCII format """
        outfile = open(filename, 'w')

        # head line containing the description of the file, i.e. the
        # dolumn names
        column_names = ' '.join(["%s" % k for k in self.columns])
        line = "".join((self.commentIndicators[0], self.commentIndicators[0],
                        " ", column_names, "\n"))
        outfile.write(line)

        # write data line by line
        for iline in range(self.data.shape[0]):
            #line = " ".join(np.str_(self.data[iline]))
            line = ""
            for x in self.data[iline]:
                line += np.str_(x) + " "
                #line += ( "%.10f " % x )
            line = line + "\n"
            outfile.write(line)

        module_logger.debug("finished writing file")

    def write_file_numpy_style(self, filename):
        '''writes the content of the MultiMap in numpy-style .npy file'''
        np.save(filename, self.data)

    def write_file_for_gnuplot_splot(self, column_x, column_y, filename):
        """
        Write data to ``filename`` in a way suitable for gnuplot's splot.

        When doing surface plots with gnuplot there have to be empty lines in
        the data file after each isoline which is performed with this method.
        """
        outfile = open(filename, 'w')

        xcol = np.str(column_x)
        ycol = np.str(column_y)

        self.sort([ycol, xcol])

        lastx = self.data[0][xcol]
        for row in self.data:
            if row[xcol] < lastx:
                outfile.write("\n")

            outfile.write(" ".join([np.str(x) for x in row]))
            outfile.write("\n")
            lastx = row[xcol]

    """
    Descriptive methods
    """
    def describe(self, few_values=10):
        """
        Describe data contents similar to MySQL describe of tables.

        :param int few_values: Data description states the range of entries
            unless less than :py:obj:`few_values` are present in the
            :py:class:`MultiMap`.

        This method is intended for use in the command line since the output is
        written directly to std::out.
        """
        try:
            print("data loaded from: %s" % self.filename)
        except AttributeError:
            # data not read from file but freshly created
            pass
        except:
            raise

        for item in self.dtype:
            print("%20s: %s" % (item[0], item[1]))

        print("MultiMap contains %i entries" % self.length())
        print("MultiMap shape: %s" % (self.data.shape, ))

        if self.length() > 1:
            for column in self.columns:
                values = self.get_possible_values(column)
                formatting = "%27s %13s: %s"
                if len(values) == 1:
                    print(formatting % ("fixed parameter", column, values[0]))
                elif len(values) < 5:
                    print(formatting % ("parameter with few values",
                                        column, values))
                else:
                    limits = ("%s - %s" % (min(values), max(values)))
                    description = "parameter with %i values" % len(values)
                    print(formatting % (description, column, limits))

    def length(self):
        """
        Return the number of data entries.
        """
        try:
            return self.data.shape[0]
        except IndexError:
            # let's just assume that we only have one entry
            return 1

    """
    Metadata manipulation
    """
    def set_column_names(self, *column_names, **options):
        """
        Set the names of the columns and - if needed - the corresponding dtype,
        this will also clear the data.

        Positional arguments will be the column names, as a default data type
        numpy.float64 is assumed if the keyword arguemnt "dType" is not given.
        If dType is specified all columns are set to this data type.

        For more complex data definitions use :py:meth:`set_data_type` or
        :py:meth:`change_data_type_for_column`.
        """
        # easy thing is setting the array of the column names
        self.columns = [x for x in column_names]

        # Now define the data type.
        new_data_type = []

        if "dType" in options.keys():
            standardType = options["dType"]
        else:
            standardType = "<f8"

        for i in column_names:
            new_data_type.append((i, standardType))

        self.set_data_type(new_data_type)

    def set_data_type(self, new_data_type):
        """
        Set the data type of the internal data storage to ``new_data_type``
        and creates a new empty numpy array.

        This method is useful in particular is several different data types
        shall be used (like mixtures of strings and numerical values).
        For more details see the :class:`~numpy:numpy.dtype` documentation.
        """
        module_logger.debug("... data type: %s" % new_data_type)
        self.dtype = new_data_type
        self.data = np.zeros(
            (0, len(new_data_type)),
            dtype=new_data_type)

        self.columns = [x[0] for x in new_data_type]

    def change_data_type_for_column(self, column, new_data_type):
        """
        Modify the datatype of a single column.
        """
        module_logger.debug("changing data type for column %s to %s" %
                            (column, new_data_type))
        icolumn = self.columns.index(column)
        new_dType = self.dtype[:]
        new_dType[icolumn] = (column, new_data_type)

        self.set_data_type(new_dType)

    def set_zero(self, value):
        """
        Set numerical zero used in comparison functions.

        This method can be used to control accuracy. We can also use for
        restricting selected data to a certain interval.
        """
        self.zero = value

    def set_N(self, value):
        """
        Set ammount of supporting points included in the Gaussian smooting of
        retrieved data.
        """
        self.running_average_N = value

    def set_grid(self, value):
        """
        Set the underlying grid for 3D data retrievement. Currently square and
        graphenegrid, i.e. a hexagonal grid, are supported.
        """
        self.chosen_grid = value

    def set_complete(self, value):
        """
        State if all points of the defined grid are occupied.

        According to this value the algorithms retrieving 3D data decide if
        datapoints have to be attributed to their coordinated explicitely or
        if a faster reshaping of the flattened data is possible.
        """
        self.complete = value

    def sort(self, column):
        """
        Sort the MultiMap along :py:obj:`column`.
        """
        self.data = np.sort(self.data, order=column)

    """
    Data retrieval
    """
    def get(self, *columns, **restrictions):
        """
        Obtain complete subsets or 1- to 3-column slices of the ``MultiMap``
        following restrictions defined in ``keywargs``.

        This methods refers to
            * :meth:`get_subset`, if ``columns`` has length 0,
            * :meth:`get_possible_values`, if ``columns`` has length 1,
            * :meth:`retrieve_2d_plot_data` if ``columns`` has length 2,
            * :meth:`retrieve_3d_plot_data` if ``columns`` has length 3.

        The options given in ``keywargs`` are used only for restrictions.
        In order to address the further parameters of the relevant methods
        use :meth:`set_N`, :meth:`set_grid` and :meth:`set_complete`.

        For details see the documentation of the respective methods.
        """
        if len(columns) == 0:
            R = dict(restrictions)
            return self.get_subset(R)
        if len(columns) == 1:
            return self.get_possible_values(
                *columns, **restrictions)
        if len(columns) == 2:
            return self.retrieve_2d_plot_data(
                *columns, restrictions=restrictions,
                N=self.running_average_N)
        if len(columns) == 3:
            return self.retrieve_3d_plot_data(
                *columns, restrictions=restrictions,
                N=self.running_average_N, grid=self.chosen_grid,
                data_is_complete=self.complete)
        else:
            pass

    def get_column(self, column_name):
        """
        Return column ``column_name`` as :class:`numpy.ndarray`-object without
        applying any restrictions.
        """
        return self.data[:][column_name]

    # methods for write access to the data in the multimap
    #
    section_write_access = True

    def get_subset(self, restrictions={}, deletion=False):
        """
        Return a subset of the MultiMap.

        :param dict restriction: List of column names and the expected value.
        :param bool deletion: True => delete the subset obeying
            ``restrictions``.
        :return: Object with the same datatype as the MultiMap.
        :rtype: :class:`numpy.ndarray`
        """
        # We define two functions for comparison of the restrictions
        # with our data.
        # hard_restriction requires an exact agreement and is suitable
        # for exact data as strings
        # soft_restriction checks for agreement within a given tolerance
        # self.zero and is used by default; hard restriction is merely a
        # fallback mechanism
        R = restrictions.items()

        def soft_restriction(n, x):
            return np.abs(x - R[n][1]) < self.zero

        def hard_restriction(n, x):
            return (x == R[n][1])

        # data to return
        result = self.data[:]
        indices = np.array(range(self.data.shape[0]))

        # reduce the result by applying restrictions
        for idx in range(len(R)):
            try:
                subset = np.where(soft_restriction(idx, result[:][R[idx][0]]))
                result = result[subset]
                indices = indices[subset]
            except TypeError:
                # most probably the variable is no float
                subset = np.where(hard_restriction(idx, result[:][R[idx][0]]))
                result = result[subset]
                indices = indices[subset]
            except:
                raise

        if deletion is True:
            self.data = np.delete(self.data, indices)

        return result

    def get_possible_values(self, column_name, **restrictions):
        """
        Return a unique list of values in column ``column_name``.

        The list of ``restrictions`` can be used to limit the resulting values.
        This method internally calls :meth:`get_column_hard_restriction` and
        removes duplicates from the result.
        """
        temp = np.unique(
            self.get_column_hard_restriction(column_name, **restrictions))
        return temp

    def retrieve_2d_plot_data(
            self, column_x, column_y, column_x_error=None, column_y_error=None,
            N=1, restrictions={}, trim=True
            ):
        """
        Return data ready for 2D plotting.

        :param str column_x: name of x-column
        :param str column_y: name of y-column
        :param str column_x_error: name of column containing x error
        :param str column_y_error: name of column containing y error
        :param int N: number of points included in running average
        :param dict restrictions: limit to a certain subset
        :param bool trim: If running average is performed (N>1) this parameter
            controls if the data is cut at the edges.
        :return: tuple with x-, y-data and possibly errorbar sizes.

        This method returns a tuple of array containing the data for the plot.
        Size of the tuple depends on given arguments. If errorbars are desired
        they are included in the returned array.
        """
        column_x = str(column_x)
        column_y = str(column_y)

        module_logger.debug(
            "retrieving plot data %s vs %s" % (column_x, column_y))
        module_logger.debug(
            "errorbars are %s, %s" % (column_x_error, column_y_error))

        # fetch the data
        self.sort(column_x)
        xvals = self.get_column_hard_restriction(column_x, **restrictions)
        yvals = self.get_column_hard_restriction(column_y, **restrictions)
        xerrs = (
            0 if column_x_error is None else
            self.get_column_hard_restriction(column_x_error, **restrictions)
        )
        yerrs = (
            0 if column_y_error is None else
            self.get_column_hard_restriction(column_y_error, **restrictions)
        )

        # perform running average
        if N > 1:
            g = gauss_kern_1d(N)
            module_logger.debug(g)
            yvals = ssignal.convolve(yvals, g, 'same')

            if trim is True:
                xvals = xvals[N:-N]
                yvals = yvals[N:-N]
                try:
                    xerrs = xerrs[N:-N]
                except TypeError:
                    pass
                try:
                    yerrs = yerrs[N:-N]
                except TypeError:
                    pass

        # setup returned tuple
        if column_x_error is None and column_y_error is None:
            return (xvals, yvals)
        elif column_x_error is None and column_y_error is not None:
            return (xvals, yvals, yerrs)
        elif column_x_error is not None and column_y_error is None:
            return (xvals, yvals, xerrs)
        else:
            return (xvals, yvals, xerrs, yerrs)

    def retrieve_3d_plot_data(self, column_x, column_y, column_z, N=2,
                              data_is_complete=True, restrictions={},
                              **kwargs):
        """
        Return data ready for a 3D plot using matplotlib.

        :param str column_x: name of x-column
        :param str column_y: name of y-column
        :param str column_z: name of z-column
        :param int N: size of (two-dimensional) smoothing area
        :param bool data_is_complete: defines if all lattice points contain
            data, which is required for fast data retrieval of large datasets,
            where reshaping is used to put data and meshgrid into accordance.
        :return: tuple of x-, y- and z-data and the x-y-extent
        """

        # evaluate options
        try:
            grid = kwargs['grid']
        except KeyError:
            grid = 'square'
        except:
            raise

        # we need a strict ordering for simple reshaping (if possible)
        self.sort([column_y, column_x])

        # Create x-y-arrays
        x = self.get_possible_values(column_x, **restrictions)
        y = self.get_possible_values(column_y, **restrictions)
        X, Y = np.meshgrid(x, y)

        data = self.get_subset(restrictions=restrictions)

        if grid == 'graphenegrid':
            module_logger.debug('assuming graphene grid')

            # for graphene we have to reduce the x-dimension by a factor of 2
            T, Y = np.meshgrid(x[::2], y)
            X = np.zeros(T.shape)

            # let the x-shift depend on the y-coordinates:
            # if the change in y is 1 / sqrt(3), there is
            # no x-shift, otherwise, it is 0.5
            x1 = data[0][column_x] - x[0]
            x3 = 0.5 - x1
            if y[1] - y[0] == 1 / np.sqrt(3):
                x2 = x1
                x4 = 0.5 - x1
            else:
                x2 = 0.5 - x1
                x4 = x1

            X[0::4] = T[0::4] + x1
            X[1::4] = T[1::4] + x2
            X[2::4] = T[2::4] + x3
            X[3::4] = T[3::4] + x4

        extent = (x.min(), x.max(), y.min(), y.max())

        Z = np.zeros(X.shape)

        if data_is_complete is False:
            for row in data:
                xi = 0
                if grid == "graphenegrid":
                    xi = int(np.where(x == row[column_x])[0][0] / 2)
                else:
                    xi = int(np.where(x == row[column_x])[0][0] / 1)
                yi, = np.where(y == row[column_y])[0]
                Z[yi, xi] = row[column_z]
        else:
            # NOTE missing values rot this reshaping, an additional method
            # for that case is the one commented out below, but this is by
            # far less fast
            difference = Y.shape[0] * Y.shape[1] - len(data[:][column_z])
            module_logger.debug('datapoints %i' % len(data[:][column_z]))
            module_logger.debug('grid size %i' % (Y.shape[0] * Y.shape[1]))
            module_logger.debug('difference to optimum entry number %i' %
                                difference)
            if difference > 0:
                data = np.sort(data, order=[column_x, column_y])
                data = np.append(data[:], data[-difference:])
                data = np.sort(data, order=[column_y, column_x])

            module_logger.debug('datapoints %i' % len(data[:][column_z]))
            Z = data[:][column_z].reshape(Y.shape)

        g = gauss_kern(N)
        Z = ssignal.convolve(Z, g, 'same')

        return (X, Y, Z, extent)

    def get_minimum_value(self, column, absolute=False):
        '''
        Return the element of ``column`` with the least value (absolute value
        if ``absolute`` is True).
        '''
        if absolute is True:
            return np.amin(np.abs(self.data[:][column]))
        else:
            return np.amin(self.data[:][column])

    def get_maximum_value(self, column, absolute=False):
        '''
        Return the element of ``column`` with the largest value (absolute value
        if ``absolute`` is True).
        '''
        if absolute is True:
            return np.amax(np.abs(self.data[:][column]))
        else:
            return np.amax(self.data[:][column])

    def get_x_of_minimum_value(self, column_x, column_y, absolute=False):
        '''
        Return the value of ``column_x`` where ``column_y`` has the least
        value (absolute value if ``absolute`` is True).
        '''
        y_value = self.get_minimum_value(column_y, absolute)
        restriction = {column_y: y_value}
        x_values = self.get_column_hard_restriction(column_x, **restriction)

        return (x_values, y_value)

    def get_x_of_maximum_value(self, column_x, column_y, absolute=False):
        '''
        Return the value of ``column_x`` where ``column_y`` has the largest
        value (absolute value if ``absolute`` is True).
        '''
        y_value = self.get_maximum_value(column_y, absolute)
        restriction = {column_y: y_value}
        x_values = self.get_column_hard_restriction(column_x, **restriction)

        return (x_values, y_value)

    def get_column_general(self, column, columns_restriction=[],
                           restriction=None, deletion=False):
        """
        Return a column allowing for very general restrictions.

        :param str column: name of the column to retrieve
        :param list columns_restriction: list of columns that enter the choice
            which entries are choosen
        :param restriction: function or method which expects a list of column
            names and returns a boolean for the selection of the columns
        :param bool deletion: Delete selected rows if set to True
        """
        result = self.data[:]
        indices = np.array(range(self.data.shape[0]))

        i = 0
        for rest_name in columns_restriction:
            result = result[np.where(restriction(i, result[:][rest_name]))]
            indices = indices[np.where(restriction(i, result[:][rest_name]))]
            i += 1

        if deletion:
            self.data = np.delete(self.data, indices, 0)

        return result[:][column]

    def get_column_hard_restriction(self, column, **restrictions):
        """
        Return the content of column desire under hard (==) restrictions.

        :param str column: Name of the column to retrieve
        :param restrictions: Dictionary object with column names as keys and
            the desired values as values.

        Returns an array according to some hard restriction, i.e.
        requesting column "desire" where all the key of "restrictions"
        exactly have the value given by the values of "restrictions"
        """
        result = self.get_subset(restrictions)

        return result[column]

    def retrieve_quiver_plot_data(self, _x, _y, _u, _v, N=5, **kwargs):
        """
        Return data ready for a quiver plot.

        For detailed information what a quiver plot is, please read
        matplotlib documentation; this method basically returns the data
        in a format as the matplotlib.pyplot.quiver method understand

        .. todo::
            haven't used this function for a long time and not sure if this
            really works.
        """
        module_logger.info("retrieve_quiver_plot_data, x-col: %s" % (_x))
        module_logger.info("retrieve_quiver_plot_data, y-col: %s" % (_y))
        module_logger.info("retrieve_quiver_plot_data, u-col: %s" % (_u))
        module_logger.info("retrieve_quiver_plot_data, v-col: %s" % (_v))

        restrictions = {}
        if 'restrictions' in kwargs.keys():
            module_logger.debug("retrieve_quiver_plot_data restrictions: %s" %
                                (kwargs["restrictions"]))
            restrictions = kwargs["restrictions"]

        grid = 'square'
        if 'grid' in kwargs.keys():
            grid = kwargs['grid']

        deletion = False
        if "deletion" in kwargs.keys():
            deletion = kwargs['deletion']

        data = self.get_subset(
            restrictions=restrictions, deletion=deletion)
        data = np.sort(data, order=[_y, _x])

        x = np.unique(data[::][_x])
        y = np.unique(data[::-1][_y])
        #u = (data[::][_u])

        # for graphene we have to reduce the x-dimension by a factor of 2
        X = np.zeros((1, 1))
        Y = np.zeros((1, 1))
        if grid == 'graphenegrid':
            module_logger.debug('assuming graphene grid')
            x0 = x[0]
            x1 = x[0] - x0
            x2 = x[1] - x0
            x3 = x[2] - x0
            x4 = x[3] - x0
            T, Y = np.meshgrid(x[::2], y)
            X = np.zeros(T.shape)
            X[0::4] = T[0::4] + x1
            X[1::4] = T[1::4] + x2
            X[2::4] = T[2::4] + x3
            X[3::4] = T[3::4] + x4
        else:
            module_logger.debug('assuming square grid')
            X, Y = np.meshgrid(x, y)

        extent = (x.min(), x.max(), y.min(), y.max())

        #U = np.zeros(X.shape)
        #V = np.zeros(X.shape)

        # NOTE missing values rot this reshaping, an additional method for that
        # case is the one commented out below, but this is by far less fast
        difference = X.shape[0] * X.shape[1] - len(data[:][_u])
        module_logger.debug('difference to optimum entry number %i' %
                            difference)
        data = np.sort(data, order=[_x, _y])
        data = np.append(data[:], data[-difference:])
        data = np.sort(data, order=[_y, _x])

        U = data[:][_u].reshape(Y.shape)
        V = data[:][_v].reshape(Y.shape)

        g = gauss_kern(N, N)
        xoffset = (Y.shape[1] % N) / 2
        yoffset = (Y.shape[0] % N) / 2
        # module_logger.debug(g)
        U = ssignal.convolve(U, g, 'same')
        V = ssignal.convolve(V, g, 'same')

        X = X[yoffset::N, xoffset::N]
        Y = Y[yoffset::N, xoffset::N]
        U = U[yoffset::N, xoffset::N]
        V = V[yoffset::N, xoffset::N]

        return (X, Y, U, V, extent)

    def mean(self, column):
        """ Return the mean value of ``column``. """
        tmp = self.get_column(column)
        return tmp.mean()

    def get_histogram(self, column, restrictions={}, **kwargs):
        """
        Return a histogram for the values in column ``column``.
        """
        kwargs = dict(kwargs)
        x = self.get_column(column)

        if 'bins' not in kwargs.keys():
            kwargs['bins'] = 20

        hist, bin_edges = np.histogram(x, **kwargs)

        x = np.zeros(hist.shape)
        x = 0.5 * (bin_edges[0:-1] + bin_edges[1:])

        return (x, hist)

    """
    Data manipulation
    """
    def add_column(self, new_name, dtype=np.float64,
                   origin=[], connection=None, args=()):
        """
        Add a new column ``new_name``.

        :param str new_name: Name of the new column, should not be used before.
        :param dtype: Data type of the new column, any Python type is
            allowed.
        :param list origin: List of columns whose values influence the values
            of the new column.
        :param func connection: Reference to function calculating entries of
            the new column from columns given in ``origin``.
        :param tuple args: Further arguments to ``connection``

        If ``connection`` is None, the new column is simply set to zero.
        Otherwise its entries are calculated using the function ``connection``
        with the columns in ``origin`` as input.
        """
        new_datatype = self.dtype
        new_datatype.append((new_name, dtype))

        if connection is None:
            newCol = np.zeros((self.data.size,))
        else:
            # create an two-dimensional array to store the function parameters
            arguments = []

            # the array shall contain the origin columns
            for colname in origin:
                arguments.append(self.data[:][colname])

            # additionally it shall contain the scalar values from args
            for value in args:
                arguments.append(value)

            # now calculate the new column
            newCol = connection(*arguments)

        module_logger.debug("new column %s: %s" % (new_name, newCol))

        tmp = np.empty(self.data.shape, dtype=new_datatype)
        for name in self.data.dtype.names:
            tmp[name] = self.data[name]

        tmp[new_name] = newCol

        self.data = tmp

        self.dtype = new_datatype
        self.columns.append(new_name)

    def append_row(self, new_row):
        """
        Append iterable ``new_row`` to the MultiMap.

        Note that ``new_row`` has to fit to the defined data type.
        """
        module_logger.debug("append_row(%s)" % (new_row,))

        try:
            new_row = np.array(tuple(new_row), dtype=self.data.dtype, ndmin=2)
            self.data = np.append(self.data, new_row)
        except:
            raise

    def append_data(self, other):
        """
        Concatenate MultiMap with a second one (with the same dType).

        .. todo::
            Check if combining data with different dtype raises an error.
        """
        try:
            self.data = np.append(self.data, other.data)
        except:
            raise

    def add_to_column(self, column, values):
        """
        Calculate sum of entries in ``column`` and ``values`` and change
        ``column`` accordingly.

        :param str column: Column that will be modified
        :param values: Scalar or numpy.array, scalar will be added to all rows
            the same ways, numpy.arrays should have the same number of entries
            as the MultiMap.
        """
        try:
            self.data[column] = self.data[column] + values.transpose()
        except AttributeError:
            # It seem's we're only adding a single number
            self.data[column] = self.data[column] + values
        except:
            raise

    def multiply_column(self, column, factor):
        """
        Multiply entries in column ``column`` by ``factor``.
        """
        try:
            self.data[column] *= factor
        except:
            raise

    def remove_columns(self, *names_of_columns):
        """
        Remove all columns given in ``names_of_columns``.
        """
        new_datatype = [x
                        for x in self.dtype
                        if x[0] not in names_of_columns]

        tmp = np.empty(self.data.shape, dtype=new_datatype)
        for name in tmp.dtype.names:
            tmp[name] = self.data[name]

        self.data = tmp
        self.dtype = new_datatype
        self.columns = list(self.data.dtype.names)

    def remove_column(self, name_of_column):
        new_datatype = [x for x in self.dtype if x[0] is not name_of_column]

        tmp = np.empty(self.data.shape, dtype=new_datatype)
        for name in tmp.dtype.names:
            tmp[name] = self.data[name]

        self.data = tmp
        self.dtype = new_datatype
        self.columns = list(self.data.dtype.names)

    def rename_column(self, old_column_name, new_column_name):
        """
        Rename column from ``old_column_name`` to ``new_column_name``.
        """
        temporary_data_type = self.dtype
        new_data_type = [(x, y) if x != old_column_name
                         else (new_column_name, y)
                         for x, y in temporary_data_type]
        new_columns = [x for x, y in new_data_type]
        self.columns = new_columns
        self.dtype = new_data_type
        self.data.dtype = new_data_type

    def reduce(self, columns_to_drop=[], static=[],
               statistics=True, method=np.mean, verbose=True):
        """
        Compress data by applying specific operations. Might include reduction
        of available information.

        :param list columns_to_drop: List of columns which can by dropped during
            reduction. This can, for instance, be a seed value which can be
            droppen when averaging over data created for a random initial
            configuration.
        :param list static: Columns listed here are used to classify data, in
            scientific data these might be external parameters which the
            data is created for, respectively.
        :param bool statistics: If this is True, statistical properties of
            processed columns (error, standard error of the mean, variance,
            standard deviation) are calculated and stored in the resulting
            ``MultiMap``.
        :param method: Method used for the reduction. Since the original
            intention of this method was the calculation of averages it defaults
            to :func:`numpy.mean`. If, for instance, reduction is performed
            via integration instead of averaging this might by approximated
            by :func:`numpy.sum`. Any previously defined function can be
            used here.
        """
        if verbose is True:
            print("performing reduction")
            print("    columns to drop: %s" % (columns_to_drop,))
            print("    static columns: %s" % (static,))

        self.add_column('__sorting__', dtype="|S200",
                        origin=static, connection=create_ordering_column)

        module_logger.debug("... added sorting column")

        # define the ordering, needed for fast array manipulation
        sorting_order = static[:]
        sorting_order.extend(columns_to_drop)
        sorting_order.extend(["__sorting__"])
        self.sort("__sorting__")

        # find key indices where __sorting__ assumes a new value
        fastest_axis = self.data[:]["__sorting__"]
        values_of_fastest_axis = self.get_possible_values("__sorting__")
        key_indices = np.searchsorted(fastest_axis, values_of_fastest_axis)
        key_indices = np.append(key_indices, [self.length()])

        # prepare and create new object
        columns_of_new_object = static[:]
        averaging_cols = []
        for col in self.columns:
            if col not in sorting_order:
                averaging_cols.append(col)
                columns_of_new_object.append(col)
                if statistics is True:
                    columns_of_new_object.append("error%s" % col)
                    columns_of_new_object.append("rms%s" % col)

        if statistics is True:
            columns_of_new_object.append("ensemble_size")

        # here the actual reduction begins
        module_logger.info(
            "... start with the actual reduction with %i key indices" %
            key_indices.shape[0])

        self.reduction_single_core(
            static, averaging_cols, columns_of_new_object,
            key_indices, method, statistics, verbose)

    def reduction_single_core(self, static, averaging_cols,
                              columns_of_new_object,
                              key_indices, method, statistics,
                              verbose=True):
        """
        Perform the actual reduction triggered by :meth:`reduce`.

        This method is intended for internal use only. It gets from within
        :meth:`reduce` and is responsible for the actual reduction.
        By default this function gets called while
        :meth:`reduction_distributed_dispy` might be favorable on machines
        running a dispynode.
        """
        import time

        # setup progressbar
        symbols = itertools.cycle(r"|/-\\")
        progressbar_width = 80

        try:
            pb_step = int(len(key_indices) / progressbar_width)
            pb_indices = list(
                (key_indices[::pb_step])[-progressbar_width + 1:])
            pb_indices.append(key_indices[-1])
        except ValueError:
            progressbar_width = len(key_indices) - 1
            pb_indices = key_indices
        #pb_indices2 = []

        if verbose is True:
            sys.stdout.write("[%s]" % (" " * progressbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (progressbar_width + 1))

        start = time.clock()
        new_object = MultiMap(columns=columns_of_new_object)
        for idx, i in enumerate(key_indices[1:]):
            current_subset = self.data[key_indices[idx]:i]

            new_row = reduce_subset(
                current_subset[:], static, averaging_cols, statistics, method)
            new_object.append_row(new_row)

            if verbose is True:
                if i in pb_indices:
                    sys.stdout.write("-")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(symbols.next())
                    sys.stdout.write("\b")
                    sys.stdout.flush()
        if verbose is True:
            sys.stdout.write("\n")

            end = time.clock()
            print(end - start)

        self.data = new_object.data
        self.dtype = new_object.dtype
        self.columns = list(self.data.dtype.names)

    def reduction_distributed_dispy(self, static, averaging_cols,
                                    columns_of_new_object, key_indices,
                                    method, statistics):
        """
        Perform the actual reduction triggered by :meth:`reduce` distributed
        among several CPUs using dispy.

        This method is intended for internal use only. It gets from within
        :meth:`reduce` and is responsible for the actual reduction.
        Currently, the choice of the reduction method is hard-coded to
        :meth:`reduction_single_core`. Taking the advantage of multi-core
        reduction has to be changed by hand.

        .. todo:
            Make choice of reduction method easier.
        """
        import dispy
        import time

        # setup progressbar
        symbols = itertools.cycle(r"|/-\\")
        progressbar_width = 80

        try:
            pb_step = int(len(key_indices) / progressbar_width)
            pb_indices = list(
                (key_indices[::pb_step])[-progressbar_width + 1:])
            pb_indices.append(key_indices[-1])
        except ValueError:
            progressbar_width = len(key_indices)
            pb_indices = range(progressbar_width)
        pb_indices2 = []

        cluster = dispy.JobCluster(reduce_subset)
        jobs = []

        sys.stdout.write("submitting: [%s]" % (" " * progressbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progressbar_width + 1))

        start = time.clock()
        for idx, i in enumerate(key_indices[1:]):
            current_subset = self.data[key_indices[idx]:i][:]

            job = cluster.submit(current_subset, static,
                                 averaging_cols, statistics, method)
            jobs.append(job)

            if i in pb_indices:
                sys.stdout.write("-")
                sys.stdout.flush()
                pb_indices2.append(idx)
            else:
                sys.stdout.write(symbols.next())
                sys.stdout.write("\b")
                sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.write("collecting: [%s]" % (" " * progressbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progressbar_width + 1))

        new_object = MultiMap(columns=columns_of_new_object)
        for idx, job in enumerate(jobs):
            new_row = job()
            new_object.append_row(new_row)

            if idx in pb_indices2:
                sys.stdout.write("-")
                sys.stdout.flush()
            else:
                sys.stdout.write(symbols.next())
                sys.stdout.write("\b")
                sys.stdout.flush()

        sys.stdout.write("\n")
        cluster.stats()

        end = time.clock()
        print(end - start)

        self.data = new_object.data
        self.dtype = new_object.dtype
        self.columns = list(self.data.dtype.names)

if __name__ == "__main__":
    print("nothing to do here...")
