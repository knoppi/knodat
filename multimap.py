#!/usr/bin/env python
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
    x, y = np.mgrid[-size:size + 1, -sizey:sizey + 1]
    g = np.exp(- (x ** 2 / float(size) + y ** 2 / float(sizey)))
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

    :param str _fileName: Name of a file to load the data from.
    :param list _cols: List of column names

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

    def __init__(self, _fileName=None, _cols=None):
        module_logger.info("MultiMap initialization:")
        module_logger.debug("-- filename: %s" % _fileName)
        module_logger.debug("-- columns: %s" % _cols)

        self.commentIndicators = ['#']
        self.columns = []
        self.dataType = []

        if _cols is not None:
            self.set_column_names(*_cols)
        if _fileName is not None:
            if _fileName[-4:] == '.npy':
                self.read_file_numpy_style(_fileName)
            else:
                self.read_file(_fileName)

        # As long as we do not define a column to be the indexing column
        # getitem_by_index should be chosen to retrieve a single row
        self.getitem_method = self.getitem_by_index

        # we need a numerical zero
        self.zero = 1e-10

        # we need a default running average size
        self.running_average_N = 1

        # we need a default grid
        self.chosen_grid = "squaregrid"

        # we need to set the completeness of our data
        self.complete = True

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
    #
    section_structure = True


    def describe(self, few_values=10):
        """
        Describe data contents similar to MySQL describe of tables.

        :param int few_values: Data description states the range of entries
            unless less than :py:obj:`few_values` are present in the
            :py:class:`MultiMap`.
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

        print "MultiMap contains %i entries" % self.length()
        print "MultiMap shape: %s" % (self.data.shape, )

        if self.length() > 1:
            for column in self.columns:
                values = self.get_possible_values(column)
                formatting = "%27s %13s: %s"
                if len(values) == 1:
                    print formatting % ("fixed parameter", column, values[0])
                elif len(values) < 5:
                    print formatting % ("parameter with few values",
                                        column, values)
                else:
                    limits = ("%s - %s" % (min(values), max(values)))
                    description = "parameter with %i values" % len(values)
                    print formatting % (description, column, limits)

    def length(self):
        """
        Return the number of data entries.
        """
        try:
            return self.data.shape[0]
        except IndexError:
            # let's just assume that we only have one entry
            # print self.data
            return 1


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
            standardType = np.float64

        for i in column_names:
            new_data_type.append((i, standardType))

        self.set_data_type(new_data_type)

    def set_data_type(self, new_dataType):
        """
        Set the data type of the internal data storage to ``new_dataType``
        and creates a new empty numpy array.

        This method is useful in particular is several different data types
        shall be used (like mixtures of strings and numerical values).
        For more details see the :class:`~numpy:numpy.dtype` documentation.
        """
        module_logger.debug("... data type: %s" % new_dataType)
        self.dataType = new_dataType
        self.data = np.zeros(
            (0, len(new_dataType)),
            dtype=new_dataType)

        self.columns = [x[0] for x in new_dataType]

    def change_data_type_for_column(self, column, new_data_type):
        """
        Modify the datatype of a single column.
        """
        module_logger.debug("changing data type for column %s to %s" %
                            (column, new_data_type))
        icolumn = self.columns.index(column)
        new_dType = self.dataType[:]
        new_dType[icolumn] = (column, new_data_type)

        self.set_data_type(new_dType)

    def set_zero(self, value):
        """
        Set numerical zero used in comparison functions can also be used to
        control accuracy.
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


    def select_indexing_column(self, name):
        """
        Choose one column containing the indices which then can be called by
        getitem.

        .. todo::
            Does this work already?

        """
        self.named_indices = True
        try:
            self.keys = self.get_column(name).tolist()
            self.index_column = name
        except None:
            module_logger.fatal("chosen invalid key-column")

    def set_x_column(self, x_name):
        """
        Choose one column containing the indices which then can be called by
        getitem.

        .. todo::
            Does this work already?

        """
        self.x_column_name = x_name
        try:
            self.x_column = self.get_column(x_name).tolist()
        except Exception:
            raise
        else:
            self.getitem_method = self.getitem_by_x

    def __getitem__(self, i):
        """
        Retrieve a single row of the MultiMap as a dict, i can either be an
        integer or a variable type key
        """
        return self.getitem_method(i)

    def getitem_by_index(self, i):
        """
        Return row i of the multimap as a dictionary.

        .. todo::
            What good is this for? Can it be deleted?
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
        Returns a single row an object as a dictionary defined by col_x = i.

        .. todo::
            What good is this for? Can it be deleted?
        """
        try:
            j = self.x_column.index(i)
            return self.getitem_by_index(j)
        except Exception:
            raise

    def getitem(self, i):
        """
        Return a single row of the MultiMap as a dict.
        """
        self.__getitem__(i)

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
        soft_restriction = lambda n, x: np.abs(x - R[n][1]) < self.zero
        hard_restriction = lambda n, x: (x == R[n][1])

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

    def show(self, **restrictions):
        """
        Convenience method for showing (and retrieving) data

        .. todo::
            WTF??
        """
        return self.get_subset(restrictions=restrictions)

    def get_column_by_index(
            self, _col_index, _col2_indices=[], _lambda=None,
            _deletion=False):
        """
        Return an array containing content of _col_index where the
        columns in _col2_index fulfill the corresponding condition in
        _lambda _col_index is a single integer, _col2_index is an
        array of integers, _lambda is a function taking two arguments:
        the index of the current restriction and the corresponding
        restriction.  Example::

            def restrictions(n, x):
                if n == 0: return (x>0)
                if n == 1: return (x<0)

            desired_values = example_multi_map.getColumnGeneral(
                1, [2,3], restrictions)

        this piece of code fetches all values in column 1, where column
        2 contains a positive value and column 3 contains a negative value

        .. todo::
            This method might be obsolete.
        """
        col_name = self.columns[_col_index]
        col2_names = []
        for index in _col2_indices:
            col2_names.append(self.columns[index])
        return self.get_column_general(
            col_name, col2_names, _lambda, _deletion)

    def get_column_general(self, _col_name, _col2_names=[], _lambda=None,
                           _deletion=False):
        """
        Return the contents of a column according to a previously defined
        function.

        :param str _col_name: name of the column to retrieve
        :param list _col2_names2: list of columns that enter the choice which
            entries are choosen
        :param _lambda: function or method which expects a list of column names
            and returns a boolean for the selection of the columns
        :param bool _deletion: Delete selected rows if set to True

        .. todo::
            improvements needed
        """
        result = self.data[:]
        indices = np.array(range(self.data.shape[0]))

        i = 0
        for rest_name in _col2_names:
            result = result[np.where(_lambda(i, result[:][rest_name]))]
            indices = indices[np.where(_lambda(i, result[:][rest_name]))]
            i += 1

        if _deletion:
                self.data = np.delete(self.data, indices, 0)

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
        """
        Return column ``desire`` without applying any restrictions as
        :class:`numpy.ndarray`-object.
        """
        return self.data[:][desire]

    # methods for write access to the data in the multimap
    #
    section_write_access = True

    def add_column(self, new_name, dataType=np.float64,
                   origin=[], connection=None, args=()):
        """
        Add a new column ``new_name``.

        :param str new_name: Name of the new column, should not be used before.
        :param dataType: Data type of the new column, any Python type is
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
        new_datatype = self.dataType
        new_datatype.append((new_name, dataType))

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

        self.dataType = new_datatype
        self.columns.append(new_name)

    def append_row(self, row):
        """
        Append iterable ``row`` to the MultiMap.

        Note that ``row`` has to fit to the defined data type.
        """
        module_logger.debug("append_row(%s)" % (row,))

        try:
            new_row = np.array(tuple(row), dtype=self.data.dtype, ndmin=2)
            self.data = np.append(self.data, new_row)
        except:
            raise

    def append_data(self, other):
        """ Concatenate MultiMap with a second one (with the same dType). """
        try:
            self.data = np.append(self.data, other.data)
        except:
            raise

    def add_to_column(self, col, values):
        """
        Add ``values`` to ``col``.

        :param str col: Column that will be modified
        :param values: Scalar or numpy.array, scalar will be added to all rows
            the same ways, numpy.arrays should have the same number of entries
            as the MultiMap.
        """
        try:
            self.data[col] = self.data[col] + values.transpose()
        except AttributeError:
            # It seem's we're only adding a single number
            self.data[col] = self.data[col] + values
        except:
            raise

    def multiply_column(self, col, factor):
        """
        Multiply entries in column ``col`` by ``factor``.
        """
        try:
            self.data[col] *= factor
        except:
            raise

    def remove_columns(self, *names_of_columns):
        """
        Remove all columns given in ``names_of_columns``.
        """
        new_datatype = [x
                        for x in self.dataType
                        if x[0] not in names_of_columns]

        tmp = np.empty(self.data.shape, dtype=new_datatype)
        for name in tmp.dtype.names:
            tmp[name] = self.data[name]

        self.data = tmp
        self.dataType = new_datatype
        self.columns = list(self.data.dtype.names)

    def remove_column(self, name_of_column):
        new_datatype = [x for x in self.dataType if x[0] is not name_of_column]

        tmp = np.empty(self.data.shape, dtype=new_datatype)
        for name in tmp.dtype.names:
            tmp[name] = self.data[name]

        self.data = tmp
        self.dataType = new_datatype
        self.columns = list(self.data.dtype.names)

    def rename_column(self, old_column_name, new_column_name):
        """
        Rename column from ``old_column_name`` to ``new_column_name``.
        """
        temporary_data_type = self.dataType
        new_data_type = [(x, y) if x != old_column_name
                         else (new_column_name, y)
                         for x, y in temporary_data_type]
        new_columns = [x for x, y in new_data_type]
        self.columns = new_columns
        self.dataType = new_data_type
        self.data.dtype = new_data_type

    def reduce(self, columns_to_drop=[], static=[],
               statistics=True, method=np.mean):
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
        print "performing reduction"
        print "    columns to drop: %s" % (columns_to_drop,)
        print "    static columns: %s" % (static,)

        self.add_column('__sorting__', dataType="|S200",
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
            key_indices, method, statistics)

    def reduction_single_core(self, static, averaging_cols,
                              columns_of_new_object,
                              key_indices, method, statistics):
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

        sys.stdout.write("[%s]" % (" " * progressbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progressbar_width + 1))

        start = time.clock()
        new_object = MultiMap(_cols=columns_of_new_object)
        for idx, i in enumerate(key_indices[1:]):
            current_subset = self.data[key_indices[idx]:i]

            new_row = reduce_subset(
                current_subset[:], static, averaging_cols, statistics, method)
            new_object.append_row(new_row)

            if i in pb_indices:
                sys.stdout.write("-")
                sys.stdout.flush()
            else:
                sys.stdout.write(symbols.next())
                sys.stdout.write("\b")
                sys.stdout.flush()
        sys.stdout.write("\n")

        end = time.clock()
        print end - start

        self.data = new_object.data
        self.dataType = new_object.dataType
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

        new_object = MultiMap(_cols=columns_of_new_object)
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
        print end - start

        self.data = new_object.data
        self.dataType = new_object.dataType
        self.columns = list(self.data.dtype.names)

    def average(self, columns_to_keep, columns_to_average):
        """
        Effectively throw away a lot of data.

        .. todo::
            Am I allowed to delete this?
        """
        # We do the averaging by creating a new ndarray.
        # The columns of the new array will be named. The names will be..

        # ... the "columns_to_keep" as sort of x-values
        columns_of_new_object = columns_to_keep[:]

        # ... the averaged columns, as well as their sem and rms
        for column in columns_to_average:
            columns_of_new_object.append(column)
            columns_of_new_object.append("error%s" % column)
            columns_of_new_object.append("rms%s" % column)

        # an extra column for the ensemble_size
        columns_of_new_object.append("ensemble_size")

        # With the columns defiend create the new multimap object
        temp_multimap = MultiMap(_cols=columns_of_new_object)

        # A somehow complicated procedure for finding the correct rows
        restriction_catalogue = [[]]

        for abscissa in columns_to_keep:
            possible_values = self.get_possible_values(abscissa)
            restriction_catalogue = [
                row + [(abscissa, x)]
                for x in possible_values
                for row in restriction_catalogue]

        module_logger.debug("averager: restrictions prepared")

        # turn restriction_catalogue into an array of dicts
        restriction_catalogue = [dict(x) for x in restriction_catalogue]

        for restriction in restriction_catalogue:
            t = self.get_subset(restriction)
            ensemble_size = t.shape[0]

            if t.shape[0] > 0:
                new_row = []

                for col in columns_to_keep:
                    new_row.append(restriction[col])
                for col in columns_to_average:
                    average = t[col].mean(dtype=np.float64)
                    std = t[col].std()
                    new_row.append(average)
                    new_row.append(std / np.sqrt(ensemble_size))
                    new_row.append(np.sqrt(np.mean((t[col] - average) ** 2)))

                new_row.append(ensemble_size)
                temp_multimap.append_row(new_row)

        self.data = temp_multimap.data
        self.dataType = temp_multimap.dataType

    # section for interactions with the filesystem
    #
    section_filesystem_interactions = True

    def read_file(self, filename, **options):
        """
        Read data from ascii file.

        :param str filename: Name of the file data is loaded from.

        This method reads data from an ascii file using the numpy function
        loadtxt which is in particular suited for reading csv-datafile.
        Further parameters are required for the control of the loadtxt function.
        """
        self.filename = filename
        for (key, val) in options:
            if key == "delimiter":
                self.separator = val
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

        self.data = np.loadtxt(filename, dtype=self.dataType)

        module_logger.debug("finished reading file")

    def read_file_numpy_style(self, filename):
        '''
        Load content of a compressed numpy ND-array

        :param str filename: file to load
        '''
        self.filename = filename
        self.data = np.load(filename)
        self.dataType = self.data.dtype.descr
        self.columns = []
        for item in self.dataType:
            self.columns.append(item[0])

    def sort(self, column):
        """
        Sort the MultiMap along :py:obj:`column`.
        """
        self.data = np.sort(self.data, order=column)

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

    def write_file_for_gnuplot_splot(self, _x, _y, _filename):
        """ when doing surface plots with gnuplot there have to be empty lines
            in the data file after each isoline which can be achieved by this
            method
        """
        outfile = open(_filename, 'w')

        xcol = np.str(_x)
        ycol = np.str(_y)

        xy_columns = [ycol, xcol]
        self.sort(xy_columns)

        lastx = self.data[0][xcol]
        for row in self.data:
            if row[xcol] < lastx:
                print ""
                outfile.write("\n")
            print " ".join([np.str(x) for x in row])
            outfile.write(" ".join([np.str(x) for x in row]))
            outfile.write("\n")
            lastx = row[xcol]

    # def read_file_numpy_style(self, filename):
        #'''loads content into the MultiMap which was formerly saved as a
        # numpy style .npy file'''
        #self.data = np.load(filename)

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

    def get_x_of_minimum_value(self, x_column, y_column, absolute=False):
        '''
        Return the value of ``x_column`` where ``y_column`` has the least
        value (absolute value if ``absolute`` is True).
        '''
        y_value = self.get_minimum_value(y_column, absolute)
        restriction = {y_column: y_value}
        x_values = self.get_column_hard_restriction(x_column, **restriction)

        return (x_values, y_value)

    def get_x_of_maximum_value(self, x_column, y_column, absolute=False):
        '''
        Return the value of ``x_column`` where ``y_column`` has the largest
        value (absolute value if ``absolute`` is True).
        '''
        y_value = self.get_maximum_value(y_column, absolute)
        restriction = {y_column: y_value}
        x_values = self.get_column_hard_restriction(x_column, **restriction)

        return (x_values, y_value)

    def mean(self, column):
        """ Return the mean value of ``column``. """
        tmp = self.get_column(column)
        return tmp.mean

    def get_possible_values(self, colName, **restrictions):
        """
        Return a unique list of values in column ``colName``.

        The list of ``restrictions`` can be used to limit the resulting values.
        This method internally calls :meth:`get_column_hard_restriction` and
        removes duplicates from the result.
        """
        temp = np.unique(
            self.get_column_hard_restriction(colName, **restrictions))
        return temp

    def get_histogram(self, col, restrictions={}, **kwargs):
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

    def get(self, *args, **keywargs):
        """
        Obtain complete subsets or 1- to 3-column slices of the ``MultiMap``
        following restrictions defined in ``keywargs``.

        This methods refers to
            * :meth:`get_subset`, if ``args`` has length 0,
            * :meth:`get_possible_values`, if ``args`` has length 1,
            * :meth:`retrieve_2d_plot_data` if ``args`` has length 2,
            * :meth:`retrieve_3d_plot_data` if ``args`` has length 3.

        The options given in ``keywargs`` are used only for restrictions.
        In order to address the further parameters of the relevant methods
        use :meth:`set_N`, :meth:`set_grid` and :meth:`set_complete`.

        For details see the documentation of the respective methods.
        """
        if len(args) == 0:
            R = dict(keywargs)
            return self.get_subset(R)
        if len(args) == 1:
            return self.get_possible_values(
                *args, **keywargs)
        if len(args) == 2:
            return self.retrieve_2d_plot_data(
                *args, restrictions=keywargs,
                N=self.running_average_N)
        if len(args) == 3:
            return self.retrieve_3d_plot_data(
                *args, restrictions=keywargs,
                N=self.running_average_N, grid=self.chosen_grid,
                data_is_complete=self.complete)
        else:
            pass

    def retrieve_2d_plot_data(
        self, _colx, _coly, errx=None, erry=None, N=1,
            restrictions={}, trim=True):
        """
        Return data ready for 2D plotting.

        :param str _colx: name of x-column
        :param str _coly: name of y-column
        :param str errx: name of column containing x error
        :param str erry: name of column containing y error
        :param int N: number of points included in running average
        :param dict restrictions: limit to a certain subset
        :param bool trim: don't know
        :return: tuple with x-, y-data and possibly errorbar sizes.

        .. todo::
            What is trim?

        This method returns a tuple of array containing the data for the plot.
        Size of the tuple depends on given arguments. If errorbars are desired
        they are included in the returned array.
        """
        _colx = str(_colx)
        _coly = str(_coly)
        module_logger.debug(
            "retrieving plot data %s vs %s" % (_colx, _coly))
        module_logger.debug("errorbars are %s, %s" % (errx, erry))
        self.sort(_colx)
        xvals = self.get_column_hard_restriction(_colx, **restrictions)
        yvals = self.get_column_hard_restriction(_coly, **restrictions)
        xerrs = (0 if errx is None else
                 self.get_column_hard_restriction(errx, **restrictions))
        yerrs = (0 if erry is None else
                 self.get_column_hard_restriction(erry, **restrictions))

        if N > 1:
            g = gauss_kern_1d(N)
            module_logger.debug(g)
            #xvals = ssignal.convolve(xvals, g, 'same')
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

        if errx is None and erry is None:
            return (xvals, yvals)
        elif errx is None and erry is not None:
            return (xvals, yvals, yerrs)
        elif errx is not None and erry is None:
            return (xvals, yvals, xerrs)
        else:
            return (xvals, yvals, xerrs, yerrs)

    def plot(self, x, y, fmt="-", **restrictions):
        import matplotlib.pyplot as plt
        xdata, ydata = self.retrieve_2d_plot_data(
            x, y, restrictions=restrictions)

        plt.plot(xdata, ydata, fmt)

        return xdata, ydata

    def retrieve_3d_plot_data(self, _x, _y, _z, N=2, data_is_complete=True,
                              *args, **kwargs):
        """
        Return data ready for a 3D plot using matplotlib.

        :param str _x: name of x-column
        :param str _y: name of y-column
        :param str _z: name of z-column
        :param int N: size of (two-dimensional) smoothing area
        :param bool data_is_complete: defines if all lattice points contain
            data, which is required for fast data retrieval of large datasets,
            where reshaping is used to put data and meshgrid into accordance.
        :return: tuple of x-, y- and z-data and the x-y-extent
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

        deletion = False
        if "deletion" in kwargs.keys():
            deletion = kwargs['deletion']

        # retrieve the subset according to restrictions
        data = self.get_subset(
            restrictions=restrictions, deletion=deletion)
        data = np.sort(data, order=[_y, _x])

        x = np.unique(data[:][_x])
        y = np.unique(data[::-1][_y])

        # for graphene we have to reduce the x-dimension by a factor of 2
        X = np.zeros((1, 1))
        Y = np.zeros((1, 1))
        if grid == 'graphenegrid':
            module_logger.debug('assuming graphene grid')
            T, Y = np.meshgrid(x[::2], y)
            X = np.zeros(T.shape)
            module_logger.debug('grid dimensions: %s' % (X.shape, ))

            # let the x-shift depend on the y-coordinates:
            # if the change in y is 1 / sqrt(3), there is
            # no x-shift, otherwise, it is 0.5
            x1 = 0.0
            if y[1] - y[0] == 1 / np.sqrt(3):
                x1 = data[0][_x] - x[0]
                x2 = x1
                x3 = 0.5 - x1
                x4 = 0.5 - x1
            else:
                x1 = data[0][_x] - x[0]
                x2 = 0.5 - x1
                x3 = 0.5 - x1
                x4 = x1

            #x1 = 0.5
            #x2 = 0
            #x3 = x2
            #x4 = x1
            # print x1, x2, x3, x4

            X[0::4] = T[0::4] + x1
            X[1::4] = T[1::4] + x2
            X[2::4] = T[2::4] + x3
            X[3::4] = T[3::4] + x4
        else:
            module_logger.debug('assuming square grid')
            module_logger.debug("x-shape: %s" % x.shape)
            module_logger.debug("y-shape: %s" % y.shape)
            X, Y = np.meshgrid(x, y)

        extent = (x.min(), x.max(), y.min(), y.max())

        Z = np.zeros(X.shape)

        if data_is_complete is False:
            for row in data:
                xi = 0
                if grid == "graphenegrid":
                    xi = int(np.where(x == row[_x])[0][0] / 2)
                else:
                    xi = int(np.where(x == row[_x])[0][0] / 1)
                yi, = np.where(y == row[_y])[0]
                Z[yi, xi] = row[_z]
        else:
            # NOTE missing values rot this reshaping, an additional method
            # for that case is the one commented out below, but this is by
            # far less fast
            difference = Y.shape[0] * Y.shape[1] - len(data[:][_z])
            module_logger.debug('datapoints %i' % len(data[:][_z]))
            module_logger.debug('grid size %i' % (Y.shape[0] * Y.shape[1]))
            module_logger.debug('difference to optimum entry number %i' %
                                difference)
            if difference > 0:
                data = np.sort(data, order=[_x, _y])
                data = np.append(data[:], data[-difference:])
                data = np.sort(data, order=[_y, _x])

            module_logger.debug('datapoints %i' % len(data[:][_z]))
            Z = data[:][_z].reshape(Y.shape)

        if N > 1:
            g = gauss_kern(N)
            Z = ssignal.convolve(Z, g, 'same')

        return (X, Y, Z, extent)

    def retrieve_quiver_plot_data(self, _x, _y, _u, _v, N=5, **kwargs):
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
        # if len(data[:][_u]) == X.shape[0]*X.shape[1]:
            #U = data[:][_u].reshape(Y.shape)
            #V = data[:][_v].reshape(Y.shape)
        # else:
            # for row in data:
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
        # module_logger.debug(g)
        U = ssignal.convolve(U, g, 'same')
        V = ssignal.convolve(V, g, 'same')

        X = X[yoffset::N, xoffset::N]
        Y = Y[yoffset::N, xoffset::N]
        U = U[yoffset::N, xoffset::N]
        V = V[yoffset::N, xoffset::N]

        return (X, Y, U, V, extent)

    # Old functions with camelCase-naming, deleted on 2014-05-13
    #
    # def setColumnNames(self, *keyw, **options):
    # def setDataType(self, new_dataType):
    # def readFile(self, filename, **options):
    # def appendRow(self, row):
    # def addColumn(self, name, dataType="|f8", origin=[], connection=None):
    # def writeFile(self, filename, **options):
    # def getIndexedColumnGeneral(self, _col_index, _col2_indices=[],
                                #_lambda=None, _deletion=False):
    # def getColumnHardRestriction(self, desire, **restrictions):
    # def pullRows(self, desire, **restrictions):
    # def plot2dData(self, _colx, _coly, errx=None, erry=None,
                          # fmt=fmt)
    # def plot_2d_data(self, _colx, _coly, errx=None, erry=None,
                     # restrictions={}, label="", fmt="", **options):
    # def retrieve2dPlotData(self, _colx, _coly, errx=None, erry=None,
                           # restrictions={}):
    # def retrieve3dPlotData(self, _x, _y, _z, **kwargs):
    # def retrieveQuiverPlotData(self, _x, _y, _u, _v, **kwargs):
    # def getColumnGeneral(self, _col_name, _col2_names=[],
                         #_lambda=None, _deletion=False):
    # def getColumn(self, desire):
    # def getPossibleValues(self, colName, **restrictions):

if __name__ == "__main__":
    set_debug_level("debug")
    module_logger.info("create multimap with column a, b and c")

    test_object = MultiMap(_cols=["a", "b", "c"])

    module_logger.info("multimap created")

    module_logger.info("fill test object with content")
    test_object.append_row([1, 2, 3])
    test_object.append_row([4, 5, 6])
    test_object.append_row([7, 8, 9])

    module_logger.info("test object filled with: %s" % test_object.data)

    module_logger.info("adding column 'd' filled with zeros")
    test_object.add_column("d")
    module_logger.info("test object in new state: %s" % test_object.data)

    module_logger.info(
        "adding column 'e' filled with the sum of columns a and b")
    test_object.add_column("e", origin=["a", "b"], connection=np.add)
    module_logger.info("test object in new state: %s" % test_object.data)

    module_logger.info("row 1 contains: %s" % test_object[0])

    module_logger.info("choose column 'b' for indexing")
    test_object.select_indexing_column("b")
    module_logger.info("row with key %s contains: %s" %
                       (5.0, test_object[5.0]))
    module_logger.info("calling non-existent key %s yields: %s" %
                       (3.0, test_object[3.0]))
    module_logger.info("calling non-existent key %s yields: %s" %
                       (1.0, test_object[1.0]))

    test_object.average(["a", "c"], ["d", "e"])
