.. include:: global.rst

knodat package
==============

module
----------------------

.. automodule:: knodat.multimap

    .. autoclass:: MultiMap

        We group the methods as
            #. :ref:`Unclassified methods <unclassified>`
            #. :ref:`Filesystem access <fs_methods>`
            #. :ref:`Descriptive methods <descriptive_methods>`
            #. :ref:`Metadata manipulation <metadata_manipulation>`
            #. :ref:`Data retrieval <data_retrieval>`
            #. :ref:`Data manipulation <data_manipulation>`
        
        .. _unclassified:

        :methodgroup:`Unclassified methods`

        Here go methods that I cannot classify yet.

        .. automethod:: __init__
        .. automethod:: __setitem__
        .. automethod:: __iter__
        .. automethod:: select_indexing_column
        .. automethod:: set_x_column
        .. automethod:: show
        .. automethod:: sort
        .. automethod:: mean
        .. automethod:: plot

        .. _fs_methods:
        
        :methodgroup:`Saving to and loading from filesystem`

        All the methods required for saving and reading file

        .. automethod:: read_file
        .. automethod:: read_file_numpy_style
        .. automethod:: write_file
        .. automethod:: write_file_numpy_style
        .. automethod:: write_file_for_gnuplot_splot

        .. _descriptive_methods:

        :methodgroup:`Descriptive methods`
        
        .. automethod:: describe
        .. automethod:: length

        .. _metadata_manipulation:
        
        :methodgroup:`Metadata manipulation`
        
        .. automethod:: set_column_names
        .. automethod:: set_data_type
        .. automethod:: change_data_type_for_column
        .. automethod:: set_zero
        .. automethod:: set_N
        .. automethod:: set_grid
        .. automethod:: set_complete

        .. _data_retrieval:
        
        :methodgroup:`Data retrieval`

        .. automethod:: get
        .. automethod:: get_column
        .. automethod:: get_subset
        .. automethod:: get_possible_values
        .. automethod:: retrieve_2d_plot_data
        .. automethod:: retrieve_3d_plot_data
        .. automethod:: get_minimum_value
        .. automethod:: get_maximum_value
        .. automethod:: get_x_of_minimum_value
        .. automethod:: get_x_of_maximum_value
        .. automethod:: __getitem__
        .. automethod:: getitem_by_index
        .. automethod:: getitem_by_x
        .. automethod:: getitem
        .. automethod:: get_column_by_index
        .. automethod:: get_histogram
        .. automethod:: get_column_general
        .. automethod:: get_column_hard_restriction
        .. automethod:: retrieve_quiver_plot_data

        .. _data_manipulation:
        
        :methodgroup:`Data manipulation`

        .. automethod:: append_row
        .. automethod:: append_data
        .. automethod:: add_to_column
        .. automethod:: multiply_column
        .. automethod:: add_column
        .. automethod:: remove_columns
        .. automethod:: rename_column
        .. automethod:: reduce
        .. automethod:: reduction_single_core
        .. automethod:: reduction_distributed_dispy
        .. automethod:: average
