.. include:: global.rst

Multimap
==============

.. automodule:: knodat.multimap

    .. autoclass:: MultiMap

        We group the methods as
            #. :ref:`Deprecated methods <deletable>`
            #. :ref:`Filesystem access <fs_methods>`
            #. :ref:`Descriptive methods <descriptive_methods>`
            #. :ref:`Metadata manipulation <metadata_manipulation>`
            #. :ref:`Data retrieval <data_retrieval>`
            #. :ref:`Data manipulation <data_manipulation>`
        
        .. _deletable:

        :methodgroup:`To delete`

        .. automethod:: select_indexing_column
        .. automethod:: set_x_column
        .. automethod:: show
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
        .. automethod:: sort

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
        .. automethod:: get_column_general
        .. automethod:: get_column_hard_restriction
        .. automethod:: retrieve_quiver_plot_data
        .. automethod:: get_histogram
        .. automethod:: mean

        .. _data_manipulation:
        
        :methodgroup:`Data manipulation`

        The following methods were written with the intention to easily alter 
        the stored data. While other methods also change the object in one way 
        or the other, these functions substantiantly modify the data - 
        potentially in an irreversible manner.


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
