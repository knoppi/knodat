.. include:: global.rst

Filesystem wrapper
==================

The functions were written for a better organisation. Historically, they
became neccessary when my numerical output covered several gigabytes.
My particular way of parallelization made it convenient to store data
in many files representing different input parameters.

So, for instance, I investigated electron transport in graphene covered by 
different adatoms. Different files were created for the various adatoms and
geometries of the nanostructures. In one file I combined the transmission
data at different energies and for different realizations of the random
system configuration (in that case the positions of the adatoms).
The functions listed here in combination with the facilities of 
:py:class:`~knodat.multimap.MultiMap` made it possible to keep an overview 
and quickly create plots of important subsets of the data.

.. automodule:: knodat.fs_wrapper
    :members:

