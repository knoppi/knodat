.. KnoDat documentation master file, created by
   sphinx-quickstart on Tue Feb  3 13:09:38 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to KnoDat's documentation!
==================================

KnoDat is a collection of python tools mainly for the evaluation of scientific 
data. Also included within this suite are helping tools used within this 
context, like reading directory content and putting it into a readable format, 
as well as parsing filename according to a given naming scheme.

In order to be able to use the tools, the following requirements have to be met:

    - Python should be installed, I'm currently running several version, 2.6.6 works fine, 
      no 3.0 version has been tested so far
    - Many tools use Numpy and Scipy for mathematical routines or data evaluation
    - For plotting Matplotlib is used

Usage instructions:

    - Install the package into a folder which is contained within you PYTHONPATH
      environment variable, then everything should be fine
    - Within knodat/evtools there are scripts which can be used directly from the
      command line. They require the last step and it is recommended to put
      their location into your PATH variable

Contents:

.. toctree::
   :maxdepth: 2

   multimap
   fs_wrapper

TODOS
============
.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

