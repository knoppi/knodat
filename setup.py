from setuptools import setup, find_packages
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'DESCRIPTION.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='KnoDat',
    version='0.9',
    description='A set of tools developed for treatment of numerical data',
    long_description=long_description,
    url='http://knodat.knofafo.de',
    author='Jan Bundesmann',
    author_email='jan.bundesmann@gmx.de',
    license='GPLv2+',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)'
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='numerics data analysis',
    packages=find_packages(exclude=['docs', 'tests*']),
    install_requires=['numpy', 'scipy'],
)
