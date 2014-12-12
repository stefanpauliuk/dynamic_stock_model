pyDSM
=====

Python class for efficient handling of dynamic stock models (DSMs)

This project contains a class and a connected unit test for modelling dynamic stocks of materials or products, 
as used in dynamic material flow analysis and industrial ecology.

Created on Mon Jun 30 17:21:28 2014

@main author: stefan pauliuk, NTNU Trondheim, Norway <br>
with contributions from <br>
Georgios Pallas, NTNU,<br>
Sebastiaan Deetman, CML, Leiden University, The Netherlands<br>
Chris Mutel, PSI, Villingen, CH<br>


<b> Installation:</b><br><br>
<b>a) As package:</b> <br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then open a console, change to the directory ../pyDSM-master/, and install the package from the command line: <br>
> python setup.py install 

This makes the package available to Python. At any other place in a system with the same python installation, pydsm is now ready to be imported simply by <br>
> import pydsm 

This setup also allows us to run the unit test: <br>

> import unittest

> import pydsm

> import pydsm.tests

> unittest.main(pydsm.tests, verbosity=2)

Or, to run a specific test

> unittest.main(pydsm.tests.test_known_results, verbosity=2)

<br>
<b>b) Manually, by modifying the python path</b><br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then include in your code the following lines <br>
> import sys 

> sys.path.append('C:\\MyPythonPackages\\pyDSM-master\\pydsm\\') 

> from pydsm import DynamicStockModel

<br><br>
<b>Tutorial:</b><br>
http://nbviewer.ipython.org/github/stefanpauliuk/pyDSM/blob/master/Doc/pyDSM_Documentation.ipynb 
