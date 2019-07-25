dynamic_stock_model
=====

Python class for efficient handling of dynamic stock models

This project contains a class and a connected unit test for modelling dynamic stocks of materials or products,
as used in dynamic material flow analysis and industrial ecology.

__Note:__ This project is maintained no longer. The dynamic stock model class is now part of ODYM, the open dynamic material systems model. The new dsm class of ODYM includes a number of lifetime distributions, different dynamic stock models, is more thoroughly tested, and consistently uses the survival function (sf) to model the decay of age-cohorts. Please check here:
https://github.com/IndEcol/ODYM

Created on Mon Jun 30 17:21:28 2014

@main author: stefan pauliuk, NTNU Trondheim, Norway <br>
with contributions from <br>
Chris Mutel, PSI, Villingen, CH<br>

<b>Dependencies:</b> <br>
numpy >= 1.9<br>
scipy >= 0.14<br>


<br>
<b>Tutorial:</b><br>
http://nbviewer.ipython.org/github/stefanpauliuk/dynamic_stock_model/blob/master/Doc/dynamic_stock_model_Documentation.ipynb 
<br><b>Documenation of all methods and functions:</b><br>
http://htmlpreview.github.com/?https://github.com/stefanpauliuk/dynamic_stock_model/blob/master/Doc/dynamic_stock_model.html

<br>

<b> Below, a quick installation guide and a link to the tutorial are provided:</b><br><br>

<b>a) Installation from the web repository:</b> <br>
This is the easiest way of installing dynamic_stock_model. Github hosts an installation package for dynamic_stock_model, which can be downloaded directly from the command line using pip: <br>

> pip install dynamic_stock_model

<b>b) Installation as package:</b> <br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then open a console, change to the directory ../dynamic_stock_model-master/, and install the package from the command line: <br>
> python setup.py install

This makes the package available to Python. At any other place in a system with the same python installation, dynamic_stock_model is now ready to be imported simply by <br>
> import dynamic_stock_model

This setup also allows us to run the unit test: <br>

> import unittest

> import dynamic_stock_model

> import dynamic_stock_model.tests

> unittest.main(dynamic_stock_model.tests, verbosity=2)

Or, to run a specific test

> unittest.main(dynamic_stock_model.tests.test_known_results, verbosity=2)

<br>
<b>c) Manual installation, by modifying the python path</b><br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then include in your code the following lines <br>
> import sys

> sys.path.append('C:\\MyPythonPackages\\dynamic_stock_model-master\\dynamic_stock_model\\')

> from dynamic_stock_model import DynamicStockModel

