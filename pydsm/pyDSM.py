# -*- coding: utf-8 -*-
"""
Class DynamicStockModel, DSM. Version 1.0. 
Last change: May 9th, 2015. 
Check https://github.com/stefanpauliuk/pyDSM for latest version.

Methods for efficient handling of dynamic stock models (DSMs)

Created on Mon Jun 30 17:21:28 2014

@author: Stefan Pauliuk, NTNU Trondheim, Norway

standard abbreviation: DSM

dependencies:
    numpy
    scipy

Repository for this class, documentation, and tutorials: https://github.com/stefanpauliuk/pyDSM

"""

import sys
import logging
import numpy as np
import scipy.stats

# check for correct version number of Python
if sys.version_info.major < 3:
    logging.warning('This package requires Python 3.0 or higher.')

class DynamicStockModel(object):

    """ Class containing a dynamic stock model

    Attributes
    ----------
    t : Series of years or other time intervals
    i : Discrete time series of inflow to stock

    o : Discrete time series of outflow from stock
    o_c :Discrete time series of outflow from stock, by cohort

    s_c : dynamic stock model (stock broken down by year and age- cohort)
    s : Discrete time series for stock, total

    lt : lifetime distribution: dictionary

    pdf: probability density function, distribution of outflow from a specific age-cohort


    name : string, optional
        Name of the dynamic stock model, default is 'DSM'
    """

    """
    Basic initialisation and dimension check methods
    """

    def __init__(self, t=None, i=None, o=None, s=None, lt=None, s_c=None, o_c=None, name='DSM', pdf=None):
        """ Init function. Assign the input data to the instance of the object."""
        self.t = t  # optional

        self.i = i  # optional

        self.s = s  # optional
        self.s_c = s_c  # optional

        self.o = o  # optional
        self.o_c = o_c  # optional

        if lt is not None:
            for ThisKey in lt.keys():
                # If we have the same scalar lifetime, stdDev, etc., for all cohorts,
                # replicate this value to full length of the time vector
                if len(lt[ThisKey]) == 1:
                    lt[ThisKey] = np.tile(lt[ThisKey], len(t))

        self.lt = lt  # optional
        self.name = name  # optional

        self.pdf = pdf # optional

    def return_version_info(self):
        """Return a brief version statement for this class."""
        return str('Class DynamicStockModel, DSM. Version 1.0. Last change: May 9th, 2015. Check https://github.com/stefanpauliuk/pyDSM for latest version.')

    """ Part 1: Checks and balances: """

    def dimension_check(self):
        """ This method checks which variables are present and checks whether data types and dimensions match
        """
        # Compile a little report on the presence and dimensions of the elements in the SUT
        try:
            DimReport = str('<br><b> Checking dimensions of dynamic stock model ' + self.name + '.')
            if self.t is not None:
                DimReport += str('Time vector is present with ' + str(len(self.t)) + ' years.<br>')
            else:
                DimReport += str('Time vector is not present.<br>')
            if self.i is not None:
                DimReport += str('Inflow vector is present with ' +
                                 str(len(self.i)) + ' years.<br>')
            else:
                DimReport += str('Inflow is not present.<br>')
            if self.s is not None:
                DimReport += str('Total stock is present with ' + str(len(self.s)) + ' years.<br>')
            else:
                DimReport += str('Total stock is not present.<br>')
            if self.s_c is not None:
                DimReport += str('Stock by cohorts is present with ' + str(len(self.s_c)
                                                                           ) + ' years and ' + str(len(self.s_c[0])) + ' cohorts.<br>')
            else:
                DimReport += str('Stock by cohorts is not present.<br>')
            if self.o is not None:
                DimReport += str('Total outflow is present with ' +
                                 str(len(self.o)) + ' years.<br>')
            else:
                DimReport += str('Total outflow is not present.<br>')
            if self.o_c is not None:
                DimReport += str('Outflow by cohorts is present with ' +
                                 str(len(self.o_c)) + ' years and ' + str(len(self.o_c[0])) + ' cohorts.<br>')
            else:
                DimReport += str('Outflow by cohorts is not present.<br>')
            if self.lt is not None:
                DimReport += str('Lifetime distribution is present with type ' +
                                 str(self.lt['Type']) + ' and mean ' + str(self.lt['Mean']) + '.<br>')
            else:
                DimReport += str('Lifetime distribution is not present.<br>')
            ExitFlag = 1 # Description of DSM was compiled successfully.
            return DimReport, ExitFlag
        except:
            ExitFlag = 0 # Unable to compile description of DSM.
            return str('<br><b> Checking dimensions of dynamic stock model ' + self.name + ' failed.'), ExitFlag

    def compute_stock_change(self):
        """ Determine stock change from time series for stock. Formula: stock_change(t) = stock(t) - stock(t-1)."""
        if self.s is not None:
            stock_change = np.zeros(len(self.s))
            stock_change[0] = self.s[0]
            for m in range(1, len(self.s)):
                stock_change[m] = self.s[m] - self.s[m - 1]
            ExitFlag = 1  # Method went allright.
            return stock_change, ExitFlag
        else:
            ExitFlag = 0  # No total stock. Calculation of stock change was not possible.
            return None, ExitFlag

    def check_stock_balance(self):
        """ Check wether inflow, outflow, and stock are balanced. If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change"""
        try:
            Balance = self.i - self.o - self.compute_stock_change()[0]
            ExitFlag = 1  # balance computation allright
            return Balance, ExitFlag
        except:
            # Could not determine balance. At least one of the variables is not defined.
            ExitFlag = 0
            return None, ExitFlag

    def compute_stock_total(self):
        """Determine total stock as row sum of cohort-specific stock."""
        if self.s is not None:
            ExitFlag = 2  # Total stock is already defined. Doing nothing.
            return self.s, ExitFlag
        else:
            try:
                self.s = self.s_c.sum(axis=1)
                ExitFlag = 1
                return self.s, ExitFlag
            except:
                ExitFlag = 3  # Could not determine row sum of s_c.
                return None, ExitFlag

    def compute_outflow_total(self):
        """Determine total outflow as row sum of cohort-specific outflow."""
        if self.o is not None:
            ExitFlag = 2  # Total outflow is already defined. Doing nothing.
            return self.o, ExitFlag
        else:
            try:
                self.o = self.o_c.sum(axis=1)
                ExitFlag = 1
                return self.o, ExitFlag
            except:
                ExitFlag = 3  # Could not determine row sum of o_c.
                return None, ExitFlag

    """ Part 2: Lifetime model. """

    def compute_outflow_pdf(self):
        """
        Lifetime model. The method compute outflow_pdf returns an array year-by-cohort of the probability of a item added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly enters the computation. All other stock variables are determined by mass balance.
        The shape of the output pdf array is NoofYears * NoofYears
        """
        if self.pdf is None:
            self.pdf = np.zeros((len(self.t), len(self.t)))
            # Perform specific computations and checks for each lifetime distribution:

            if self.lt['Type'] == 'Fixed':
                for m in range(0, len(self.t)):  # cohort index
                    ExitYear = m + self.lt['Mean'][m]
                    if ExitYear <= len(self.t) - 1:
                        self.pdf[ExitYear, m] = 1
                ExitFlag = 1

            if self.lt['Type'] == 'Normal':
                for m in range(0, len(self.t)):  # cohort index
                    # year index, year larger or equal than cohort
                    for n in range(m + 1, len(self.t)):
                        self.pdf[n, m] = scipy.stats.norm(self.lt['Mean'][m], self.lt['StdDev'][m]).pdf(n - m)  # Call scipy's Norm function with Mean, StdDev, and Age
                ExitFlag = 1

            if self.lt['Type'] == 'Weibull': # Equivalent to the Frechet distribution
                for m in range(0, len(self.t)):  # cohort index
                    # year index, year larger or equal than cohort
                    for n in range(m + 1, len(self.t)):
                        self.pdf[n, m] = scipy.stats.weibull_min(self.lt['Shape'][m], 0, self.lt['Scale'][m]).cdf(n -m) - scipy.stats.weibull_min(self.lt['Shape'][m], 0, self.lt['Scale'][m]).cdf(n -m -1) # Call scipy's Weibull_min function with Shape, offset (0), Scale, and Age
                ExitFlag = 1

            return self.pdf, ExitFlag
        else:
            ExitFlag = 2  # pdf already exists
            return self.pdf, ExitFlag

    """
    Part 3: Inflow driven model
    Given: inflow, lifetime dist.
    Default order of methods:
    1) determine stock by cohort
    2) determine total stock
    2) determine outflow by cohort
    3) determine total outflow
    4) check mass balance.
    """

    def compute_s_c_inflow_driven(self):
        """ With given inflow and lifetime distribution, the method builds the stock by cohort.
        """
        if self.i is not None:
            if self.lt is not None:
                self.s_c = np.zeros((len(self.i), len(self.i)))
                # construct the pdf of a product of cohort tc leaving the stock in year t
                self.compute_outflow_pdf()
                for m in range(0, len(self.i)):  # cohort index
                    self.s_c[m, m] = self.i[m]  # inflow on diagonal
                    for n in range(m, len(self.i)):  # year index, year >= cohort year
                        self.s_c[n, m] = self.i[m] * (1 - self.pdf[0:n + 1, m].sum())
                    ExitFlag = 1
                return self.s_c, ExitFlag
            else:
                ExitFlag = 2  # No lifetime distribution specified
                return None, ExitFlag
        else:
            ExitFlag = 3  # No inflow specified
            return None, ExitFlag

    def compute_o_c_from_s_c(self):
        """Compute outflow by cohort from stock by cohort."""
        if self.s_c is not None:
            if self.o_c == None:
                self.o_c = np.zeros(self.s_c.shape)
                for m in range(0, len(self.s_c)):  # for all cohorts
                    for n in range(m + 1, len(self.s_c)):  # for all years each cohort exists
                        self.o_c[n, m] = self.s_c[n - 1, m] - self.s_c[n, m]
                ExitFlag = 1
                return self.o_c, ExitFlag
            else:
                ExitFlag = 3  # o_c already exists. Doing nothing.
                return self.o_c, ExitFlag
        else:
            ExitFlag = 2  # s_c does not exist. Doing nothing
            return None, ExitFlag

    def compute_i_from_s(self, InitialStock):
        """Given a stock at t0 broken down by different cohorts tx ... t0, an "initial stock". This method calculates the original inflow that generated this stock."""
        if self.i == None:
            if len(InitialStock) == len(self.t):
                self.i = np.zeros(len(self.t))
                # construct the pdf of a product of cohort tc leaving the stock in year t
                self.compute_outflow_pdf()
                Cumulative_Leaving_Probability = self.pdf.sum(axis=0)
                for Cohort in range(0, len(self.t)):
                    if Cumulative_Leaving_Probability[Cohort] != 1:
                        self.i[Cohort] = InitialStock[Cohort] / \
                            (1 - Cumulative_Leaving_Probability[Cohort])
                    else:
                        self.i[Cohort] = 0  # Not possible with given lifetime distribution
                ExitFlag = 1
                return self.i, ExitFlag
            else:
                ExitFlag = 3  # The length of t and InitialStock needs to be equal
                return None, ExitFlag
        else:
            ExitFlag = 2  # i already exists. Doing nothing
            return None, ExitFlag

    """
    Part 4: Stock driven model
    Given: total stock, lifetime dist.
    Default order of methods:
    1) determine inflow, outflow by cohort, and stock by cohort
    2) determine total outflow
    3) check mass balance.
    """

    def compute_stock_driven_model(self):
        """ With given total stock and lifetime distribution, the method builds the stock by cohort and the inflow.
        The extra parameter InitialStock is a vector that contains the age structure of the stock at time t0, and it covers as many historic cohorts as there are elements in it."""
        if self.s is not None:
            if self.lt is not None:
                self.s_c = np.zeros((len(self.t), len(self.t)))
                self.o_c = np.zeros((len(self.t), len(self.t)))
                self.i = np.zeros(len(self.t))
                # construct the pdf of a product of cohort tc leaving the stock in year t
                self.compute_outflow_pdf()
                # First year:
                self.i[0] = self.s[0]
                self.s_c[0, 0] = self.s[0]
                for m in range(1, len(self.t)):  # for all years m, starting in second year
                    for n in range(0, m):  # for all cohort n from first to last year
                        # 1) determine outflow and remaining stock:
                        self.o_c[m, n] = self.pdf[m, n] * self.i[n]  # outflow
                        # remaining stock
                        self.s_c[m, n] = (1 - self.pdf[0:m + 1, n].sum()) * self.i[n]
                    self.i[m] = self.s[m] - self.s_c[m, :].sum()  # mass balance
                    self.s_c[m, m] = self.i[m]  # add inflow to stock
                    ExitFlag = 1
                return self.s_c, self.o_c, self.i, ExitFlag
            else:
                ExitFlag = 2  # No lifetime distribution specified
                return None, None, ExitFlag
        else:
            ExitFlag = 3  # No stock specified
            return None, None, ExitFlag
