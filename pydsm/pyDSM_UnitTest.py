# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:19:39 2014

@author: Georgios, Stefan
"""

import numpy as np
# import pylab
from pyDSM import DSM
import unittest

###############################################################################
"""My Input for fixed lifetime"""
Time_T_FixedLT = np.zeros((10,1))
Inflow_T_FixedLT = [1,2,3,4,5,6,7,8,9,10]
lifetime_FixedLT = {'Type': 'Fixed', 'Mean': np.array([5])}
#lifetime_FixedLT = {'Type': 'Fixed', 'Mean': np.array([5,5,5,5,5,5,5,5,5,5])}
lifetime_NormLT = {'Type': 'Normal', 'Mean': np.array([5]), 'StdDev' : np.array([1.5])}
###############################################################################
"""My Output for fixed lifetime"""
Outflow_T_FixedLT = np.array([0,0,0,0,0,1,2,3,4,5])

Outflow_TC_FixedLT = np.array([[0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0],
                               [0,0,0,0,0,0,0,0,0,0],
                               [1,0,0,0,0,0,0,0,0,0],
                               [0,2,0,0,0,0,0,0,0,0],
                               [0,0,3,0,0,0,0,0,0,0],
                               [0,0,0,4,0,0,0,0,0,0],
                               [0,0,0,0,5,0,0,0,0,0]])

Stock_T_FixedLT = np.array([1,3,6,10,15,20,25,30,35,40])

StockChange_T_FixedLT = np.array([1,2,3,4,5,5,5,5,5,5])

Stock_TC_FixedLT = np.array([[1,0,0,0,0,0,0,0,0,0],
                             [1,2,0,0,0,0,0,0,0,0],
                             [1,2,3,0,0,0,0,0,0,0],
                             [1,2,3,4,0,0,0,0,0,0],
                             [1,2,3,4,5,0,0,0,0,0],
                             [0,2,3,4,5,6,0,0,0,0],
                             [0,0,3,4,5,6,7,0,0,0],
                             [0,0,0,4,5,6,7,8,0,0],
                             [0,0,0,0,5,6,7,8,9,0],
                             [0,0,0,0,0,6,7,8,9,10]])

Bal = np.array([0,0,0,0,0,0,0,0,0,0]) 
"""My Output for normally distributed lifetime"""                       
                             
Stock_TC_NormLT = np.array([[1,	0,	0,	0,	0,	0,	0, 0,	0,	0], # computed with Excel and taken from there
                            [0.992402676,	2,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0.956408698,	1.984805352,	3,	0,	0,	0,	0,	0,	0,	0],
                            [0.847068649,	1.912817397,    2.977208028,	4,	0,	0,	0,	0,	0,	0],
                            [0.634103312,	1.694137297,	2.869226095,	3.969610704,	5,	0,	0,	0,	0,	0],
                            [0.368141791,	1.268206623,	2.541205946,	3.825634793,	4.96201338,	6,	0,	0,	0,	0],
                            [0.155176454,	0.736283582,	1.902309935,	3.388274594,	4.782043492,	5.954416056,	7,	0,	0,	0],
                            [0.045836404,	0.310352908,	1.104425374,	2.536413246,	4.235343243,	5.73845219,	6.946818732,	8,	0,	0],
                            [0.009842427,	0.091672809,	0.465529363,	1.472567165,	3.170516558,	5.082411891,	6.694860888,	7.939221408,	9,	0],
                            [0.002245103,	0.019684854,	0.137509213,	0.620705817,	1.840708956,	3.804619869,	5.92948054,	7.651269586,	8.931624084,	10]])
                                                         
Stock_T_NormLT = np.array([1,2.992402676,5.94121405,9.737094073,14.16707741,18.96520253,23.91850411,28.9176421,33.92662251,38.93784802])

Outflow_T_NormLT = np.array([0,0.007597324,0.051188626,0.204119977,0.570016666,1.201874874,2.04669842,3.000862016,3.991019589,4.988774486])

Outflow_TC_NormLT = np.array([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                              [0.007597324,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                              [0.035993978,	0.015194648,	0,	0,	0,	0,	0,	0,	0,	0],
                              [0.10934005,	    0.071987955,	0.022791972,	0,	0,	0,	0,	0,	0,	0],
                              [0.212965337,	0.2186801,	    0.107981933,	0.030389296,	0,	0,	0,	0,	0,	0],
                              [0.26596152,	    0.425930674,	0.328020149,	0.143975911,	0.03798662,	0,	0,	0,	0,	0],
                              [0.212965337,	0.531923041,	0.638896011,	0.437360199,	0.179969888,	0.045583944,	0,	0,	0,	0],
                              [0.10934005,	    0.425930674,	0.797884561,	0.851861348,	0.546700249,	0.215963866,	0.053181268,	0,	0,	0],
                              [0.035993978,	0.2186801,	    0.638896011,	1.063846081,	1.064826685,	0.656040299,	0.251957844,	0.060778592,	0,	0],
                              [0.007597324,	0.071987955,	0.328020149,	0.851861348,	1.329807601,	1.277792022,	0.765380348,	0.287951821,	0.068375916,	0]])

StockChange_T_NormLT = np.array([1,1.992402676,2.948811374,3.795880023,4.429983334,4.798125126,4.95330158,4.999137984,5.008980411,5.011225514])
                        
""" Test case with fixed lifetime for initial stock"""                        
Time_T_FixedLT_X   = np.arange(1,9,1)
lifetime_FixedLT_X = {'Type': 'Fixed', 'Mean': np.array([5])}
InitialStock_X = np.array([0,0,0,7,5,4,3,2])
Inflow_X = np.array([0,0,0,7,5,4,3,2])

Time_T_FixedLT_XX   = np.arange(1,11,1)
lifetime_NormLT_X = {'Type': 'Normal', 'Mean': np.array([5]), 'StdDev' : np.array([1.5])}                      
InitialStock_XX   = np.array([0.01,	0.01,	0.08,	0.2,	0.2,	2,	2,	3,	4,	7.50])
Inflow_XX         = np.array([4.454139122,	1.016009592,	1.745337597,	1.288855329,	0.543268938,	3.154060172,	2.361083725,	3.136734333,	4.030621941,	7.5])

                            
###############################################################################
"""Create Dynamic Stock Models and hand over the pre-defined values."""
# For fixed LT
myDSM  = DSM(t = Time_T_FixedLT, i = Inflow_T_FixedLT, lt = lifetime_FixedLT)
myDSM2 = DSM(t = Time_T_FixedLT, s = Stock_T_FixedLT, lt = lifetime_FixedLT)
myDSMx = DSM(t = Time_T_FixedLT_X, lt = lifetime_FixedLT_X)
TestInflow_X = myDSMx.compute_i_from_s(InitialStock = InitialStock_X)[0]
myDSMxy= DSM(t = Time_T_FixedLT, i = TestInflow_X, lt = lifetime_FixedLT)

# For normally distributed Lt
myDSM3 = DSM(t = Time_T_FixedLT, i = Inflow_T_FixedLT, lt = lifetime_NormLT)
myDSM4 = DSM(t = Time_T_FixedLT, s = Stock_T_NormLT, lt = lifetime_NormLT)
myDSMX = DSM(t = Time_T_FixedLT_XX, lt = lifetime_NormLT_X)
TestInflow_XX = myDSMX.compute_i_from_s(InitialStock = InitialStock_XX)[0]
myDSMXY= DSM(t = Time_T_FixedLT_XX, i = TestInflow_XX, lt = lifetime_NormLT)
# Compute full stock model in correct order
###############################################################################
"""Unit Test Class"""
class KnownResults(unittest.TestCase):    
        
    def test_inflow_driven_model_fixedLifetime(self):
        """Test Inflow Driven Model with Fixed product lifetime."""
        np.testing.assert_array_equal(myDSM.compute_s_c_inflow_driven()[0],Stock_TC_FixedLT)
        np.testing.assert_array_equal(myDSM.compute_stock_total()[0],Stock_T_FixedLT)   
        np.testing.assert_array_equal(myDSM.compute_o_c_from_s_c()[0],Outflow_TC_FixedLT)        
        np.testing.assert_array_equal(myDSM.compute_outflow_total()[0],Outflow_T_FixedLT)                
        np.testing.assert_array_equal(myDSM.compute_stock_change()[0],StockChange_T_FixedLT)
        np.testing.assert_array_equal(myDSM.check_stock_balance()[0],Bal.transpose())
        
    def test_stock_driven_model_fixedLifetime(self):
        """Test Stock Driven Model with Fixed product lifetime."""
        np.testing.assert_array_equal(myDSM2.compute_stock_driven_model()[0],Stock_TC_FixedLT)
        np.testing.assert_array_equal(myDSM2.compute_stock_driven_model()[1],Outflow_TC_FixedLT)        
        np.testing.assert_array_equal(myDSM2.compute_stock_driven_model()[2],Inflow_T_FixedLT)        
        np.testing.assert_array_equal(myDSM2.compute_outflow_total()[0],Outflow_T_FixedLT)                
        np.testing.assert_array_equal(myDSM2.compute_stock_change()[0],StockChange_T_FixedLT)
        np.testing.assert_array_equal(myDSM2.check_stock_balance()[0],Bal.transpose())        
        
    def test_inflow_driven_model_normallyDistLifetime(self):
        """Test Inflow Driven Model with normally distributed product lifetime."""        
        np.testing.assert_array_almost_equal(myDSM3.compute_s_c_inflow_driven()[0],Stock_TC_NormLT,9)
        np.testing.assert_array_almost_equal(myDSM3.compute_stock_total()[0],Stock_T_NormLT,8)
        np.testing.assert_array_almost_equal(myDSM3.compute_o_c_from_s_c()[0],Outflow_TC_NormLT,9) 
        np.testing.assert_array_almost_equal(myDSM3.compute_outflow_total()[0],Outflow_T_NormLT,9)                
        np.testing.assert_array_almost_equal(myDSM3.compute_stock_change()[0],StockChange_T_NormLT,9)
        np.testing.assert_array_almost_equal(myDSM3.check_stock_balance()[0],Bal.transpose(),12)

    def test_stock_driven_model_normallyDistLifetime(self):
        """Test Stock Driven Model with normally distributed product lifetime."""        
        np.testing.assert_array_almost_equal(myDSM4.compute_stock_driven_model()[0],Stock_TC_NormLT,8)
        np.testing.assert_array_almost_equal(myDSM4.compute_stock_driven_model()[1],Outflow_TC_NormLT,9)
        np.testing.assert_array_almost_equal(myDSM4.compute_stock_driven_model()[2],Inflow_T_FixedLT,8)
        np.testing.assert_array_almost_equal(myDSM4.compute_outflow_total()[0],Outflow_T_NormLT,9)                
        np.testing.assert_array_almost_equal(myDSM4.compute_stock_change()[0],StockChange_T_NormLT,8)
        np.testing.assert_array_almost_equal(myDSM4.check_stock_balance()[0],Bal.transpose(),12)
        
    def test_inflow_from_stock_fixedLifetime(self):
        """Test computation of inflow from stock with Fixed product lifetime."""        
        np.testing.assert_array_equal(TestInflow_X,Inflow_X)                
        np.testing.assert_array_equal(myDSMxy.compute_s_c_inflow_driven()[0][-1,:],InitialStock_X)                

    def test_inflow_from_stock_normallyDistLifetime(self):
        """Test computation of inflow from stock with normally distributed product lifetime."""                
        np.testing.assert_array_almost_equal(TestInflow_XX,Inflow_XX,9)                
        np.testing.assert_array_almost_equal(myDSMXY.compute_s_c_inflow_driven()[0][-1,:],InitialStock_XX,9)
        
    if __name__ == '__main__':
        unittest.main()
