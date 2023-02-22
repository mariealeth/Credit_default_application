from main import app
import unittest
from fastapi.testclient import TestClient
import pandas as pd
import pickle
import lightgbm
from function import *

with open("lightGBM_definitif.pkl", "rb") as f:
    lightGBM_definitif = pickle.load(f)

    
class TestSuppEmptyFeatures(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                                'B': [5, 6, 7, np.nan],
                                'C': [8, 9, 10, 11],
                                'D': [12, np.nan, np.nan, np.nan]})

    def test_sup_empty_features(self):
        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'C': [8, 9, 10, 11]})
        self.assertEqual(supp_empty_features(self.df), expected_output)

        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'C': [8, 9, 10, 11], 'D': [12, np.nan, np.nan, np.nan]})
        self.assertEqual(supp_empty_features(self.df, percent_mini=25), expected_output)

        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'D': [12, np.nan, np.nan, np.nan]})
        self.assertEqual(supp_empty_features(self.df, features_excluded=['B', 'C']), expected_output)
    

class MyTestCase(unittest.TestCase):
    
    def test_proba1(self):
        some_customers = pd.read_csv("some_customers.csv")
        customer_data = some_customers[some_customers['SK_ID_CURR']==100001]
        customer_data = customer_data.iloc[:, 1: -1].values
        result = lightGBM_definitif.predict_proba(customer_data)[0][1]
        self.assertGreater(0.5, result) 
        
    def test_proba2(self):
        some_customers = pd.read_csv("some_customers.csv")
        customer_data = some_customers[some_customers['SK_ID_CURR']==101780]
        customer_data = customer_data.iloc[:, 1: -1].values
        result = lightGBM_definitif.predict_proba(customer_data)[0][1]
        self.assertGreater(result, 0.6) 

        
