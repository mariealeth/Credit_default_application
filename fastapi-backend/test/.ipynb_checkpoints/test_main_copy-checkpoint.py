
import unittest
import pandas as pd
from functions import *

    
class TestSuppEmptyFeatures(unittest.TestCase):
    
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, np.nan, 4],
                                'B': [5, 6, np.nan, np.nan],
                                'C': [8, 9, 10, 11],
                                'D': [12, np.nan, np.nan, np.nan]})

    def test_sup_empty_features(self):
        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': [5, 6, np.nan, np.nan], 'C': [8, 9, 10, 11]})
        pd.util.testing.assert_frame_equal(supp_empty_features(self.df), expected_output)

        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'C': [8, 9, 10, 11]})
        pd.util.testing.assert_frame_equal(supp_empty_features(self.df, percent_mini=25), expected_output)

        expected_output = pd.DataFrame({'A': [1, 2, np.nan, 4], 'C': [8, 9, 10, 11], 'B': [5, 6, np.nan, np.nan], })
        pd.util.testing.assert_frame_equal(supp_empty_features(self.df, features_excluded=['B'],  percent_mini=25), expected_output)
        
        
class TestOneHotEncoder(unittest.TestCase):

    def test_one_hot_encoder(self):
        
        data = {'col1': ['A', 'B', 'C', 'A'], 'col2': ['X', 'X', 'Z', None], 'col3': [1, 2, 3, 4]}
        df = pd.DataFrame(data)
        encoded_df, new_columns = one_hot_encoder(df, nan_as_category=True)
        encoded_df1, new_columns1 = one_hot_encoder(df, nan_as_category=False)
               
        expected_encoded_cols = ['col1_A', 'col1_B', 'col1_C', 'col1_nan', 'col2_X', 'col2_Z', 'col2_nan']
        expected_encoded_cols1 = ['col1_A', 'col1_B', 'col1_C', 'col2_X', 'col2_Z']
        
        self.assertListEqual(expected_encoded_cols, new_columns)
        self.assertListEqual(expected_encoded_cols1, new_columns1)
        
        expected_values = [[1, 1, 0, 0, 0, 1, 0, 0],
                           [2, 0, 1, 0, 0, 1, 0, 0],
                           [3, 0, 0, 1, 0, 0, 1, 0],
                           [4, 1, 0, 0, 0, 0, 0, 1]]
        expected_values1 = [[1, 1, 0, 0, 1, 0],
                           [2, 0, 1, 0, 1, 0],
                           [3, 0, 0, 1, 0, 1],
                           [4, 1, 0, 0, 0, 0]]
        
        self.assertTrue((expected_values == encoded_df.values).all())
        self.assertTrue((expected_values1 == encoded_df1.values).all())

class TestApplicationTrainTest(unittest.TestCase):
    
    def test_returns_dataframe(self):
        result = application_train_test(num_rows=1000)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_no_null_values(self):
        result = application_train_test(num_rows=1000)
        self.assertFalse(result.isnull().values.any())
    
    def test_new_features_added(self):
        result = application_train_test(num_rows=1000)
        self.assertIn('DAYS_EMPLOYED_PERC', result.columns)
        self.assertIn('INCOME_CREDIT_PERC', result.columns)
        self.assertIn('INCOME_PER_PERSON', result.columns)
        self.assertIn('ANNUITY_INCOME_PERC', result.columns)
        self.assertIn('PAYMENT_RATE', result.columns) 
        
        
class TestApplicationTrainTest(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'CODE_GENDER': ['F', 'M', 'XNA'],
            'FLAG_OWN_CAR': ['N', 'Y', 'N'],
            'AMT_INCOME_TOTAL': [100000, 200000, 150000],
            'AMT_CREDIT': [500000, 600000, 700000],
            'CNT_FAM_MEMBERS': [2, 1, 3],
            'AMT_ANNUITY': [25000, 30000, 35000],
            'DAYS_BIRTH': [-15000, -20000, -25000],
            'DAYS_EMPLOYED': [-5000, -10000, 365243]
        })
        self.df.to_csv('application_train.csv')
        self.df.to_csv('application_test.csv')

    def test_XNA(self):
        df = application_train_test()
        self.assertEqual(len(df), 4)  # 4 rangs (les XNA sont supprimés)
        self.assertFalse('XNA' in df['CODE_GENDER'].values)  # XNA supprimés

    def test_day_employed(self):
        df = application_train_test()
        self.assertFalse(365243 in df['DAYS_EMPLOYED'].values)  

    def test_new_features(self):
        df = application_train_test(num_rows=None, nan_as_category=False)
        self.assertAlmostEqual(df['DAYS_EMPLOYED_PERC'][0], 0.333) 
        self.assertAlmostEqual(df['INCOME_CREDIT_PERC'][0], 0.2)  
        self.assertAlmostEqual(df['INCOME_PER_PERSON'][0], 50000)  
        self.assertAlmostEqual(df['ANNUITY_INCOME_PERC'][0], 0.25)  
        self.assertAlmostEqual(df['PAYMENT_RATE'][0], 0.05)  


        

        
        

        

        
        


    

