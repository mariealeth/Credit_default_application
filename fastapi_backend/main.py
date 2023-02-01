from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import pickle
import uvicorn
import lightgbm
import shap
import streamlit as st
from matplotlib import pyplot as plt

app = FastAPI()

#class Customer(BaseModel):
#    identity : int

with open("lightGBM_definitif.pkl", "rb") as f:
    lightGBM_definitif = pickle.load(f)
with open('explainer.pkl', 'rb') as explainer:
    explain = pickle.load(explainer)
features = pd.read_csv('features.csv')
expected_value = pd.read_csv('expected_value.csv')

class WaterfallData():
    def __init__ (self, shap_test, col, expected_value, data):
        self.values = shap_test[col].values
        self.base_values = expected_value
        self.data = data
        self.feature_names = shap_test.index
        self.display_data = shap_test.index
        
def plot_features_importance(customer_input, shap_values, nb_features):
    features = pd.read_csv("features.csv")['features'].values
    expected_value_mean = pd.read_csv("expected_value.csv")['0'].mean()
    features_importantes = pd.DataFrame(index=features, columns=['shap'], data=shap_values)
    features_importantes['abs'] = features_importantes['shap'].apply(lambda x: abs(x))
    features_importantes = features_importantes.sort_values('abs', ascending=False).iloc[:nb_features] 
    shap.plots.waterfall(WaterfallData(features_importantes, 'shap', expected_value_mean, customer_input), max_display=nb_features)
    st.pyplot(bbox_inches='tight',dpi=500, pad_inches=0)
    #plt.clf()    

@app.post("/prediction")
async def make_predictions(data: Request):
    json_data = await data.json()
    customer_id = json_data[0]['inputs'] 
    some_customers = pd.read_csv("some_customers.csv")
    customer_data = some_customers[some_customers['SK_ID_CURR']==customer_id]
    customer_data = customer_data.iloc[:, 1: -1].values
    
    predict1 = lightGBM_definitif.predict_proba(customer_data)[0][1]
    #
    return ({'prediction': [predict1]})
    
@app.post("/importance")
async def make_predictions(data: Request):
    json_data = await data.json()
    customer_id = json_data[0]['inputs'] 
    some_customers = pd.read_csv("some_customers.csv")
    customer_data = some_customers[some_customers['SK_ID_CURR']==customer_id]
    customer_data = customer_data.iloc[:, 1: -1].values
    
    X_input_transformed = pd.DataFrame(columns=features, data=lightGBM_definitif['scl'].transform(customer_data))
    shap_values = explain.shap_values(X_input_transformed)[0].tolist()
    
    return ({'feature_importance': shap_values, 'customer_data': customer_data[0].tolist()})
    
    





