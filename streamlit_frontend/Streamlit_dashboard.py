import streamlit as st

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

import requests
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
import shap
st.set_option('deprecation.showPyplotGlobalUse', False)

#url1= "http://backend:80/prediction"
url1 = "http://fastapi_backend:80/prediction/"
url2 = "http://fastapi_backend:80/importance/"


def request_prediction(url, data):
    headers = {"Content-Type": "application/json"}    
    #data_json = json.dumps(data)
    data_json = [{'inputs': data}]    
    response = requests.post(headers=headers, url=url, json=data_json)    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()


def gauge(proba):
    greys15 = n_colors('rgb(55, 126, 184)','rgb(226, 26, 28)',  10, colortype='rgb')
    #fig, ax = plt.subplots()
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 0.5], 'y': [0, 1]},
    value = proba,
    mode = "gauge+number",
    title = {'text': "Risk of default"},
    gauge = {'axis': {'range': [None, 100], 'tickwidth': 2},
             'bar': {'color': "black", 'thickness': 0.15},
             
             'steps' : [
                 {'range': [10*i, 10+10*i], 'color': greys15[i]} for i in range(10)],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 52}}))
    
    st.plotly_chart(fig)
    
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
        

def main():
    
    st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #0c0080)
    }
   .sidebar .sidebar-content {
        background-color: #0c0080)
    }
    </style>
    """,
    unsafe_allow_html=True
)
    
    some_customers = pd.read_csv("some_customers.csv")
    
    
    #MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    
    st.header("Credit Default Risk Analysis")
    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    
    with row0_1:
        st.markdown("Credit default risk estimated from customer's information.")
        
    with row0_2:
        
        st.subheader('Streamlit App by Aleth Andre')
        st.markdown("Source code available in the [Openclassrooms_P7 GitHub Repository](https://github.com/mariealeth/Openclassrooms_P7.git)")
        
        
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.subheader("I have the customer's Id")
    t = st.sidebar.radio("", [True, False])
    st.sidebar.text('')
    st.sidebar.text('')
    if t:
        identity = st.sidebar.number_input('Enter Id', min_value=0, value=136116, step=1)
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    predict_btn = st.sidebar.button('Prédire')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    
    
    
    if predict_btn:
    
        row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((.2, 4., .1, 5., .2))
        row2_spacer1, row2_1, row2_spacer2 = st.columns((.5, 8, .5))
            
        if identity not in list(some_customers['SK_ID_CURR']):
            with row1_1:
                st.subheader('This customer Id is not is the Database.')
                st.markdown("Please check the number you've entered.")
                st.markdown("Or choose No in the lateral choice bar and then enter the informations requested on the client to make a prediction.")
                
        else:
            pred = request_prediction(url1, identity)
            result = pred['prediction'][0]
            
            with row1_1:
                gauge(result*100)
                
            with row1_2:
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                best_threshold = 0.5157894736842106
                if result > 0.6:
                    st.subheader('Credit will be refused') 
                    st.markdown(f"Based on credit history, the risk of default is assessed at {round((result)*100, 2)} on a scale of 0 to 1000.")
                elif result > 0.5157894736842106:
                    st.subheader('Credit might be refused') 
                    st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 1000.")
                elif result > 0.3:
                    st.subheader('Credit should be accepted') 
                    st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                else:
                    st.subheader('Credit will be accepted')
                    st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                    
            
            with row2_1:
                st.markdown('')
                st.markdown('')
                st.markdown('')
                st.markdown('')
                
                st.markdown('')
                st.markdown('')
                
            row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 8, .1))
                
            
            with row3_1:
                st.header("Most important contributions in the results")
                shapley = request_prediction(url2, identity)
                st.markdown('')
                st.markdown('')
                plot_features_importance(shapley['customer_data'], shapley['feature_importance'], 20)
                    
                st.markdown('')
                st.markdown('')
               
        
             
            row4_spacer1, row4_1, row4_spacer2, row4_2, row4_spacer3 = st.columns((.1, 2., .1, 5., .1))
            
            with row4_1:
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.header('Features')
                st.markdown('Choose a feature to get a definition and some data visualization.')
                st.text("")
                features_20 = list(pd.read_csv('features_20.csv')['feature'])
                features_choice = st.selectbox ("", features_20, key = 'what')
                
 
 
 
 

#
#

#                    st.markdown(f"Based on credit history, the risk of default is assessed at {round((pred['predictions'][0][1])*100, 2)} on a scale of 0 to 1000.")
            
            #with row1_2:
                #plot_features_importance(customer_data.iloc[:, 1: -1].values, pred['feature_importance'], 10)
                
if __name__ == '__main__':
   
    main()


