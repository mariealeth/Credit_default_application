import streamlit as st

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

import requests
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
import seaborn as sns
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors
import shap
#from azure.storage.blob import BlobServiceClient
st.set_option('deprecation.showPyplotGlobalUse', False)


url1 = "http://fastapi-backend:80/prediction/"
url2 = "http://fastapi-backend:80/importance/"
url3 = "http://fastapi-backend:80/estimation/"


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
    greys15 = n_colors('rgb(19, 148, 249)','rgb(225, 47, 123)', 10, colortype='rgb')
    fig = go.Figure(go.Indicator(
    domain = {'x': [0, 0.5], 'y': [0, 1]},
    value = proba,
    mode = "gauge+number",
    title = {'text': "Risk of default"},
    gauge = {'axis': {'range': [None, 100], 'tickwidth': 2},
             'bar': {'color': "black", 'thickness': 0.50},
             
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
    st.pyplot(bbox_inches='tight',dpi=100, pad_inches=0)  

def plot_feature(datas, feature, n_bins, value):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    data = datas[datas[feature].notna()]
    heights0, bins0 = np.histogram(data[data['target']==0][feature], density=True, bins=n_bins)
    heights, bins = np.histogram(data[data['target']==1][feature], density=True, bins=bins0)
    heights *= -1
    bin_width = np.diff(bins)[0]
    bin_pos =( bins[:-1] + bin_width / 2)
    sns.histplot(ax=ax, x=data[data['target']==0][feature], stat="density", bins=n_bins, edgecolor='black', color='#2f78e8')
    plt.bar(bin_pos, heights, width=bin_width, edgecolor='black', color='#f6245a')
    if value!=np.NaN:
        ax.text(x=value+bin_width/2, y=min(heights)*1.2, s=f'{feature} = {round(value, 6)}', weight='bold')
        plt.vlines(value, max(heights0)*1.2, min(heights)*1.2, color='black', label='customer', linewidths=3)
    from matplotlib.lines import Line2D
    legend_elements = [Patch(facecolor='#2f78e8', edgecolor='#2f78e8',label='0'),
                      Patch(facecolor='#f6245a', edgecolor='#f6245a', label='1')]
    ax.legend(handles=legend_elements, loc='lower left')
    ax.set_xlabel('')
    st.pyplot(fig)
    
def user_input_features():
    input_features = {}
    
    SOURCE_1 = st.sidebar.slider(key='EXT_SOURCE_1', label='Score 1', min_value=0, max_value=100, value=51, help='Normalized score from external data source 1')
    input_features["EXT_SOURCE_1"] =  SOURCE_1/100
    
    input_features["EXT_SOURCE_2"] = st.sidebar.slider(key='EXT_SOURCE_2', label='Score 2', min_value=0, max_value=100, value=57, help='Normalized score from external data source 2')/100
    
    input_features["EXT_SOURCE_3"]= st.sidebar.slider(key='EXT_SOURCE_3', label='Score 3', min_value=0, max_value=100, value=53, help='Normalized score from external data source 3')/100
    
    
    income = st.sidebar.number_input(label='Income ($)', min_value=0, max_value=10000000000, key='income', help='Total income', value=147000)
    loan = st.sidebar.number_input(label='Loan ($)', min_value=1, max_value=10*income, key='loan', help='Loan amount', value=514000)
    input_features["INCOME_CREDIT_PERC"] = income/loan
    
    car = st.sidebar.selectbox(key='Car_loan', label='Number of car loans', options=[0, 1, 2, 3, 4, 5], help='number of Credit Bureau car loan')
    other = st.sidebar.selectbox(key='Other_loan', label='Number of other loans', options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='number of other Credit Bureau credits')
    if (car+other)==0:
        input_features["BURO_CREDIT_TYPE_Car loan_MEAN"] = 0
    else:
        input_features["BURO_CREDIT_TYPE_Car loan_MEAN"] = car/(car+other)
    
    input_features["CLOSED_AMT_CREDIT_SUM_SUM"] = st.sidebar.number_input(label='Sum of closed credits', min_value=0, max_value=1000000000, key='CLOSED_AMT_CREDIT_SUM_SUM', help='Sum of all amounts of all closed credits', value=435000)
    
    refused = st.sidebar.selectbox(key='refused', label='Number of refused applications SCOFR', options=[0, 1, 2, 3, 4, 5], help='Number of applications refused with the code SCOFR')
    submissions = st.sidebar.selectbox(key='applications', label='Number of other applications', options=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help='Number of applications for a credit either accepted or refused but not with the code SCOFR')
    if (submissions)==0:
        input_features["PREV_CODE_REJECT_REASON_SCOFR_MEAN"] = 0
    else:
        input_features["PREV_CODE_REJECT_REASON_SCOFR_MEAN"] = refused/submissions
    
    input_features["APPROVED_AMT_ANNUITY_MAX"] = st.sidebar.number_input(key='APPROVED_AMT_ANNUITY_MAX', label='Max annuity of previous applications ($)', min_value=0, max_value=400000, help='Annuity max of previous approuved applications', value=16300)
    
    a = st.sidebar.slider(key='CC_CNT_DRAWINGS_ATM_CURRENT_MEAN', label='Average number of drawings per month', min_value=0, max_value=10, help='Average number of drawings at ATM during one month', value=1)
    input_features["CC_CNT_DRAWINGS_ATM_CURRENT_MEAN"] = a
    
    input_features["CC_CNT_DRAWINGS_CURRENT_MAX"] = st.sidebar.slider(key='CC_CNT_DRAWINGS_CURRENT_MAX', label='Number max of drawings per month', min_value=a, max_value=20, help='Number max of drawings during one month', value=3)

    return [input_features]




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
    unsafe_allow_html=True)
    
    some_customers = pd.read_csv("some_customers.csv")
    data_20 = pd.read_csv('data_20.csv')
    some_customers2 = pd.read_csv('some_customers2.csv')
    
    st.title("Credit Default Risk Analysis")
    row0_spacer1, row0_1, row0_spacer2= st.columns((.1, 8, .1))
    
    with row0_1:
        st.header("Credit default risk estimated from customer's information.")
        st.subheader('Streamlit App by Aleth Andre')
        st.caption("Source code available in the [Openclassrooms_P7 GitHub Repository](https://github.com/mariealeth/Openclassrooms_P7.git)")

    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')
    number = st.sidebar.radio(label="I have the customer's Id", options=['Select', True, False], key="number")
    if number==True:
        identity = st.sidebar.number_input('Enter Id', min_value=0, value=136116, step=1)
        st.sidebar.text('')
        predict_btn = st.sidebar.radio(label="Choose purpose", options=['Select', 'Predict', 'Overview'], key="lightgbm")
        if predict_btn=='Predict':
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
                        st.markdown(f"Based on credit history, the risk of default is assessed at {round((result)*100, 2)} on a scale of 0 to 100.")
                    elif result > 0.5157894736842106:
                        st.subheader('Credit might be refused') 
                        st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                    elif result > 0.3:
                        st.subheader('Credit should be accepted') 
                        st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                    else:
                        st.subheader('Credit will be accepted')
                        st.markdown(f"Based on credit history, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")

# Features importance
                        
                st.sidebar.text('')
                s = st.sidebar.radio(label="Display the most important contributions in the results", options=[False, True], key="shapley")
                row3_spacer1, row3_1, row3_spacer2,  row3_3, row3_spacer4 = st.columns((.1, 4.,1., 3., .1))
                if s:
                    with row3_1:
                        st.header("Most important contributions in the results")
                        st.markdown("The bottom of a waterfall plot starts as the expected value of the risk, and then each row shows how the positive (red) or negative (blue) contribution of each feature moves the value from the expected risk over the background dataset to the risk for this prediction.")
                        shapley = request_prediction(url2, identity)
                        plot_features_importance(shapley['customer_data'], shapley['feature_importance'], 20) 
                    with row3_3:
                        features_20_df = pd.read_csv('features_20.csv')
                        features_20 = list(features_20_df['feature'])
                        features_20 = ['choose a feature']+features_20
                        features_choice = st.sidebar.selectbox ("Feature visualization.", features_20, key = 'what')
                        if features_choice!='choose a feature':
                            st.header(features_choice)
                            st.markdown(f"{features_20_df[features_20_df['feature']==features_choice]['definitions'].values[0]}")
                            if some_customers2[some_customers2['SK_ID_CURR']==identity][features_choice].values[0]==np.NaN:
                                st.markdown("Data is not available for this customer")
                            else:
                                value_custom = some_customers2[some_customers2['SK_ID_CURR']==identity][features_choice].values[0]
                                plot_feature(data_20, features_choice, 20, value_custom)
                                st.markdown(f"Distribution of customers for {features_choice}, based on their default status (red customers 1 defaulted, blue customers 0 did not defaulted)")
                            
        if predict_btn=='Overview':
            row4_1, row4_spacer2, row4_2= st.columns((3, .2, 6.))
            row5_1, row5_spacer2, row5_2= st.columns((5., .2, 5.))
            st.markdown('')
            st.markdown('')
            st.markdown('')
                
            with row4_1:
                if identity not in list(some_customers['SK_ID_CURR']):
                    st.subheader('This customer Id is not is the Database.')
                    st.markdown("Please check the number you've entered.")
                    st.markdown("Or choose No in the lateral choice bar and then enter the informations requested on the client to make a prediction.")     
                else:
                    st.header('')
                    st.header('')
                    
                    if some_customers2[some_customers2['SK_ID_CURR']==identity]['DAYS_BIRTH'].values[0]==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        age_custom = int(some_customers2[some_customers2['SK_ID_CURR']==identity]['DAYS_BIRTH'].values[0]*(-1)/365)
                        st.subheader(f"{age_custom} year's old")
                        st.markdown('Age at the time of application')
                    if some_customers2[some_customers2['SK_ID_CURR']==identity]['CODE_GENDER'].values[0]==np.NaN: 
                        st.markdown("Data is not available for this customer")
                    else:
                        Gender_custom = some_customers[some_customers['SK_ID_CURR']==identity]['CODE_GENDER'].values[0]
                        if Gender_custom==1:
                            st.subheader(f"Gender : man")
                        if Gender_custom==0:
                            st.subheader(f"Gender: woman")
                    if some_customers2[some_customers2['SK_ID_CURR']==identity]['DAYS_EMPLOYED'].values[0]==np.NaN: 
                        st.markdown("Data is not available for this customer")
                    else:
                        employment_custom = int(round(some_customers[some_customers['SK_ID_CURR']==identity]['DAYS_EMPLOYED'].values[0]/(-30), 0))
                        st.subheader(f"Current employment: {employment_custom} months")
                        st.markdown('How long ago did it start')
                    
                with row4_2:
                    st.header('')
                    st.header("Normalized scores from external data source")
                    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                    
                    data_20_EXT_SOURCE_1 = data_20[data_20["EXT_SOURCE_1"].notna()]
                    EXT_SOURCE_1_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['EXT_SOURCE_1'].values[0]
                    sns.violinplot(ax=ax[0], y=data_20_EXT_SOURCE_1["EXT_SOURCE_1"], color='#86b2a0')
                    ax[0].set_xlabel('EXT_SOURCE_1', weight='bold')
                    ax[0].set_ylabel('')
                    ax[0].set_yticks([], minor=False)
                    ax[0].set_yticklabels('', fontdict=None, minor=False)
                    if EXT_SOURCE_1_custom==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        ax[0].axhline(EXT_SOURCE_1_custom, ls='--', color='#9b1408')
                        ax[0].text(x=-0.2, y=EXT_SOURCE_1_custom*1.02, s=f"S1 = {round(EXT_SOURCE_1_custom, 4)}", weight='bold', color='#9b1408' , fontsize=14)
                        
                        
                    data_20_EXT_SOURCE_2 = data_20[data_20["EXT_SOURCE_2"].notna()]
                    EXT_SOURCE_2_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['EXT_SOURCE_2'].values[0]
                    sns.violinplot(ax=ax[1], y=data_20_EXT_SOURCE_2["EXT_SOURCE_2"], color='#86b2a0')
                    ax[1].set_xlabel('EXT_SOURCE_2', weight='bold')
                    ax[1].set_ylabel('')
                    ax[1].set_yticks([], minor=False)
                    ax[1].set_yticklabels('', fontdict=None, minor=False)
                    if EXT_SOURCE_2_custom==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        ax[1].axhline(EXT_SOURCE_2_custom, ls='--', color='#9b1408')
                        ax[1].text(x=-0.2, y=EXT_SOURCE_2_custom*1.02, s=f"S 2 ={round(EXT_SOURCE_2_custom, 4)}", weight='bold', color='#9b1408', fontsize=16)
                        
                    data_20_EXT_SOURCE_3 = data_20[data_20["EXT_SOURCE_3"].notna()]
                    EXT_SOURCE_3_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['EXT_SOURCE_3'].values[0]
                    sns.violinplot(ax=ax[2], y=data_20_EXT_SOURCE_3["EXT_SOURCE_3"], color='#86b2a0')
                    ax[2].set_xlabel('EXT_SOURCE_3', weight='bold')
                    ax[2].set_ylabel('')
                    ax[2].set_yticks([], minor=False)
                    ax[2].set_yticklabels('', fontdict=None, minor=False)
                    if EXT_SOURCE_3_custom==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        ax[2].axhline(EXT_SOURCE_3_custom, ls='--', color='#9b1408')
                        ax[2].text(x=-0.2, y=EXT_SOURCE_3_custom*1.02, s=f"S 3 = {round(EXT_SOURCE_3_custom, 4)}", weight='bold', color='#9b1408', fontsize=18)
                    
                    st.pyplot(fig)
                    
                st.markdown('')
                st.markdown('')
                st.markdown('')
                
                PAYMENT_RATE_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['PAYMENT_RATE'].values[0]
                PREV_APP_CREDIT_PERC_MIN_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['PREV_APP_CREDIT_PERC_MIN'].values[0]
                LOAN_custom =  some_customers2[some_customers2['SK_ID_CURR']==identity]['AMT_CREDIT'].values[0]
                ANNUITY_custom = some_customers2[some_customers2['SK_ID_CURR']==identity]['AMT_ANNUITY'].values[0]
                PAYMENT_RATE_mean = data_20['PAYMENT_RATE'].mean()
                PREV_APP_CREDIT_PERC_MIN_mean = data_20['PREV_APP_CREDIT_PERC_MIN'].mean()
                LOAN_mean = data_20['AMT_CREDIT'].mean()
                        
                with row5_1:
                    if LOAN_custom==np.NaN or PREV_APP_CREDIT_PERC_MIN_custom==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        st.header('')
                        st.header(f"Credit amount: {LOAN_custom} $")
                        st.subheader('Loan compared to precedent application')
                        st.header('')
                        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
                        ax[0].pie(x=[PREV_APP_CREDIT_PERC_MIN_custom*LOAN_custom, LOAN_custom], 
                                  explode=None, 
                                  labels=['Previous application', 'Loan'], 
                                  labeldistance=0.4, 
                                  colors=[ '#95a8cd', '#6697f9'])
                        ax[0].set_title('Client', weight='bold')
                        ax[1].pie(x=[PREV_APP_CREDIT_PERC_MIN_mean*LOAN_mean, LOAN_mean], 
                                  explode=None, 
                                  labels=['Previous application', 'Loan'], 
                                  labeldistance=0.4,
                                  colors=[ '#86b2a0', '#6fe2b3'])
                        ax[1].set_title('Mean', weight='bold')
                        st.pyplot(fig)
                    
                with row5_2:
                    if ANNUITY_custom==np.NaN:
                        st.markdown("Data is not available for this customer")
                    else:
                        st.header('')
                        st.header(f"Annuity: {ANNUITY_custom} $")
                        st.subheader('Payment rate = Loan annuity / credit amount of the loan')
                        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
                        data_20_PAYMENT_RATE = data_20[data_20["PAYMENT_RATE"].notna()]
                        sns.histplot(ax=ax, data=data_20_PAYMENT_RATE, x='PAYMENT_RATE', color='#86b2a0')
                        #sns.kdeplot(data_20['PREV_APP_CREDIT_PERC_MIN'], shade=True, bw=0.5, color="#6fe2b3")
                        ax.vlines(PAYMENT_RATE_custom, 0, 50, color='#9b1408', linewidths=3)
                        ax.text(x=PAYMENT_RATE_custom, y=-300, s=f"Client's payment rate = {round(PAYMENT_RATE_custom, 3)}", weight='bold', color='#9b1408')
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.set_yticks([], minor=False)
                        ax.set_yticklabels('', fontdict=None, minor=False)
                    
                        st.pyplot(fig)
                    
                                
    if number==False:
        st.sidebar.header('User Input features')
        customer_data = user_input_features()
        st.sidebar.text("")
        st.sidebar.text("")
        predict_btn2 = st.sidebar.button(label='Predict', key='lr')
        
        if predict_btn2:
            row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((.2, 4., .1, 5., .2))
            row2_spacer1, row2_1, row2_spacer2 = st.columns((.5, 8, .5))
                
            pred = request_prediction(url3, customer_data)
            result = pred['estimation'][0]
            
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
                best_threshold = 0.5131578947368421
                if result > 0.5131578947368421:
                    st.subheader('Credit should be refused') 
                    st.markdown(f"Based on given information, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                else:
                    st.subheader('Credit should be accepted')
                    st.markdown(f"Based on given information, the risk of default is assessed at {round(result*100, 2)} on a scale of 0 to 100.")
                    
                    
         
                
if __name__ == '__main__':
       
    main()
    
    
