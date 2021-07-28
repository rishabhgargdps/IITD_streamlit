#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import io
import string
import time
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import joblib as joblib
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

st.title('RUL Predictor')

st.write("""
## This app uses two different datasets
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Unnormalized RMS Prediction','RMS Prediction', 'Slope Prediction', 'Final prediction !')
)

st.write(dataset_name, "Dataset")

if dataset_name == 'Final prediction !':
    regression_file = st.file_uploader('Please upload RMS regressor model', type=['pkl'])
    classifier_file = st.file_uploader('Please upload Machine state classifier', type=['pkl'])
    regressor_model = joblib.load(regression_file)
    classifier_model = joblib.load(classifier_file)
    
    raw_data_file = st.file_uploader('Please upload the dataset', type=['csv','xls','xlsx'])
    
    def read_preprocess(file):
        df = pd.read_excel(file)
        pd.set_option('display.max_rows', 15)
        df = df.iloc[14:70, :]
        df.drop(df.columns[[0, 4, 8, 9, 10, 11, 12]], axis=1, inplace = True)
        df = df[df.applymap(np.isreal).all(1)]
        df.drop(df.loc[:, 'Unnamed: 13':'Unnamed: 15'].columns, axis = 1, inplace = True)
        df.dropna(subset = ['RMS'], inplace = True)
        return df
    
    df = read_preprocess(raw_data_file)
    X_test = df.iloc[:, 0:5]
    RMS_prediction = regressor_model.predict(X_test)
    RMS_df = pd.DataFrame(RMS_prediction, columns = ['RMS'])
    final_prediction = classifier_model.predict(RMS_df)
    st.write("Returning the predictions...")
    final_df = pd.DataFrame(final_prediction, columns = ['Status'])
    st.dataframe(final_df)
    

elif dataset_name == 'Unnormalized RMS Prediction':
    def read_preprocess(file):
        df = pd.read_excel(file)
        pd.set_option('display.max_rows', 15)
        df = df.iloc[14:70, :]
        df.drop(df.columns[[0, 4, 8, 9, 10, 11, 12]], axis=1, inplace = True)
        df = df[df.applymap(np.isreal).all(1)]
        df.drop(df.loc[:, 'Unnamed: 13':'Unnamed: 15'].columns, axis = 1, inplace = True)
        df.dropna(subset = ['RMS'], inplace = True)
        return df

    uploaded_file = st.file_uploader("Please upload the datafile",type=['csv','xls', 'xlsx'])

    time_counter = 0

    while uploaded_file is None:
        time.sleep(1)
        time_counter += 3
        if time_counter > 20:break

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head(10))
    else:
        raise ValueError("This isn't a file!")

    df = read_preprocess(uploaded_file)
    st.dataframe(df)

    reg_name = st.sidebar.selectbox(
        'Select regressor',
        ('Linear regressor', 'SVR')
    )
    
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVR':
            degree = st.sidebar.slider('degree', 1, 10)
            params['degree'] = degree
        return params

    params = add_parameter_ui(reg_name)
    
    def get_regressor(clf_name, params):
        clf = None
        if clf_name == 'SVR':
            clf = SVR(kernel = 'poly', degree = params['degree'])
        elif clf_name == 'Linear regressor':
            clf = LinearRegression()
        return clf

    regressor = get_regressor(reg_name, params)

    #### REGRESSION ####
    
    st.write("Training the model...")
    X = df.iloc[:, 0:5]
    y = df["RMS"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    regressor.fit(X_train, y_train)
    
    # predictions
    st.write("Returning the predictions...")
    y_predict = regressor.predict(X_test)

    st.write(y_predict)

    #### PLOT DATASET ####
    st.write("Plotting the results...")
    x_array = list(range(1, y_predict.shape[0]+1))
    fig = plt.figure()
    plt.scatter(x_array, y_predict, label = "Predicted")
    plt.scatter(x_array, y_test, label = "Actual values")

    plt.xlabel('Experiment Number')
    plt.ylabel('RMS')
    plt.legend(loc = 2)

    #plt.show()
    st.pyplot(fig)
    
    #### PERFORMANCE ANALYSIS ####
    st.write("Accuracy metrics...")
    st.write("RMSE")
    RMS_error = mean_squared_error(y_test, y_predict)
    st.write(RMS_error)
    st.write("Mean Absolute Error")
    MA_error = mean_absolute_error(y_test, y_predict)
    st.write(MA_error)
    st.write("R2 Score")
    r2 = r2_score(y_test, y_predict)
    st.write(r2)
    st.write("Adjusted R2 Score")
    adj_r2_score = 1 - ((1-r2)*(61-1)/(61-5-1))
    st.write(adj_r2_score)

elif dataset_name == 'RMS Prediction':
    def read_preprocess(file):
        df = pd.read_excel(file)
        pd.set_option('display.max_rows', 15)
        df = df.iloc[14:70, :]
        df.drop(df.columns[[0, 4, 7, 8, 9, 10, 11, 12]], axis=1, inplace = True)
        df.drop(df.loc[:, 'Unnamed: 14':'Unnamed: 15'].columns, axis = 1, inplace = True)
        df.rename(columns={'Unnamed: 13': 'RMS(g)'}, inplace = True)
        df.dropna(subset = ['RMS(g)'], inplace = True)
        return df

    uploaded_file = st.file_uploader("Please upload the datafile",type=['csv','xls', 'xlsx'])

    time_counter = 0

    while uploaded_file is None:
        time.sleep(1)
        time_counter += 3
        if time_counter > 20:break

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head(10))
    else:
        raise ValueError("This isn't a file!")

    df = read_preprocess(uploaded_file)

    reg_name = st.sidebar.selectbox(
        'Select regressor',
        ('Linear regressor', 'SVR')
    )
    
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVR':
            degree = st.sidebar.slider('degree', 1, 10)
            params['degree'] = degree
        return params

    params = add_parameter_ui(reg_name)
    
    def get_regressor(clf_name, params):
        clf = None
        if clf_name == 'SVR':
            clf = SVR(kernel = 'poly', degree = params['degree'])
        elif clf_name == 'Linear regressor':
            clf = LinearRegression()
        return clf

    regressor = get_regressor(reg_name, params)

    #### REGRESSION ####
    
    st.write("Training the model...")
    X = df.iloc[:, 0:5]
    y = df["RMS(g)"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    regressor.fit(X_train, y_train)
    
    # predictions
    st.write("Returning the predictions...")
    y_predict = regressor.predict(X_test)

    st.write(y_predict)

    #### PLOT DATASET ####
    st.write("Plotting the results...")
    x_array = list(range(1, y_predict.shape[0]+1))
    fig = plt.figure()
    plt.scatter(x_array, y_predict, label = "Predicted")
    plt.scatter(x_array, y_test, label = "Actual values")

    plt.xlabel('Experiment Number')
    plt.ylabel('RMS')
    plt.legend(loc = 2)

    #plt.show()
    st.pyplot(fig)
    
    #### PERFORMANCE ANALYSIS ####
    st.write("Accuracy metrics...")
    st.write("RMSE")
    RMS_error = mean_squared_error(y_test, y_predict)
    st.write(RMS_error)
    st.write("Mean Absolute Error")
    MA_error = mean_absolute_error(y_test, y_predict)
    st.write(MA_error)
    st.write("R2 Score")
    r2 = r2_score(y_test, y_predict)
    st.write(r2)
    st.write("Adjusted R2 Score")
    adj_r2_score = 1 - ((1-r2)*(61-1)/(61-5-1))
    st.write(adj_r2_score)
    
    
elif dataset_name == 'Slope Prediction':
    def read_preprocess(file):
        df = pd.read_excel(file)
        pd.set_option('display.max_rows', 15)
        df = df.iloc[72:85, :]
        df.drop(df.columns[[0, 4, 8, 9, 10, 11, 12]], axis=1, inplace=True)
        df.drop(df.loc[:, 'Unnamed: 13':'Unnamed: 15'].columns, axis = 1, inplace = True)
        df.rename(columns={'RMS': 'Slope of the line'},inplace=True)
        return df

    uploaded_file = st.file_uploader("Please upload the datafile",type=['csv','xls', 'xlsx'])

    time_counter = 0

    while uploaded_file is None:
        time.sleep(1)
        time_counter += 3
        if time_counter > 20:break

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head(10))
    else:
        raise ValueError("This isn't a file!")

    df = read_preprocess(uploaded_file)

    reg_name = st.sidebar.selectbox(
        'Select regressor',
        ('Linear regressor', 'SVR')
    )
    
    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVR':
            degree = st.sidebar.slider('degree', 1, 10)
            params['degree'] = degree
        return params

    params = add_parameter_ui(reg_name)
    
    def get_regressor(clf_name, params):
        clf = None
        if clf_name == 'SVR':
            clf = SVR(kernel = 'poly', degree = params['degree'])
        elif clf_name == 'Linear regressor':
            clf = LinearRegression()
        return clf
    
    regressor = get_regressor(reg_name, params)

    #### REGRESSION ####

    st.write("Training the model...")
    X = df.iloc[:, 0:5]
    y = df["Slope of the line"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    regressor.fit(X_train, y_train)
    

    # predictions
    st.write("Returning the predictions...")
    y_predict = regressor.predict(X_test)

    st.write(y_predict)

    #### PLOT DATASET ####
    st.write("Plotting the results...")
    x_array = list(range(1, y_predict.shape[0]+1))
    fig = plt.figure()
    plt.scatter(x_array, y_predict, label = "Predicted")
    plt.scatter(x_array, y_test, label = "Actual values")

    plt.xlabel('Experiment Number')
    plt.ylabel('Slope')
    plt.legend(loc = 2)

    #plt.show()
    st.pyplot(fig)
    
    #### PERFORMANCE ANALYSIS ####
    st.write("Accuracy metrics...")
    st.write("RMSE")
    RMS_error = mean_squared_error(y_test, y_predict)
    st.write(RMS_error)
    st.write("Mean Absolute Error")
    MA_error = mean_absolute_error(y_test, y_predict)
    st.write(MA_error)
    st.write("R2 Score")
    r2 = r2_score(y_test, y_predict)
    st.write(r2)
    st.write("Adjusted R2 Score")
    adj_r2_score = 1 - ((1-r2)*(61-1)/(61-5-1))
    st.write(adj_r2_score)


# In[ ]:


get_ipython().system('jupyter nbconvert   --to script streamlit_final.ipynb')
get_ipython().system("awk '!/ipython/' streamlit_final.py >  temp.py && mv temp.py app.py && rm streamlit_final.py")
get_ipython().system('streamlit run app.py')


# In[ ]:




