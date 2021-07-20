#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import io
import string
import time
import os
import pandas as pd
import numpy as np
import joblib as joblib
import matplotlib.pyplot as plt

st.title('RUL Predictor')

st.write("""
## This app uses two different datasets
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('RMS Prediction', 'Slope Prediction')
)

st.write(dataset_name, "Dataset")

if dataset_name == 'RMS Prediction':
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
        time_counter += 7
        if time_counter > 20:break

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head(10))
    else:
        raise ValueError("This isn't a file!")

    df = read_preprocess(uploaded_file)

    def get_model(file):
        regressor = joblib.load(file)
        return regressor

    uploaded_file = st.file_uploader("Please upload the rms regressor saved model",type=['pkl'])

    time_counter = 0

    while uploaded_file is None:
        time.sleep(1)
        time_counter += 7
        if time_counter > 15:break
    if uploaded_file is not None:
        st.write("Model uploaded succesfully")
    else:
        raise ValueError("This isn't a file")

    regressor = get_model(uploaded_file) 

    #### REGRESSION ####

    X_test = df.iloc[:, 0:5]
    y_test = df["RMS(g)"]

    # predictions
    st.write("Returning the predictions...")
    y_predict = regressor.predict(X_test)

    st.write(y_predict)

    #### PLOT DATASET ####
    st.write("Plotting the results...")
    x_array = list(range(1, y_predict.shape[0]+1))
    fig = plt.figure()
    plt.scatter(x_array, y_predict)
    plt.scatter(x_array, y_test)

    plt.xlabel('Experiment Number')
    plt.ylabel('RMS')

    #plt.show()
    st.pyplot(fig)
    
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
        time_counter += 7
        if time_counter > 20:break

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write(df.head(10))
    else:
        raise ValueError("This isn't a file!")

    df = read_preprocess(uploaded_file)

    def get_model(file):
        regressor = joblib.load(file)
        return regressor

    uploaded_file = st.file_uploader("Please upload the slope regressor saved model",type=['pkl'])

    time_counter = 0

    while uploaded_file is None:
        time.sleep(1)
        time_counter += 7
        if time_counter > 15:break
    if uploaded_file is not None:
        st.write("Model uploaded succesfully")
    else:
        raise ValueError("This isn't a file")

    regressor = get_model(uploaded_file) 

    #### REGRESSION ####

    X_test = df.iloc[:, 0:5]
    y_test = df["Slope of the line"]

    # predictions
    st.write("Returning the predictions...")
    y_predict = regressor.predict(X_test)

    st.write(y_predict)

    #### PLOT DATASET ####
    st.write("Plotting the results...")
    x_array = list(range(1, y_predict.shape[0]+1))
    fig = plt.figure()
    plt.scatter(x_array, y_predict)
    plt.scatter(x_array, y_test)

    plt.xlabel('Experiment Number')
    plt.ylabel('Slope')

    #plt.show()
    st.pyplot(fig)
