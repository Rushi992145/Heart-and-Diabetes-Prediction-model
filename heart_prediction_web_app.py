# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 16:36:26 2024

@author: RUSHIKESH HIRAY
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('D:/122B1F034/FODS_models/trained_model.sav','rb'))

#function for predicting
def heart_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if(prediction==0) :
        return'The person does not have Heart Disease'
    else :
        return'The person has Heart Disease'
        
def main():
    
    # giving a  title
    st.title('Heart Disease Prediction Web-App')
    
    # get the input from user
    
    age = st.text_input('Age')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain types')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    restecg = st.text_input('Resting Electrocardiographic results')
    thalach = st.text_input('Maximum Heart Rate achieved')
    exang = st.text_input('Exercise Induced Angina')
    oldpeak = st.text_input('ST depression induced by exercise')
    slope = st.text_input('Slope of the peak exercise ST segment')
    ca = st.text_input('Major vessels colored by flourosopy')
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    #diagnosis code
    
    diagnosis = ''
    # button for get result
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    if st.button('Heart Disease cheak'):
        user_input = [float(x) for x in user_input]
        diagnosis = heart_prediction(user_input)
        
    st.success(diagnosis)
    
    
if __name__=='__main__':
    main()
        
    
    
    
    