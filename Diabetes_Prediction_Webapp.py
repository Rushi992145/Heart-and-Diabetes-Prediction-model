
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the data
diabetes_data = pd.read_csv('C:/Users/Saurabh/OneDrive/Desktop/ai/diabetes.csv')

# Separate features and target variable
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save the trained model
with open('your_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save the scaler for later use in Streamlit
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Check accuracy (optional)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('your_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app UI
st.title("Diabetes Prediction Model")

# Input fields for each feature
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    # Create an input array and reshape it
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
    
    # Standardize the input data
    std_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(std_data)
    
    # Show prediction result
    if prediction[0] == 0:
        st.write("The person is **not diabetic**.")
    else:
        st.write("The person is **diabetic**.")




