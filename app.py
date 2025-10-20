import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from keras.src.saving import load_model

# Load model and preprocessing objects
model = load_model("model.h5")

with open("onehotencoder.pkl", 'rb') as file:
    onehotencoder = pickle.load(file)

with open('labelencoder.pkl', 'rb') as file:
    labelencoder = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', labelencoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'CreditScore': [credit_score],
    'Gender': [labelencoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehotencoder.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data = input_data.drop('Geography', axis=1)

# Reorder columns to match scaler
columns_order = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
                 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] + \
                list(onehotencoder.get_feature_names_out(['Geography']))
input_data = input_data[columns_order]

# Scale
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)
st.write("Churn Probability:", round(prediction[0][0], 4))
st.write("Will the customer churn?" , "Yes" if prediction[0][0] > 0.5 else "No")
