import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder

# load the model : 
model = tf.keras.models.load_model('model.h5')

with open('one_hot_encoder_geo.pkl','rb') as file :
    one_hot_encoder_geo = pickle.load(file)


with open('label_encoder_gender.pkl','rb') as file :
    label_encoder_gender = pickle.load(file)


with open('scaler.pkl','rb') as file :
    scaler = pickle.load(file)

#stremlit app : 
st.title('Customer Churn Predicition')

#User Input : 
geography = st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimate Salary')
tenure = st.slider('Tenure')
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Convert dict → DataFrame (single row)
input_df = pd.DataFrame([input_data])

# one hot encode :Geography 
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

# Convert the dict to a single-row DataFrame
input_df = pd.DataFrame([input_data])

# Concatenate with geo_encoded_df
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_df)

#predict churn : 
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability : {prediction_proba:.2f}')

if prediction_proba > 0.5 : 
    st.write("the customer is likely to churn ....")
else : 
    st.write("the customer is not likely to churn ....")
