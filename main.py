import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open("../house_price_prediction/artifacts/model.pkl","rb"))
df = pickle.load(open('../house_price_prediction/artifacts/data.pkl','rb'))

st.title("House Price Predictor")

bedroom = st.number_input('Number of bedrooms')
space = st.number_input('Space of the house')
room = st.number_input('Number of rooms')
lot = st.number_input('Lot')
tax = st.number_input('Tax need to pay')
bathroom = st.number_input('Number of bathrooms')
garage = st.number_input('Number of garage')
condition = st.selectbox('condition',df['Condition'].unique())

if st.button('Predict Price'):
    query = np.array([bedroom,space,room,lot,tax,bathroom,garage,condition])
    query = query.reshape(1,8)
    st.title("The predicted price of this house is " + str(int(pipe.predict(query)[0])))