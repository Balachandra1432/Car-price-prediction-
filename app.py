import pandas as pd
import numpy as np
import pickle as pl
import streamlit as st 

model=pl.load(open("car_price_pridiction_model.pkl","rb"))

st.header("Car Price Prediction using ML Model")

cars_data=pd.read_csv("Car Price.csv")

brand=st.selectbox("Select The Car Brand",cars_data["Brand"].unique())
year=st.selectbox("Select The year of the manufacturer",cars_data["Year"].unique())
km=st.slider("No of KM have been driven",1000,200000)
fuel=st.selectbox("Select fuel type",cars_data["Fuel"].unique())
seller_type=st.selectbox("Sellet type",cars_data["Seller_Type"].unique())
transmission=st.selectbox("Transmission type",cars_data["Transmission"].unique())
owner=st.selectbox("Owner",cars_data["Owner"].unique())


if st.button("Pridict"):
    input_data=pd.DataFrame([[brand,year,km,fuel,seller_type,transmission,owner]],
                        columns=["Brand","Year","KM_Driven","Fuel","Seller_Type","Transmission","Owner"])
    #lebel encoding 
    input_data["Brand"].replace(['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota',
       'Chevrolet', 'Renault', 'Volkswagen', 'Skoda', 'Nissan', 'Audi', 'BMW',
       'Fiat', 'Datsun', 'Mercedes-Benz', 'Jaguar', 'Mitsubishi', 'Land',
       'Volvo', 'Ambassador', 'Jeep', 'MG', 'OpelCorsa', 'Daewoo', 'Force',
       'Isuzu', 'Kia'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],inplace=True)
    input_data["Fuel"].replace(['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'],[1,2,3,4,5],inplace=True)
    input_data["Seller_Type"].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace=True)
    input_data["Transmission"].replace(['Manual', 'Automatic'],[1,2],inplace=True)
    input_data["Owner"].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner',
       'Test Drive Car'],[1,2,3,4,5],inplace=True)
    #st.write(input_data)
    car_price=model.predict(input_data)
    st.markdown("Car price is "+str(round(car_price[0],2))+" Ruppes")


