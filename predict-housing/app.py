import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv('housing.csv')

# Encode categorical variable
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Features and target
X = df.drop('Price', axis=1)
y = df['Price']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Streamlit App UI
# ------------------------------
st.title("🏠 Housing Price Predictor")
st.write("Predict the price of a house based on its features!")

# User input
size = st.number_input("Size (in sqft)", min_value=500, max_value=5000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
age = st.number_input("Age of House (in years)", min_value=0, max_value=100, value=10)
location = st.selectbox("Location", ["Suburb", "City"])

# Prepare input for model
if location == "City":
    location_city = 1
else:
    location_city = 0

input_data = pd.DataFrame({
    'Size':[size],
    'Bedrooms':[bedrooms],
    'Bathrooms':[bathrooms],
    'Age':[age],
    'Location_City':[location_city]
})

# Prediction
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${predicted_price:,.2f}")

# Optional: show dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)
