import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💰", layout="centered")

st.title("💰 Insurance Cost Prediction using KNN Regression")
st.markdown("Predict medical insurance charges based on user inputs")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("insurance.csv")

# ---------------- ENCODING ----------------
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])
df["smoker"] = le_smoker.fit_transform(df["smoker"])
df["region"] = le_region.fit_transform(df["region"])

X = df.drop("charges", axis=1)
y = df["charges"]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_scaled, y)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Patient Details")

age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["female", "male"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["no", "yes"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# ---------------- ENCODE INPUT ----------------
sex_val = le_sex.transform([sex])[0]
smoker_val = le_smoker.transform([smoker])[0]
region_val = le_region.transform([region])[0]

input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

input_scaled = scaler.transform(input_data)

# ---------------- PREDICTION ----------------
if st.button("💡 Predict Insurance Cost"):

    prediction = model.predict(input_scaled)

    st.subheader("Prediction Result:")

    st.success(f"💰 Estimated Insurance Cost: ${prediction[0]:.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using KNN Regression and Streamlit")