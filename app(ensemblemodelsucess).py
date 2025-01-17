import pickle

import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Set page layout
st.set_page_config(layout="wide")

# Load the pre-trained ensemble model and scaler
@st.cache_resource
def load_model_and_scaler():
    ensemble_model = pickle.load(open('sources/ensemble_model.pkl', 'rb'))  # Load the ensemble model
    scaler = joblib.load('sources/scaler.pkl')  # Load the scaler
    return ensemble_model, scaler

ensemble_model, scaler = load_model_and_scaler()

# Streamlit app title
st.title("Gestational Diabetes Mellitus (GDM) Prediction")
st.write("This app predicts the likelihood of GDM based on patient input data.")

# User Input Form
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
#pregnancy = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
gestation = st.number_input("Gestation in Weeks", min_value=0, max_value=50, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hdl = st.number_input("HDL Cholesterol", min_value=0.0, max_value=100.0, value=50.0)
family_history = st.selectbox("Family History of Diabetes (Yes=1, No=0)", [0, 1])
#prenatal_loss = st.number_input("Number of Prenatal Losses", min_value=0, max_value=20, value=0)
#large_child = st.selectbox("Delivered Large Child (>4 kg) (Yes=1, No=0)", [0, 1])
pcos = st.selectbox("PCOS (Yes=1, No=0)", [0, 1])
#sys_bp = st.number_input("Systolic Blood Pressure", min_value=0.0, max_value=200.0, value=120.0)
dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
ogtt = st.number_input("OGTT (mg/dL)", min_value=0.0, max_value=500.0, value=140.0)
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0)
#sedentary_lifestyle = st.selectbox("Sedentary Lifestyle (Yes=1, No=0)", [0, 1])
prediabetes = st.selectbox("History of Prediabetes (Yes=1, No=0)", [0, 1])


# Prediction Button
if st.button("Predict GDM"):
    # Gather input data
    input_data = np.array([[age, gestation, bmi, hdl, family_history, pcos, dia_bp, ogtt, hemoglobin,prediabetes]])

    st.write("### Input Data")
    st.write(input_data)

    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    st.write("### Scaled Data")
    st.write(input_data_scaled)

    try:
        # Make prediction
        prediction_proba = ensemble_model.predict_proba(input_data_scaled)[0][1]  # Probability for GDM class
        predicted_class = "GDM" if prediction_proba > 0.5 else "Non-GDM"

        st.write("### Prediction Result")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Probability of GDM: {prediction_proba:.2f}")
    except Exception as e:
        st.error("An error occurred during prediction. Please check your input data.")
        st.error(str(e))
