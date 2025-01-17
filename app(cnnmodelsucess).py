import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Set page layout
st.set_page_config(layout="wide")

# Load the pre-trained CNN model and scaler
@st.cache_resource
def load_model_and_scaler():
    cnn_model = load_model('sources/cnn_model.h5')  # Load the CNN model
    scaler = joblib.load('sources/scaler.pkl')  # Load the scaler
    return cnn_model, scaler

cnn_model, scaler = load_model_and_scaler()

# Streamlit app title
st.title("Gestational Diabetes Mellitus (GDM) Prediction")
st.write("This app predicts the likelihood of GDM based on patient input data.")

# User Input Form
st.header("Enter Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gestation = st.number_input("Gestation in Weeks", min_value=0, max_value=50, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hdl = st.number_input("HDL Cholesterol", min_value=0.0, max_value=100.0, value=50.0)
family_history = st.selectbox("Family History of Diabetes (Yes=1, No=0)", [0, 1])
pcos = st.selectbox("PCOS (Yes=1, No=0)", [0, 1])
dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
ogtt = st.number_input("OGTT (mg/dL)", min_value=0.0, max_value=500.0, value=140.0)
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0)
prediabetes = st.selectbox("History of Prediabetes (Yes=1, No=0)", [0, 1])

# Prediction Button
if st.button("Predict GDM"):
    # Gather input data
    input_data = np.array([[age, gestation, bmi, hdl, family_history, pcos, dia_bp, ogtt, hemoglobin, prediabetes]])

    st.write("### Input Data")
    st.write(input_data)

    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    st.write("### Scaled Data")
    st.write(input_data_scaled)

    # Reshape input data for CNN (if required)
    # Assuming the CNN expects input of shape (1, 10, 1)
    cnn_input = input_data_scaled.reshape(1, input_data_scaled.shape[1], 1)

    try:
        # Make prediction
        prediction_proba = cnn_model.predict(cnn_input)[0][0]  # Assuming single output neuron for probability
        predicted_class = "GDM" if prediction_proba > 0.5 else "Non-GDM"

        st.write("### Prediction Result")
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Probability of GDM: {prediction_proba:.2f}")
    except Exception as e:
        st.error("An error occurred during prediction. Please check your input data.")
        st.error(str(e))
