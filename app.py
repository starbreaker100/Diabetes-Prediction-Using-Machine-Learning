import pandas as pd
import pickle
import streamlit as st

# Load the trained models
with open('voting_classifier_model.pkl', 'rb') as file:
    voting_classifier = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

# CSS Styling for better visuals
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stNumberInput input {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def calculate_bmi(weight, height):
    return round(weight / (height ** 2), 2) if height > 0 else None

def get_hba1c_level(glucose_avg):
    return round((glucose_avg + 46.7) / 28.7, 2)

# Streamlit UI
st.title("🌟 Diabetes Prediction and Risk Analysis")
st.write("### Input your health data to predict diabetes and analyze risk trends")

# Input fields for user data with unique keys
weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0)
height = st.number_input("Height (m)", min_value=0.0, value=1.75)
glucose_avg = st.number_input("Average Glucose Level (mg/dL)", min_value=0.0, value=120.0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", options=[0, 1])
heart_disease = st.selectbox("Heart Disease", options=[0, 1])
gender = st.selectbox("Gender", options=['Male', 'Female'])
smoking_history = st.selectbox("Smoking History", options=['never', 'current', 'former'])

# Derived features
bmi = calculate_bmi(weight, height)
hba1c_level = get_hba1c_level(glucose_avg)

st.write("Calculated BMI:", bmi)
st.write("Estimated HbA1c Level:", hba1c_level)

# Button to make prediction
if st.button("Predict Diabetes and Show Risk"):
    if bmi is None:
        st.error("Please provide a valid height and weight.")
    else:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [glucose_avg],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'gender_Male': [1 if gender == 'Male' else 0],
            'smoking_history_former': [1 if smoking_history == 'former' else 0],
            'smoking_history_never': [1 if smoking_history == 'never' else 0],
            'smoking_history_current': [1 if smoking_history == 'current' else 0],
        })

        # Make predictions
        diabetes_prediction = voting_classifier.predict(input_data)
        rf_prediction_proba = random_forest_model.predict_proba(input_data)[:, 1]

        if diabetes_prediction[0] == 1:
            st.error("🚨 **Prediction:** You are likely diabetic. Please consult a healthcare professional for detailed advice.")
        else:
            st.success("✅ **Prediction:** You are likely non-diabetic. Continue maintaining a healthy lifestyle.")

        # Calculate risk of becoming diabetic in the future
        future_risk = rf_prediction_proba[0]  # Probability from Random Forest

        # Provide detailed risk feedback
        if future_risk < 0.1:
            st.write("**Minimal Risk (<10%)**: Great! Keep up your healthy habits.")
        elif future_risk < 0.3:
            st.write(f"**Low Risk ({future_risk * 100:.1f}%)**: Maintain a healthy diet and regular exercise.")
        elif future_risk < 0.7:
            st.write(f"**Moderate Risk ({future_risk * 100:.1f}%)**: Consider consulting a healthcare professional for preventive advice.")
        else:
            st.write(f"**High Risk ({future_risk * 100:.1f}%)**: Seek immediate consultation with a healthcare professional.")
