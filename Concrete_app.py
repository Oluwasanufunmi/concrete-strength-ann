import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Concrete Strength Prediction App", layout="wide")

st.markdown("""<style>

/* Main App Background */
.stApp {background-color: Green;}

/* Main Title */
h1 {color: white;
    text-align: center;
    font-weight: bold;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: Grey;
}

[data-testid="stSidebar"] * {color: white;}

/* Input Boxes */
.stNumberInput input {border-radius: 8px;}

</style>""", unsafe_allow_html=True)

ann_model = load_model("ann_concrete_strength_model.keras")
scaler = joblib.load("concrete_scaler.pkl")

st.markdown("""<h1>
Concrete Compressive Strength Prediction App
</h1>
""", unsafe_allow_html=True)

st.write("""
This application predicts the Compressive Strength of Concrete using an
Artificial Neural Network (ANN) Deep learning model.
""")

# Navigation Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Information about the Project",
        "Model Performance",
        "Prediction"
    ]
)

if page == "Information about the Project":

    st.header("About the Project")

    st.write("""
    Concrete compressive strength is one of the most important properties in
    civil engineering and construction. It determines how much load a concrete
    structure can withstand before failure. And the Dataset is downloaded from the UCI Machine Learning Repository.

    This project uses a Deep Learning Artificial Neural Network ANN model
    to predict concrete compressive strength based on the quantities of
    ingredients used in the concrete mixture.
    """)

    st.subheader("Project Objective")

    st.write("""
    The main objective of this project is to build a regression-based deep
    learning model that can predict the compressive strength of concrete using
    its mixture components and curing age.
    """)

    st.subheader("Dataset Information")

    st.write("""
    The dataset is a **Regression Multivariate dataset**.  
    It contains numerical input features representing concrete ingredients and
    the age of the concrete sample.

    The target variable is:

    **Concrete Compressive Strength MPa**
    """)

    st.subheader("Dataset Features")

    features = pd.DataFrame({
        "Feature": [
            "Cement",
            "Blast Furnace Slag",
            "Fly Ash",
            "Water",
            "Superplasticizer",
            "Coarse Aggregate",
            "Fine Aggregate",
            "Age"
        ],
        "Description": [
            "Amount of cement in kg/m³",
            "Amount of blast furnace slag in kg/m³",
            "Amount of fly ash in kg/m³",
            "Amount of water in kg/m³",
            "Amount of superplasticizer in kg/m³",
            "Amount of coarse aggregate in kg/m³",
            "Amount of fine aggregate in kg/m³",
            "Curing age of concrete in days"
        ]
    })

    st.dataframe(features, use_container_width=True)
    

    st.subheader("Project Workflow")
    st.write("""
        1. Data Collection
        2. Data Cleaning
        3. Exploratory Data Analysis
        4. Feature Scaling
        5. Machine Learning Modeling
        6. Hyperparameter Tuning
        7. ANN Deep Learning Modeling
        8. Model Evaluation
        """)

    st.subheader("Why Deep Learning?")
    st.write("""
    Concrete strength does not depend on one ingredient alone. It depends on
    complex relationships between cement, water, aggregates, and age.

    An ANN model can learn nonlinear patterns from the data, making it suitable
    for this regression prediction task.
    """)

elif page == "Model Performance":

    st.header("Model Performance")

    st.write("""
    Several machine learning models were trained first as baseline models.
    The ANN model was then built as the main deep learning model for this project.
    """)

    st.subheader("Machine Learning Model Results")

    ml_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Gradient Boosting",
            "Random Forest",
            "Tuned Hist Gradient Boosting"
        ],
        "R² Score": [
            0.5801,
            0.8979,
            0.9120,
            0.9390
        ],
        "MAE": [
            8.8960,
            4.0987,
            3.4702,
            2.7575
        ],
        "RMSE": [
            11.1922,
            5.5194,
            5.1245,
            4.2644
        ]
    })

    st.dataframe(ml_results, use_container_width=True)

    st.subheader("Deep Learning ANN Result")

    ann_results = pd.DataFrame({
        "Model": ["ANN Deep Learning Model"],
        "R² Score": [0.8723],
        "MAE": [4.2183],
        "RMSE": [6.1729]
    })

    st.dataframe(ann_results, use_container_width=True)

    st.subheader("Final Model Selected for Deployment")

    st.success("""
    The ANN Deep Learning model was selected for deployment because this project
    is focused on building a deep learning regression solution for concrete
    compressive strength prediction.
    """)

    st.write("""
    Although the Tuned Hist Gradient Boosting model achieved the highest
    predictive accuracy, the ANN model is used in the prediction interface
    because the main objective of this project is Deep Learning.
    """)

elif page == "Prediction":

    st.header("Predict Concrete Compressive Strength")

    st.write("""
    Enter the concrete mixture values below. The ANN model will predict the
    concrete compressive strength in MPa.
    """)

    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input(
            "Cement kg/m³",
            min_value=0.0,
            value=350.0
        )

        blast_furnace_slag = st.number_input(
            "Blast Furnace Slag kg/m³",
            min_value=0.0,
            value=100.0
        )

        fly_ash = st.number_input(
            "Fly Ash kg/m³",
            min_value=0.0,
            value=50.0
        )

        water = st.number_input(
            "Water kg/m³",
            min_value=0.0,
            value=180.0
        )

    with col2:
        superplasticizer = st.number_input(
            "Superplasticizer kg/m³",
            min_value=0.0,
            value=8.0
        )

        coarse_aggregate = st.number_input(
            "Coarse Aggregate kg/m³",
            min_value=0.0,
            value=1000.0
        )

        fine_aggregate = st.number_input(
            "Fine Aggregate kg/m³",
            min_value=0.0,
            value=750.0
        )

        age = st.number_input(
            "Age days",
            min_value=1,
            value=28
        )

    if st.button("Predict Concrete Strength"):

        input_data = pd.DataFrame({
            "Cement": [cement],
            "Blast_Furnace_Slag": [blast_furnace_slag],
            "Fly_Ash": [fly_ash],
            "Water": [water],
            "Superplasticizer": [superplasticizer],
            "Coarse_Aggregate": [coarse_aggregate],
            "Fine_Aggregate": [fine_aggregate],
            "Age": [age]
        })

        input_scaled = scaler.transform(input_data)

        prediction = ann_model.predict(input_scaled)

        st.success(
            f"Predicted Concrete Compressive Strength: {prediction[0][0]:.2f} MPa"
        )

        st.subheader("Input Values Used")
        st.dataframe(input_data, use_container_width=True)