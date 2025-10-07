# ------------------
# Import Libraries
# ------------------
import streamlit as st
import numpy as np
import joblib

# ------------------
# Page Configuration
# ------------------
# This sets the title and icon of your app's browser tab.
st.set_page_config(
    page_title="Crude Oil Price Predictor",
    page_icon="🛢️",
    layout="wide"
)

# ------------------
# Load Models & Scalers
# ------------------
# We load all models at the start to make the app responsive.
# Using a dictionary makes it easy to manage multiple models.
try:
    models = {
        "XGBoost": joblib.load("xgboost.pkl"),
        "ARIMA": joblib.load("arima.pkl"),
        "SARIMA": joblib.load("sarima.pkl"),
        "LSTM": joblib.load("lstm.pkl"),
        "ANN": joblib.load("ann_model.pkl")
    }
    # Scalers are stored separately as they are only needed for specific models.
    scalers = {
        "LSTM": joblib.load("lstm_scaler.pkl"),
        "ANN": joblib.load("scaler.pkl") # Assuming you have a general or ANN-specific scaler
    }
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please ensure all .pkl files are in the directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading files: {e}")
    st.stop()


# ------------------
# Sidebar (Navigation Bar)
# ------------------
st.sidebar.title("Model Selection")
st.sidebar.markdown("Choose the machine learning model you want to use for the prediction.")
model_choice = st.sidebar.selectbox(
    "Select Model:",
    list(models.keys()) # Creates a dropdown from the model names
)

# ------------------
# Main App Interface
# ------------------
st.title(f"🛢️ Crude Oil Price Predictor")
st.markdown(f"**Using the `{model_choice}` Model**")
st.write("Enter the prices for the last 5 days to predict the price for the next day.")

st.header("Enter Recent Prices")

# Create input fields for the last 5 days' prices. These are used by XGBoost, ANN, and LSTM.
# ARIMA/SARIMA models use their internal history but we show the inputs for a consistent UI.
cols = st.columns(5)
lag_inputs = []
default_prices = [80.0, 81.0, 82.0, 83.0, 84.0] # More realistic default values
for i, col in enumerate(cols):
    with col:
        lag_inputs.append(st.number_input(f"Day -{i+1} ($)", min_value=0.0, step=0.5, value=default_prices[i]))

# ------------------
# Prediction Logic
# ------------------
if st.button(f"Predict Price with {model_choice}"):
    selected_model = models[model_choice]
    prediction = None

    try:
        # --- Logic for models that use input features (XGBoost, ANN, LSTM) ---
        if model_choice in ["XGBoost", "ANN", "LSTM"]:
            input_array = np.array(lag_inputs).reshape(1, -1)

            # Check if the model requires data scaling
            if model_choice in scalers:
                scaler = scalers[model_choice]
                input_array = scaler.transform(input_array)
            
            # LSTM requires a 3D input shape [samples, timesteps, features]
            if model_choice == "LSTM":
                input_array = input_array.reshape(1, 5, 1)

            prediction = selected_model.predict(input_array)

        # --- Logic for time-series forecasting models (ARIMA, SARIMA) ---
        elif model_choice in ["ARIMA", "SARIMA"]:
            # These models typically use .forecast() based on the data they were trained on.
            # The .pkl file must contain the fitted model object.
            prediction = selected_model.forecast(steps=1)

        # --- Display the final prediction ---
        if prediction is not None:
            # Extract the single prediction value, regardless of the output format
            final_prediction = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction[0]
            st.success(f"**Predicted Price for the Next Day: ${float(final_prediction):.2f}**")
        else:
            st.warning("Prediction could not be made for the selected model.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")