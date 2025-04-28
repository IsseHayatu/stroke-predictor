from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and scaler
model = load_model("stroke_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# List of feature names expected by scaler
FEATURES = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collecting input from form
        input_values = [
            float(request.form["gender"]),
            float(request.form["age"]),
            float(request.form["hypertension"]),
            float(request.form["heart_disease"]),
            float(request.form["ever_married"]),
            float(request.form["work_type"]),
            float(request.form["Residence_type"]),
            float(request.form["avg_glucose_level"]),
            float(request.form["bmi"]),
            float(request.form["smoking_status"]),
        ]

        # Turn input into a pandas DataFrame with correct feature names
        input_df = pd.DataFrame([input_values], columns=FEATURES)

        # Scale the data correctly
        scaled = scaler.transform(input_df)

        # Reshape for CNN input
        reshaped = np.expand_dims(scaled, axis=2)

        # Predict
        probability = model.predict(reshaped)[0][0]

        if probability < 0.3:
            result = "Low Risk"
        elif probability <= 0.7:
            result = "Medium Risk"
        else:
            result = "High Risk"

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        return f"⚠️ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
