# app.py (Improved for local Flask use)
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("stroke_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [
            int(request.form['gender']),
            float(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            int(request.form['ever_married']),
            int(request.form['work_type']),
            int(request.form['Residence_type']),  # Use capital 'R' to match training feature
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            int(request.form['smoking_status'])
        ]

        columns = ["gender", "age", "hypertension", "heart_disease", "ever_married",
                   "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
        df = pd.DataFrame([data], columns=columns)

        scaled = scaler.transform(df)
        reshaped = scaled.reshape((scaled.shape[0], scaled.shape[1], 1))

        pred = model.predict(reshaped, batch_size=1)
        risk = np.argmax(pred, axis=1)[0]

        label = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        return render_template("index.html", prediction=f"Stroke Risk: {label[risk]}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
