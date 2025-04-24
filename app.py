from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os

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
        input_values = [
            float(request.form["gender"]),
            float(request.form["age"]),
            float(request.form["hypertension"]),
            float(request.form["heart_disease"]),
            float(request.form["ever_married"]),
            float(request.form["work_type"]),
            float(request.form["residence_type"]),
            float(request.form["avg_glucose_level"]),
            float(request.form["bmi"]),
            float(request.form["smoking_status"]),
        ]

        input_array = np.array([input_values])
        scaled = scaler.transform(input_array)
        reshaped = np.expand_dims(scaled, axis=2)

        # Predict: single sigmoid output
        probability = model.predict(reshaped)[0][0]
        if probability < 0.3:
            result = "Low"
        elif probability <= 0.7:
            result = "Medium"
        else:
            result = "High"

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        return f"⚠️ Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
