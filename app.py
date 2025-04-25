from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
print("📦 Loading model...")
model = load_model("stroke_model.h5")
print("✅ Model loaded.")

print("📦 Loading scaler...")
scaler = pickle.load(open("scaler.pkl", "rb"))
print("✅ Scaler loaded.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        for key in request.form:
            value = request.form[key]
            if value.strip() == "":
                raise ValueError(f"Missing value for {key}")
            input_values.append(float(value))

        # Scale input
        scaled = scaler.transform([input_values])
        reshaped = np.expand_dims(scaled, axis=2)

        # Predict
        prediction = model.predict(reshaped)
        result = ["Low", "Medium", "High"][np.argmax(prediction)]

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        print("❌ Crash:", e)
        return f"⚠️ Server Error: {e}"

# Render uses gunicorn, no need for app.run()
