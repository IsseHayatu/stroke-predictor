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
        print("📥 Form Keys:", request.form.keys())
        for key in request.form:
            value = request.form[key]
            print(f"🔍 {key}: '{value}'")
            if value.strip() == "":
                raise ValueError(f"Missing value for '{key}'")
            input_values.append(float(value))

        print("✅ Inputs:", input_values)

        scaled = scaler.transform([input_values])
        print("✅ Scaled:", scaled)

        reshaped = np.expand_dims(scaled, axis=2)
        print("📐 Reshaped:", reshaped.shape)

        prediction = model.predict(reshaped)
        print("✅ Raw prediction:", prediction)

        result = ["Low", "Medium", "High"][np.argmax(prediction)]

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        import traceback
        print("❌ Crash:", e)
        traceback.print_exc()
        return f"⚠️ Server Error: {e}"
