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
        print("🌐 Starting prediction route")
        input_values = []
        for key in request.form:
            value = request.form[key]
            print(f"🔍 Received {key}: '{value}'")
            if value.strip() == "":
                raise ValueError(f"Missing value for {key}")
            input_values.append(float(value))

        print("📊 Raw input:", input_values)
        scaled = scaler.transform([input_values])
        print("✅ Scaled input:", scaled)

        reshaped = np.expand_dims(scaled, axis=2)  # Makes shape (1, 10, 1)
        print("📐 Reshaped input:", reshaped.shape)

        prediction = model.predict(reshaped)
        print("✅ Prediction output:", prediction)

        result = ["Low", "Medium", "High"][np.argmax(prediction)]
        print("🎯 Predicted class:", result)

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        import traceback
        print("❌ Exception during prediction:", e)
        traceback.print_exc()
        return f"⚠️ Server Error: {e}"
