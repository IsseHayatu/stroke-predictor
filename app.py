from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np

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
        # Convert form inputs to float
        inputs = [float(request.form[key]) for key in request.form if request.form[key].strip() != '']
        
        if len(inputs) != 10:
            return render_template("index.html", error="Please fill in all 10 fields correctly.")

        # Preprocess
        scaled = scaler.transform([inputs])
        reshaped = np.expand_dims(scaled, axis=2)

        # Predict class (one-hot style)
        prediction = model.predict(reshaped)
        result = ["Low", "Medium", "High"][np.argmax(prediction)]

        return render_template("index.html", prediction=result, user_input=inputs)
    except Exception as e:
        return render_template("index.html", error=f"⚠️ Invalid input: {e}")

# NOTE: Remove app.run() when deploying to Render (Render uses Gunicorn via Procfile)
