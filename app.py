from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = load_model("stroke_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        inputs = []
        for key in request.form:
            value = request.form[key].strip()
            if value == "":
                return render_template("index.html", error="⚠️ Please fill in all fields.")
            inputs.append(float(value))

        if len(inputs) != 10:
            return render_template("index.html", error="⚠️ Exactly 10 inputs are required.")

        # Scale the inputs and reshape for CNN
        scaled = scaler.transform([inputs])
        reshaped = np.expand_dims(scaled, axis=2)

        # Predict with CNN model
        prediction = model.predict(reshaped)
        result = ["Low", "Medium", "High"][np.argmax(prediction)]

        return render_template("index.html", prediction=result, user_input=inputs)
    
    except Exception as e:
        return render_template("index.html", error=f"⚠️ Error: {e}")
