from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np
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
        inputs = [float(request.form[key]) for key in request.form]
        scaled = scaler.transform([inputs])
        reshaped = np.expand_dims(scaled, axis=2)
        prediction = model.predict(reshaped)
        result = ["Low", "Medium", "High"][np.argmax(prediction)]
        return render_template("index.html", prediction=result, user_input=inputs)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
