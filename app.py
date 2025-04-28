from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import numpy as np

app = Flask(__name__)
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
    app.run(debug=True)
