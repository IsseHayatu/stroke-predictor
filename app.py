from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os

app = Flask(__name__)

print("📦 Loading model...")
model = load_model("stroke_model.h5")
print("✅ Model loaded.")
print(f"📐 Model input shape: {model.input_shape}")

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

        input_array = np.array([input_values])
        scaled = scaler.transform(input_array)
        reshaped = np.expand_dims(scaled, axis=2)
        prediction = model.predict(reshaped)

        result = ["Low", "Medium", "High"][np.argmax(prediction)]
        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        print("❌ Crash:", e)
        return f"⚠️ Server Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
