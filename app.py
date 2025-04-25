from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import os
import time

app = Flask(__name__)

# Load the model and scaler
print("📦 Loading model...")
model = load_model("stroke_model.h5")
print("✅ Model loaded.")
print("📐 Model input shape:", model.input_shape)

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
            print(f"🔍 {key}: '{value}'")
            if value.strip() == "":
                raise ValueError(f"Missing value for {key}")
            input_values.append(float(value))

        print("📊 Raw input values:", input_values)
        scaled = scaler.transform([input_values])
        reshaped = scaled  # ✅ No reshape needed
        print("📐 Input to model:", reshaped.shape)

        start = time.time()
        prediction = model.predict(reshaped)
        duration = time.time() - start
        print(f"✅ Prediction took {duration:.2f} seconds")

        result = ["Low", "Medium", "High"][np.argmax(prediction)]
        print("🎯 Predicted risk:", result)

        return render_template("index.html", prediction=result, user_input=input_values)

    except Exception as e:
        import traceback
        print("❌ Exception:", e)
        traceback.print_exc()
        return f"⚠️ Server Error: {e}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
