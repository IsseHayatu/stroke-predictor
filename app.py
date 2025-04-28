from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Initialize the app
app = Flask(__name__)

# Load the trained CNN model and scaler once at startup
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route - render the form
@app.route('/')
def home():
    return render_template('index.html')

# Predict route - handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        residence_type = int(request.form['residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Make input array
        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married,
                                work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Reshape for CNN model input
        reshaped = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))

        # Predict with batch_size=1 to reduce memory usage
        prediction = model.predict(reshaped, batch_size=1)

        # Get the risk class
        risk = np.argmax(prediction, axis=1)[0]

        # Map risk class to label
        risk_label = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

        return render_template('index.html', prediction_text=f'Stroke Risk: {risk_label[risk]}')

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# Run the app (for local testing only)
if __name__ == "__main__":
    app.run(debug=True)
