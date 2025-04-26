# app.py (updated for 11 features, binary classification)

from flask import Flask, render_template, request
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and preprocessing tools
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('form.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Encode categorical inputs
        gender = encoders['gender'].transform([request.form['gender']])[0]
        ever_married = encoders['ever_married'].transform([request.form['ever_married']])[0]
        work_type = encoders['work_type'].transform([request.form['work_type']])[0]
        residence_type = encoders['Residence_type'].transform([request.form['Residence_type']])[0]
        smoking_status = encoders['smoking_status'].transform([request.form['smoking_status']])[0]

        # Get numerical inputs
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # Combine all inputs (11 total)
        features = np.array([[
            gender,
            age,
            hypertension,
            heart_disease,
            ever_married,
            work_type,
            residence_type,
            avg_glucose_level,
            bmi,
            smoking_status
        ]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0][0]
        result = "High" if prediction > 0.7 else "Medium" if prediction > 0.4 else "Low"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
