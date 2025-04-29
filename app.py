from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Initialize the app
app = Flask(__name__)

# Load model and scaler
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect inputs
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])  # NOTE: keep capital R here
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Input order must match scaler training order
        input_data = pd.DataFrame([[
            gender, age, hypertension, heart_disease, ever_married,
            work_type, Residence_type, avg_glucose_level, bmi, smoking_status
        ]], columns=[
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ])

        # Scale input and reshape
        input_scaled = scaler.transform(input_data)
        reshaped = input_scaled.reshape((1, 10, 1))

        # Predict
        prediction = model.predict(reshaped, batch_size=1)
        risk = np.argmax(prediction, axis=1)[0]
        risk_label = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

        return render_template('index.html', prediction=f'Stroke Risk: {risk_label[risk]}')

    except Exception as e:
        return render_template('index.html', prediction=f"An error occurred during prediction: {str(e)}")

# Run app
if __name__ == "__main__":
    app.run(debug=True)
