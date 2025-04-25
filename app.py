from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and tools
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('label_encoders.pkl', 'rb'))  # Only if you're using label encoders

RISK_LABELS = ['Low', 'Medium', 'High']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Encode categorical fields
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        df = pd.DataFrame([input_data])

        # Apply label encoders if used
        for col in encoders:
            df[col] = encoders[col].transform(df[col])

        # Scale input
        scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]
        result = RISK_LABELS[predicted_class]

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run()
