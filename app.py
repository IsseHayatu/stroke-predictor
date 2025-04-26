from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load model and scaler
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Label mapping dictionaries
gender_map = {"Male": 1, "Female": 0, "Other": 2}
married_map = {"Yes": 1, "No": 0}
residence_map = {"Urban": 1, "Rural": 0}
work_map = {
    "Private": 0,
    "Self-employed": 1,
    "Govt_job": 2,
    "children": 3,
    "Never_worked": 4
}
smoking_map = {
    "formerly smoked": 0,
    "never smoked": 1,
    "smokes": 2,
    "Unknown": 3
}

RISK_LABELS = ['Low', 'Medium', 'High']

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
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

        # Create dataframe
        df = pd.DataFrame([{
            'gender': gender_map.get(gender, 0),
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': married_map.get(ever_married, 0),
            'work_type': work_map.get(work_type, 0),
            'Residence_type': residence_map.get(Residence_type, 0),
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_map.get(smoking_status, 3)
        }])

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
    app.run(debug=True)