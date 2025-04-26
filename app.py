# app.py — Flask App for Stroke Prediction (Binary Classification)

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle

app = Flask(__name__)

# Load model and tools
model = load_model('stroke_model.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = label_encoders['gender'].transform([request.form['gender']])[0]
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = label_encoders['ever_married'].transform([request.form['ever_married']])[0]
        work_type = label_encoders['work_type'].transform([request.form['work_type']])[0]
        residence_type = label_encoders['Residence_type'].transform([request.form['residence_type']])[0]
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = label_encoders['smoking_status'].transform([request.form['smoking_status']])[0]

        input_data = np.array([[gender, age, hypertension, heart_disease, ever_married,
                                work_type, residence_type, avg_glucose_level, bmi, smoking_status]])

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        predicted_class = int(prediction[0][0] >= 0.5)  # 0 or 1
        result = "Low" if predicted_class == 0 else "High"

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
