from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Model and scaler will be loaded inside the predict function to prevent startup overload

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

def safe_int(value, default=0):
    try:
        return int(value)
    except:
        return default

def safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/ping')
def ping():
    return "✅ App is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("📥 Received POST /predict")
        model = load_model('stroke_model.h5')
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        # Extract and safely convert form data
        gender = request.form['gender']
        age = safe_float(request.form['age'])
        hypertension = safe_int(request.form['hypertension'])
        heart_disease = safe_int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = safe_float(request.form['avg_glucose_level'])
        bmi = safe_float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

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

        print("🔄 DataFrame created")
        print(df)

        scaled = scaler.transform(df)
        print("📏 Scaled input")

        prediction = model.predict(scaled)
        print("✅ Model predicted:", prediction)

        predicted_class = np.argmax(prediction, axis=1)[0]
        result = RISK_LABELS[predicted_class]

        return render_template('result.html', prediction=result)

    except Exception as e:
        import traceback
        print("❌ Error:", traceback.format_exc())
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
