import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model and scaler
model = load_model("stroke_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# Example high-risk input:
# [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]
test_input = [[1.0, 85.0, 1.0, 1.0, 1.0, 2.0, 1.0, 300.0, 45.0, 2.0]]  # high-risk

# Scale and reshape
scaled = scaler.transform(test_input)
reshaped = np.expand_dims(scaled, axis=2)

# Predict
prediction = model.predict(reshaped)
risk_levels = ["Low", "Medium", "High"]
predicted_class = risk_levels[np.argmax(prediction)]

# Output
print("Class probabilities:", prediction)
print("Predicted Stroke Risk:", predicted_class)
