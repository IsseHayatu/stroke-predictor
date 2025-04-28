import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Encode categorical features
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1, 'Other': 2})
df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
df['work_type'] = df['work_type'].map({
    'Govt_job': 0, 'Private': 1, 'Self-employed': 2, 'children': 3, 'Never_worked': 4})
df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})
df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3})

# Drop missing BMI
df = df.dropna(subset=['bmi'])

# Map stroke risk (0 = Low, 1 = Medium, 2 = High)
df['stroke'] = df['stroke'].map({0: 0, 1: 2})  # You can modify based on your logic

# Define features and target
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# One-hot encode labels
y_encoded = pd.get_dummies(y).values

# Reshape for CNN
X_scaled = np.expand_dims(X_scaled, axis=2)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y)

# Save as .npz
np.savez("data.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
print("✅ Preprocessing done: scaler.pkl, data.npz created.")
print("Class distribution:\n", df['stroke'].value_counts())
