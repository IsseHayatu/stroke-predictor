import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Drop 'id' column (not useful)
df = df.drop(columns=["id"])

# Encode categorical columns
categorical_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders (optional for Flask app later)
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Separate features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for CNN input
X_scaled = np.expand_dims(X_scaled, axis=2)

# Train/Validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save datasets
np.savez("data.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

print("✅ Data preprocessing complete.")
