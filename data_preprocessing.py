import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("stroke_data.csv")

# Print unique values in the 'stroke' column to help debug
print('Unique values in stroke column before cleaning:')
print(df['stroke'].unique())

# Standardize 'stroke' column text (lowercase everything and remove extra spaces)
df['stroke'] = df['stroke'].astype(str).str.strip().str.lower()

# Print unique values after cleaning
print('Unique values in stroke column after cleaning:')
print(df['stroke'].unique())

# Only keep rows with valid labels
valid_labels = ['low', 'medium', 'high']
df = df[df['stroke'].isin(valid_labels)]

# Print how many rows are left after cleaning
print('Remaining rows after cleaning:')
print(df.shape)

# Map 'stroke' to numeric 'risk' column
df['risk'] = df['stroke'].map({'low': 0, 'medium': 1, 'high': 2})
df.drop('stroke', axis=1, inplace=True)

# Encode all categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data into features (X) and labels (y)
X = df.drop('risk', axis=1)
y = df['risk']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler and encoders
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Reshape data for CNN model (add a third dimension)
X_scaled = np.expand_dims(X_scaled, axis=2)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Save the preprocessed data
np.savez("data.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)

print("Preprocessing complete!")
