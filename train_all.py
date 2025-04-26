# train_all.py – Train and save model, scaler, encoders
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load data
df = pd.read_csv('stroke_data.csv')

# Encode categoricals
encoders = {
    'gender': LabelEncoder(),
    'ever_married': LabelEncoder(),
    'work_type': LabelEncoder(),
    'Residence_type': LabelEncoder(),
    'smoking_status': LabelEncoder()
}

for col, le in encoders.items():
    df[col] = le.fit_transform(df[col])

# Drop NaNs and 'id'
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['stroke', 'id'])
y = df['stroke']

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, class_weight=class_weights)

# Save model
model.save('stroke_model.h5')
print("✅ Model, scaler, and encoders saved successfully!")
