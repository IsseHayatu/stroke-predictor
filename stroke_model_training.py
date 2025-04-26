# stroke_model_training.py (binary classification + NaN fix)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Step 1: Load dataset
df = pd.read_csv('stroke_data.csv')

# Step 2: Encode categorical features
le_gender = LabelEncoder()
le_married = LabelEncoder()
le_work = LabelEncoder()
le_residence = LabelEncoder()
le_smoke = LabelEncoder()

# Apply encoding
df['gender'] = le_gender.fit_transform(df['gender'])
df['ever_married'] = le_married.fit_transform(df['ever_married'])
df['work_type'] = le_work.fit_transform(df['work_type'])
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
df['smoking_status'] = le_smoke.fit_transform(df['smoking_status'])

# Step 3: Remove rows with NaN or inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Step 4: Separate features and label
X = df.drop(columns=['stroke'])
y = df['stroke']

# Save encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'gender': le_gender,
        'ever_married': le_married,
        'work_type': le_work,
        'Residence_type': le_residence,
        'smoking_status': le_smoke
    }, f)

# Step 5: Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 6: Use labels directly (binary classification)
y_cat = y

# Step 7: Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Step 8: Compute class weights
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# Step 9: Build binary classification model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 10: Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, class_weight=class_weights)

# Step 11: Save the model
model.save('stroke_model.h5')

print("✅ Model trained and saved as stroke_model.h5")
