# stroke_model_training.py

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

df['gender'] = le_gender.fit_transform(df['gender'])
df['ever_married'] = le_married.fit_transform(df['ever_married'])
df['work_type'] = le_work.fit_transform(df['work_type'])
df['Residence_type'] = le_residence.fit_transform(df['Residence_type'])
df['smoking_status'] = le_smoke.fit_transform(df['smoking_status'])

# Step 3: Remove rows with NaN or inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Step 4: Drop ID column and separate features/labels
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# Save label encoders
with open('encoders.pkl', 'wb') as f:
    pickle.dump({
        'gender': le_gender,
        'ever_married': le_married,
        'work_type': le_work,
        'Residence_type': le_residence,
        'smoking_status': le_smoke
    }, f)

# Step 5: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 6: Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Class weight handling
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# Step 8: Build the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, class_weight=class_weights)

# Step 10: Save the model
model.save('stroke_model.h5')
print("✅ Model trained and saved as stroke_model.h5")
