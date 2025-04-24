from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

# Dummy input shape (adjust if your real dataset is different)
INPUT_DIM = 10  # 10 features in your form

# Create a simple CNN-like model (for tabular data, Dense layers are fine)
model = Sequential([
    Dense(64, activation='relu', input_shape=(INPUT_DIM,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Low, Medium, High
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save empty (initialized) model
model.save("stroke_model.h5")

# Dummy scaler (just for interface compatibility)
scaler = StandardScaler()
dummy_data = np.random.rand(100, INPUT_DIM)
scaler.fit(dummy_data)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
