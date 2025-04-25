from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model = Sequential([
    Dense(32, input_shape=(10,), activation="relu"),
    Dense(3, activation="softmax")  # 3 classes: Low, Medium, High
])

model.compile(optimizer="adam", loss="categorical_crossentropy")
model.save("stroke_model.h5")
print("✅ Fast model saved as stroke_model.h5")
