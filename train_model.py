import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
data = np.load("data.npz")
X_train, X_val = data["X_train"], data["X_val"]
y_train_raw = data["y_train"]
y_val_raw = data["y_val"]

# If one-hot encoded already, just keep them
if len(y_train_raw.shape) == 1:
    y_train_labels = y_train_raw
    y_val_labels = y_val_raw
else:
    y_train_labels = np.argmax(y_train_raw, axis=1)
    y_val_labels = np.argmax(y_val_raw, axis=1)

# Class weights
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights_array))

# Convert to binary labels (if not already)
if len(y_train_raw.shape) == 1:
    y_train = y_train_raw
    y_val = y_val_raw
else:
    y_train = y_train_labels
    y_val = y_val_labels

# Model (binary classification)
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    class_weight=class_weights,
    verbose=1
)

model.save("stroke_model.h5")
print("✅ Model trained and saved as stroke_model.h5")
