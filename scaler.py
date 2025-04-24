import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# ⚠️ Use your actual dataset shape if known
# Example assumes 10 input features and 100 samples
dummy_data = np.random.rand(100, 10)

scaler = StandardScaler()
scaler.fit(dummy_data)

# Save the scaler to file
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ scaler.pkl created successfully.")
