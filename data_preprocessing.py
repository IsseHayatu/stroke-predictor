import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv("stroke_data.csv")
df['risk'] = df['risk'].map({'low': 0, 'medium': 1, 'high': 2})

X = df.drop('risk', axis=1)
y = df['risk']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_scaled = np.expand_dims(X_scaled, axis=2)
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
np.savez("data.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
