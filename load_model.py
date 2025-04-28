import os
import pickle
from tensorflow.keras.models import load_model

# Define the absolute path to the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to your files
MODEL_PATH = os.path.join(BASE_DIR, "stroke_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")  # Only if used

# Load model and other files
model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))
encoders = pickle.load(open(ENCODERS_PATH, "rb"))  # Comment out if not used
