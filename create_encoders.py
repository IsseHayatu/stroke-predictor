import pickle
from sklearn.preprocessing import LabelEncoder

encoders = {
    "gender": LabelEncoder().fit(["Male", "Female", "Other"]),
    "ever_married": LabelEncoder().fit(["Yes", "No"]),
    "work_type": LabelEncoder().fit(["Private", "Self-employed", "Govt_job", "children", "Never_worked"]),
    "Residence_type": LabelEncoder().fit(["Urban", "Rural"]),
    "smoking_status": LabelEncoder().fit(["formerly smoked", "never smoked", "smokes", "Unknown"])
}

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("✅ label_encoders.pkl created.")
