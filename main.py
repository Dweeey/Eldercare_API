from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import __main__  # 🚨 Added this to fix the Colab naming issue!

# 1. Initialize the Server
app = FastAPI(title="Eldercare Hypertension API")

# 2. Recreate the custom class EXACTLY as it was in Colab
class HighSensitivityModel:
    def __init__(self, model, threshold=0.30):
        self.model = model
        self.threshold = threshold
        
    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 0]
        return np.where(probs >= self.threshold, 'At-Risk', 'Normal')

# 🚨 THE MAGIC FIX 🚨
# Tell the server: "If the .pkl file asks for __main__.HighSensitivityModel, use the class right above this!"
__main__.HighSensitivityModel = HighSensitivityModel

# Load the AI Brain into the server's memory
print("Loading AI Brain...")
try:
    ai_brain = joblib.load('clinical_hypertension_watch_brain.pkl')
    print("✅ Brain successfully loaded!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 3. Define the exact format of the data the Watch will send
class WatchData(BaseModel):
    Age: int
    BMI: float
    HR: float
    RMSSD: float

# 4. Create the API Endpoint (The "Listening" Port)
# 4. Create the API Endpoint (The "Listening" Port)
@app.post("/predict_vitals")
def predict_blood_pressure(data: WatchData):
    if data.Age >= 85:
        age_group = 15
    elif data.Age >= 55:
        age_group = 9 + ((data.Age - 55) // 5)
    else:
        age_group = 8

    if data.RMSSD == 0:
        autonomic_ratio = 0 
    else:
        autonomic_ratio = data.HR / data.RMSSD

    features = np.array([[age_group, data.BMI, data.HR, data.RMSSD, autonomic_ratio]])

    # THE MEDICAL GUARDRAIL
    # If the watch sends extreme, deadly numbers, bypass the AI completely.
    if data.RMSSD < 10.0 or data.HR > 130.0:
        return {
            "status": "success",
            "medical_assessment": "At-Risk (CRITICAL)",
            "ai_suspicion_level": "100.00% (Manual Override)"
        }
    
    # Otherwise, if the numbers are normal, ask the AI to do its math...
    prediction = ai_brain.predict(features)[0]
    # ... (rest of your code)

    # 1. Get the final text prediction (Normal or At-Risk)
    prediction = ai_brain.predict(features)[0]

    # 2. 🚨 MIND READER: Get the exact percentage of suspicion! 🚨
    # We ask the inner model for the probability of Class 0 ('At-Risk')
    suspicion_percentage = ai_brain.model.predict_proba(features)[0][0] * 100

    return {
        "status": "success",
        "medical_assessment": str(prediction),
        "ai_suspicion_level": f"{suspicion_percentage:.2f}%"
    }