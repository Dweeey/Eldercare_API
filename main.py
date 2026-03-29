from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import __main__  # Fix for Colab naming issue

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

# Tell the server to use this exact class
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
@app.post("/predict_vitals")
def predict_blood_pressure(data: WatchData):
    
    # 🚨 LIVE LOGGING: Print incoming data to the Render terminal!
    print(f"📡 WATCH INCOMING! HR: {data.HR} | RMSSD: {data.RMSSD} | Age: {data.Age}")

    # Convert real Age to the PhysioNet Age Group
    if data.Age >= 85:
        age_group = 15
    elif data.Age >= 55:
        age_group = 9 + ((data.Age - 55) // 5)
    else:
        age_group = 8

    # Calculate the engineered Autonomic Ratio
    if data.RMSSD == 0:
        autonomic_ratio = 0 
    else:
        autonomic_ratio = data.HR / data.RMSSD

    features = np.array([[age_group, data.BMI, data.HR, data.RMSSD, autonomic_ratio]])

    # 🚨 THE MEDICAL GUARDRAIL 🚨
    # If the watch sends extreme, deadly numbers, bypass the AI completely.
    if data.RMSSD < 10.0 or data.HR > 130.0:
        print("⚠️ GUARDRAIL TRIGGERED: Extreme Vitals Detected!")
        return {
            "status": "success",
            "medical_assessment": "At-Risk (CRITICAL)",
            "ai_suspicion_level": "100.00% (Manual Override)"
        }
    
    # Otherwise, if the numbers are normal, ask the AI to do its math
    prediction = ai_brain.predict(features)[0]

    # Get the exact percentage of suspicion
    suspicion_percentage = ai_brain.model.predict_proba(features)[0][0] * 100

    print(f"🧠 AI ASSESSMENT: {prediction} ({suspicion_percentage:.2f}%)")

    return {
        "status": "success",
        "medical_assessment": str(prediction),
        "ai_suspicion_level": f"{suspicion_percentage:.2f}%"
    }