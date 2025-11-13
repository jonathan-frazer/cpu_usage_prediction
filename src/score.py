import json
import joblib
import pandas as pd
import os

def init():
    global model
    # Azure automatically copies model into this folder
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)
    print("Model loaded from:", model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
