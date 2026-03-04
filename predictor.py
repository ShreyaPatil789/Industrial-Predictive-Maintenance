import joblib
import pandas as pd
from config import MODEL_PATH, PREDICTION_THRESHOLD

# Load model once (production style)
model = joblib.load(MODEL_PATH)


def predict_machine_failure(input_data: dict):

    """
    input_data should be a dictionary like:

    {
        "Air temperature [K]": 300,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 120,
        "Type_L": 0,
        "Type_M": 1
    }
    """

    df = pd.DataFrame([input_data])

    probability = float(model.predict_proba(df)[0][1])
    prediction = int(probability >= PREDICTION_THRESHOLD)

    result = {
        "failure_probability": round(probability, 4),
        "threshold_used": PREDICTION_THRESHOLD,
        "prediction": prediction,
        "alert": "⚠️ Maintenance Required" if prediction == 1 else "✅ Machine Healthy"
    }

    return result