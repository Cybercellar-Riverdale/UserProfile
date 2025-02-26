from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import json
import logging
from xgboost import XGBClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load trained model
try:
    model = joblib.load("xgboost_model.pkl")
    if not isinstance(model, XGBClassifier):
        raise TypeError("Loaded object is not an XGBClassifier. Ensure the correct model is saved.")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

app = FastAPI()

# Define input schema
class PredictionInput(BaseModel):
    warning_word_score: float
    exclamation_score: float
    spam_word_score: float
    warning_words_found: int
    spam_words_found: int
    exclamation_patterns: int
    risk_label: str  # Requires manual encoding before prediction

@app.get("/")
def read_root():
    return {"message": "Welcome to the Email Risk Analysis API!"}

@app.get("/health")
def health_check():
    return {"status": "Server is running"}

@app.get("/sample_input")
def get_sample_input():
    try:
        with open("test_data.json", "r") as file:
            data = json.load(file)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sample data: {str(e)}")

@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        input_array = np.array([[
            input_data.warning_word_score,
            input_data.exclamation_score,
            input_data.spam_word_score,
            input_data.warning_words_found,
            input_data.spam_words_found,
            input_data.exclamation_patterns
        ]])
        
        # Make prediction
        prediction = model.predict(input_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
