from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib
import os

API_KEY = os.environ["SECOM_API_KEY"]
app = FastAPI()
pipeline: Pipeline = joblib.load("model.joblib")

async def verify_api_key(x_api_key: str= Header()):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get('/health')
def health_status():
    return {'Status' : "Ok"}
    

class Input(BaseModel):
    features: list[float] = Field(min_length=590, max_length=590)

@app.post('/predict', dependencies=[Depends(verify_api_key)])
def predict(data: Input):
    df = pd.DataFrame([data.features], columns=list(map(str, range(len(data.features)))))
    prediction = pipeline.predict(df)
    return {'prediction': prediction.tolist()} #type: ignore