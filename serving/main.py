from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

app = FastAPI()
pipeline: Pipeline = joblib.load("model.joblib")

@app.get('/health')
def health_status():
    return {'Status' : "Ok"}
    

class Input(BaseModel):
    features: list[float] = Field(min_length=590, max_length=590)

@app.post('/predict')
def predict(data: Input):
    df = pd.DataFrame([data.features], columns=list(map(str, range(len(data.features)))))
    prediction = pipeline.predict(df)
    return {'prediction': prediction.tolist()}