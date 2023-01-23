import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

model_file = 'XGB_model.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

features = ['homeplanet', 'cryosleep', 'cabin', 'destination', 'age',
            'vip', 'roomservice', 'foodcourt', 'shoppingmall', 'spa', 'vrdeck']

class Passenger(BaseModel):
    homeplanet: Optional[str] = None
    cryosleep: Optional[float] = None
    cabin: Optional[str]= None
    destination: Optional[str] = None
    age: Optional[float]= None
    vip: Optional[float] = None
    roomservice: Optional[float] = None
    foodcourt: Optional[float] = None
    shoppingmall: Optional[float] = None
    spa: Optional[float] = None
    vrdeck: Optional[float] = None

    class Config:
      schema_extra = {
        "example": {
          "homeplanet": "europa",
          "cryosleep": 0,
          "cabin": "e/608/s",
          "destination": "55_cancri_e",
          "age": 32,
          "vip": 0,
          "roomservice": 0,
          "foodcourt": 1049,
          "shoppingmall": 0,
          "spa": 353,
          "vrdeck": 3235
        }            
      }

app = FastAPI()
@app.get("/ping")
def ping():
    return {"message": "PONG"}

@app.post("/predict")
def predict(passenger: Passenger):

    passenger_dict = passenger.dict()
    X = dv.transform(passenger_dict)
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())
    y_pred = model.predict(dX)

    result = {
        'Transported': float(y_pred)
    }
    
    return result