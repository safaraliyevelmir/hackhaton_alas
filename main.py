from fastapi import FastAPI

import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from app.models import CreditCard, Phone
from app.utils import data_proccess_credit_card, data_proccess_phone_prices

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_credentials = True,
    allow_headers=["*"],
)   

@app.get('/')
async def root():
    return {"Alas Hackhaton":"Looosers"}

@app.post("/predict")
async def predict(credit_card: CreditCard):
    df = pd.DataFrame([credit_card.__dict__])
    prediction = data_proccess_credit_card(df)
    return {"prediction": str(prediction)}

@app.post('/price')
async def model2(data: Phone):
    df = pd.DataFrame([data.__dict__])
    prediction = data_proccess_phone_prices(df)
    return {"prediction": str(prediction)}
