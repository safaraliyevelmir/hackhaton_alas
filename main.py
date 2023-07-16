from fastapi import FastAPI,UploadFile, File
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from app.models import CreditCard, Phone
from app.utils import data_proccess_credit_card, data_proccess_phone_prices
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

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

model = tf.keras.models.load_model('./models/model.h5')


@app.post('/digits')
async def upload(file: UploadFile = File(...)):

    try:
        im = Image.open(BytesIO(await file.read()))
        im = im.resize((227, 227))
        im = im.convert("RGB")
        im = np.array(im)
        im = im.reshape((1, 227, 227, 3))
        prediction = np.argmax(model.predict(im))

        print(prediction)

    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"prediction": str(prediction)}