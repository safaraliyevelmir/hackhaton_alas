import pandas as pd
import joblib



model_credit_card = joblib.load('./models/credit_card_pipeline.pkl')
model_phone_price = joblib.load('./models/phone_price_pipeline.pkl')

def data_proccess_credit_card(df:pd.DataFrame) -> float:
    df.columns = ['c', 'e', 'f','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u']
    processed_data = model_credit_card['preprocessing'].transform(df)
    prediction = model_credit_card['model'].predict(processed_data)
    return prediction[0]

def data_proccess_phone_prices(df: pd.DataFrame) -> float:
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
       'o', 'p', 'q', 'r', 's', 't']
    
    scaled_data = model_phone_price['scaler'].transform(df)


    prediction = model_phone_price['model'].predict(scaled_data)
    return prediction[0]