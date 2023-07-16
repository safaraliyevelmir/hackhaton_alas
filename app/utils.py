import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib



model_credit_card = joblib.load('./models/credit_card_pipeline.pkl')
model_phone_price = joblib.load('./models/phone_price_pipeline.pkl')



def data_proccess_credit_card(df:pd.DataFrame) -> float:
    df.columns = ['c', 'e', 'f','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u']
    LE = LabelEncoder()
    for cat in ['f', 'h', 'i']:
        df[cat] = LE.fit_transform(df[cat])

    processed_data = model_credit_card['preprocessing'].transform(df)

    # Tahmin yap
    prediction = model_credit_card['model'].predict(processed_data)
    return prediction[0]

def data_proccess_phone_prices(df: pd.DataFrame) -> float:
    df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
       'o', 'p', 'q', 'r', 's', 't']
    
    # Verileri ölçeklendir
    scaled_data = model_phone_price['scaler'].transform(df)


    prediction = model_phone_price['model'].predict(scaled_data)
    return prediction[0]