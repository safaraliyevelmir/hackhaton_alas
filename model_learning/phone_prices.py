import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib


# Load Dataset
data = pd.read_csv("./datas/mobile_phones_Data.csv")


# to allocate the target column of data
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Scaling
    ('model', LogisticRegression()) # Create Model
])


# Split data train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


# Fit pipeline
pipeline.fit(x_train, y_train)

# Create prediction from model
y_pred = pipeline.predict(x_test)

# Test model Accuracy
accuracy = accuracy_score(y_test, y_pred) 

print("Accuracy:", accuracy)

# Save Model
joblib.dump(pipeline, 'models/phone_price_pipeline.pkl')
