from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import joblib

# Download Dataset
df = pd.read_csv("./datas/credit_card_churn_data.csv")
df.drop(['a','w','v','d','g'], axis=1, inplace=True)

# Variable lists to be standardized and normalized
stand_var = ['c', 'j', 's', 't']
norm_var = ['e', 'k', 'l', 'm', 'o', 'p', 'q', 'r', 'u']

# Label Encoding
LE = LabelEncoder()
for cat in ['b', 'f', 'h', 'i']:
    df[cat] = LE.fit_transform(df[cat])

# to allocate the target column of data
X = df.drop(['b'], axis=1).astype(float).values
y = df['b'].astype(float).values

# Create Pipline
preprocessing_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standard Scaling
    ('normalizer', MinMaxScaler()),  # Normalize data
    ('label_encoder', LabelEncoder()),  # Encoding

])

pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('model', GradientBoostingClassifier())  # Create Model
])

# Split data test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit Pipeline
pipeline.fit(X_train, y_train)

# make predictions on test data
y_pred = pipeline.predict(X_test)

# Calculate the truth value of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Save Model
joblib.dump(pipeline, './models/credit_card_pipeline.pkl')
