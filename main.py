from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# lets load our saved model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# lets initialize our application
app = FastAPI()

# lets create our pydantic model
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chloride: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# lets create our endpoints
# home endpoints - home
@app.get('/')
def home():
    return {"message": "Welcome To Wne Quality Predictor"}

# prediction endpoint
# convert the features to 2d numpy array using [[]]
@app.post('/predict')
def predict(wine: WineFeatures):
    features = np.array([[
        wine.fixed_acidity,
        wine.volatile_acidity,
        wine.citric_acid,
        wine.residual_sugar,
        wine.chloride,
        wine.free_sulfur_dioxide,
        wine.total_sulfur_dioxide,
        wine.density,
        wine.pH,
        wine.sulphates,
        wine.alcohol
    ]])

    # lets scale our input features using the loaded scaler(to normalize the input)
    scaled_features = scaler.transform(features)

    # lets make prediction with the loaded model
    prediction = model.predict(scaled_features)

    # return the prediction - and the prediction converted to string for serializat
    return {"predicted_quality": str(prediction[0])}

# lets run our prediction app
# run app with --- uvicorn wine_app:app --reload