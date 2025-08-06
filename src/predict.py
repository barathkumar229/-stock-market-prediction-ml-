import pandas as pd
import joblib

def predict_new(data_point):
    model = joblib.load('../models/stock_model.pkl')
    prediction = model.predict([data_point])
    return prediction
