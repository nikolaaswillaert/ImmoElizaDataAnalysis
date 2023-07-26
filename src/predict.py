import json
import pickle
import pandas as pd

# new_data 
def predict_price(df):
    with open('models/xgb_regression_model.pkl', 'rb') as f:
        pickle_model = pickle.load(f)
    predictions = pickle_model.predict(df)
    
    return predictions

