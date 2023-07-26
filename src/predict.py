import json
import pickle
import pandas as pd
from preprocessing import preprocess_new_data

def predict_new_data(new_data):
    data_dict = json.loads(new_data)
    df = pd.DataFrame(data_dict)
    with open('models/xgb_regression_model.pkl', 'rb') as f:
        pickle_model = pickle.load(f)
    #new data is in json format
    predictions = pickle_model.predict(df)
    return predictions

# new_data = pd.read_csv('data/cleaned.csv')
# json_data = new_data.to_json()

# new_data = preprocess_new_data(json_data)
# predict_new_data(new_data)
