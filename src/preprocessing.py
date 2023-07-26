import pandas as pd
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# get new data (in json format) and preprocess data - return Dataframe
def preprocess_new_data(df):
    df.drop_duplicates()
    df = df.dropna()

    # Define the categorical columns + numerical columns
    cat_cols = ['property_type','property_subtype','kitchen','building_state','region','province']
    numerical_cols = ['number_rooms', 'living_area', 'surface_land', 'number_facades','latitude','longitude']

    # load encoder and scaler from original training
    encoder = joblib.load('models/encoder.joblib')
    scaler = joblib.load('models/scaler.joblib')

    encoded_columns = encoder.get_feature_names_out(input_features=cat_cols)
    
    X_test_enc = encoder.transform(df[cat_cols])
    X_test_enc_df = pd.DataFrame(X_test_enc.toarray(), columns=encoded_columns)

    # scale them
    X_test_scale = scaler.transform(df[numerical_cols])
    
    X_test_merged = pd.concat([pd.DataFrame(X_test_scale, columns=numerical_cols), X_test_enc_df], axis=1)

    return X_test_merged



