import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# get new data (in json format) and preprocess data - return Dataframe
def preprocess_new_data(json_data):
    data_dict = json.loads(json_data)
    df = pd.DataFrame(data_dict)

    df.drop_duplicates()
    # Read in and clean postcode dataframe 
    postcodes = pd.read_csv('data/zipcode-belgium.csv')
    postcodes = postcodes.drop(columns=['lat', 'long'])
    postcodes.head()

    # merge postalcodes with the cleaned dataset
    postalcode_merge_df = pd.merge(postcodes, df, on='locality', how='left')
    postalcode_merge_df.drop_duplicates()
    df = postalcode_merge_df.dropna()

    # Define the categorical columns + numerical columns
    cat_cols = ['property_type','property_subtype','kitchen','building_state','region','province']
    numerical_cols = ['price','number_rooms', 'living_area', 'surface_land', 'number_facades','latitude','longitude']

    # create dummies van categorical columns
    dummies = pd.get_dummies(df[cat_cols], columns=cat_cols)

    # merge the dummies and numerical cols
    new_df = pd.concat([df[numerical_cols], dummies], axis=1)
    new_df.reset_index().drop(columns=['index'], inplace=True)

    X = new_df.drop(columns=['price'], axis=1)
    y = new_df['price']

    # Scale the X 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df_new_data_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # convert to json
    result_data_json = df_new_data_scaled.to_json()
    return result_data_json


# json_data = pd.read_csv('data/cleaned.csv').to_json()
# H = preprocess_new_data(json_data)
# print(H)
# print(type(H))
# data_dict = json.loads(H)
# df = pd.DataFrame(data_dict)
# print(df)