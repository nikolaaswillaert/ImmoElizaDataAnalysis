from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import pandas as pd
import joblib

def get_model():
    cat_cols = ['property_type','property_subtype','kitchen','building_state','region','province']
    use_cols = ['number_rooms', 'living_area', 'surface_land', 'number_facades','latitude','longitude']

    grid_results = {'colsample_bytree': 0.3, 'gamma': 0.0, 'learning_rate': 0.15, 'max_depth': 8, 'min_child_weight': 1}

    df = pd.read_csv('data/cleaned.csv').drop('Unnamed: 0', axis=1)

    X = df.drop(columns=['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

    transformer = ColumnTransformer(
        transformers=[
            ('onehotencoder', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('minmaxscaler', MinMaxScaler(), use_cols)
        ])

    pipeline = Pipeline([
        ('preprocessor', transformer),
        ('regressor', XGBRegressor(**grid_results))
    ])

    model = pipeline.fit(X_train, y_train)

    joblib.dump(model, 'models/xgb_model.joblib')

    y_preds = model.predict(X_test)

    r2 = r2_score(y_test, y_preds)
    print(r2)

    return model, r2

