import time
import datetime
import pandas as pd
import sklearn.metrics as sm
import src.model_functions as mf
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score

def grid_search_xgb(X_train, y_train):
  params = {
        'learning_rate': [0.05, 0.10, 0.15],
        'max_depth': [7,8,10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.0, 0.1, 0.2],
        'colsample_bytree': [0.3, 0.4],
    }
  
  gsc = GridSearchCV(
    estimator=XGBRegressor(),
    param_grid=params,
    cv=5,
    scoring='r2',
    verbose=0,
    n_jobs=-1)
  
  grid_result = gsc.fit(X_train, y_train)
  best_params = grid_result.best_params_
  return grid_result, best_params

def preprocess_data():
    # Add postcodes to the table
    postcodes = pd.read_csv('data/zipcode-belgium.csv')
    postcodes = postcodes.drop(columns=['lat', 'long'])
    postcodes.head()

    # Load in cleaned dataset
    df = pd.read_csv('data/cleaned.csv').drop('Unnamed: 0', axis=1)
    df.drop_duplicates()
    # merge postalcodes with the cleaned dataset
    postalcode_merge_df = pd.merge(postcodes, df, on='locality', how='left')
    postalcode_merge_df.drop_duplicates()
    df = postalcode_merge_df.dropna()

    # Define the categorical columns + numerical columns
    cat_cols = ['property_type','property_subtype','kitchen','building_state','region','province']
    numerical_cols = ['price','number_rooms', 'living_area', 'surface_land', 'number_facades','latitude','longitude']

    # create dummies van categorical columns
    dummies = pd.get_dummies(df[cat_cols], columns=cat_cols)

    new_df = pd.concat([df[numerical_cols], dummies], axis=1)
    new_df.reset_index().drop(columns=['index'], inplace=True)
    return new_df

if __name__ == '__main__':
    new_df = preprocess_data()
    X = new_df.drop(columns=['price'], axis=1)
    y = new_df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)
    print('--------------------------------------')
    print("Getting the best parameters for Xgboost through GridSearchCV")
    print('--------------------------------------')
    start_time = time.time()
    grid_results, best_params = grid_search_xgb(X_train, y_train)
    print('--------------------------------------')
    print(f"Best Parameters (from GridSearchCV): {best_params}")
    print('--------------------------------------')
    end_time = time.time()
    print(f'Elapsed time to get best parameters: {round(((end_time - start_time)/60), 3)} minutes')

    start_time = time.time()
    print('--------------------------------------')
    print("Re-training model with best parameters . . .")
    y_test, y_preds, model = mf.train_XGBoost_regression(X, y, 'XGBoost - GridSearchTrain', **best_params)
    end_time = time.time()
    print(f'Elapsed time to (re)train with best parameters: {round(((end_time - start_time)/60), 3)} minutes')
    print('--------------------------------------')
    with open('XGB_best_model_details.txt', 'a') as f:
       f.write(datetime.datetime.now())
       f.write('XGBOOST REGRESSION - HYPERPARAMETER TUNING')
       f.write('Best parameters:')
       f.write(f'{best_params}')
       f.write(f'-------------------------------')
       f.write(f"Mean absolute error = {round(sm.mean_absolute_error(y_test, y_preds), 4)}")
       f.write(f"Mean squared error = {round(sm.mean_squared_error(y_test, y_preds), 4)}") 
       f.write(f"Median absolute error = {round(sm.median_absolute_error(y_test, y_preds), 4)}")
       f.write(f"Explain variance score = {round(sm.explained_variance_score(y_test, y_preds), 4)}")
       f.write(f"R2 score *coefficient of Determination = {round(sm.r2_score(y_test, y_preds), 4)}")
       f.write('--------------------------------------')
       f.write(f'TRAINING SCORE: {model.score(X_train, y_train)}')
       f.write(f'TESTING SCORE: {model.score(X_test, y_test)}')
       f.write('--------------------------------------')
       kfold = KFold(n_splits = 5)
       scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kfold)
       f.write(f'Cross validation scores: \n {scores}') 