import sklearn.metrics as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

def linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model):
        print(f"Mean absolute error TEST = {round(sm.mean_absolute_error(y_test, y_preds), 4)}\n")
        print(f"Mean squared error TEST = {round(sm.mean_squared_error(y_test, y_preds), 4)}\n") 
        print(f"Median absolute error TEST = {round(sm.median_absolute_error(y_test, y_preds), 4)}\n")
        print(f"Explain variance score TEST = {round(sm.explained_variance_score(y_test, y_preds), 4)}\n")
        print(f"R2 score *coefficient of Determination TEST = {round(sm.r2_score(y_test, y_preds), 4)}\n")
        print('--------------------------------------')
        print(f'TRAINING SCORE: {model.score(X_train, y_train)}')
        print(f'TESTING SCORE: {model.score(X_test, y_test)}')
        print('--------------------------------------')
        scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=10)
        print(f'Cross validation scores: \n {scores}')   
        plt.scatter(X_test['living_area'], y_test, marker='+', alpha=0.3, c='blue', label='Actual')
        plt.scatter(X_test['living_area'], y_preds, marker='o', alpha=0.3, c='red', label='Predicted')
        plt.xlabel('Living Area')
        plt.ylabel('Price')
        plt.legend()


def scale_data(X_train, X_test):
        scaler = MinMaxScaler()
        norm_x_train = scaler.fit_transform(X_train)
        norm_x_test = scaler.transform(X_test)

        df_train_scaled = pd.DataFrame(norm_x_train, columns=X_train.columns)
        df_test_scaled = pd.DataFrame(norm_x_test, columns=X_test.columns)

        return df_train_scaled, df_test_scaled

def train_linear_regr(X, y):
        # Scale the Numerical Data with MinMax Scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)

        X_train, X_test = scale_data(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model)


def train_knn_regr(X, y, **params):
        # Scale the Numerical Data with MinMax Scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)

        X_train, X_test = scale_data(X_train, X_test)

        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model)

        return model, X_train, X_test, y_train, y_test


def train_polynomial_regr(X, y, degree):
        steps = [('polynomial', PolynomialFeatures(degree=degree)), ('linearregres', LinearRegression())]
        pipe = Pipeline(steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        pipe.fit(X_train, y_train)
        y_preds = pipe.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, pipe)

def train_huberregressor(X,y):
        model = HuberRegressor(max_iter=1000)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model)

def train_decessiontree_regression(X, y):
        model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model)

def train_XGBoost_regression(X, y):
        model = xgb.XGBRegressor(objective="reg:squarederror")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model)
        