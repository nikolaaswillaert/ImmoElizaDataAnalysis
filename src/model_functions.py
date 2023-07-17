import sklearn.metrics as sm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor

def linear_evaluation(X_test, y_test, y_preds):
        print(f"Mean absolute error = {round(sm.mean_absolute_error(y_test, y_preds), 2)}\n")
        print(f"Mean squared error = {round(sm.mean_squared_error(y_test, y_preds), 2)}\n") 
        print(f"Median absolute error = {round(sm.median_absolute_error(y_test, y_preds), 2)}\n")
        print(f"Explain variance score = {round(sm.explained_variance_score(y_test, y_preds), 2)}\n")
        print(f"R2 score *coefficient of Determination = {round(sm.r2_score(y_test, y_preds), 2)}\n")
        print('\n')

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        X_train, X_test = scale_data(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_test, y_test, y_preds)


def train_knn_regr(X, y, **params):
        # Scale the Numerical Data with MinMax Scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        X_train, X_test = scale_data(X_train, X_test)

        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_test, y_test, y_preds)

        return model, X_train, X_test, y_train, y_test


def train_polynomial_regr(X, y, degree):
        steps = [('polynomial', PolynomialFeatures(degree=degree)), ('linearregres', LinearRegression())]
        pipe = Pipeline(steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        X_train, X_test = scale_data(X_train, X_test)

        pipe.fit(X_train, y_train)
        y_preds = pipe.predict(X_test)

        linear_evaluation(X_test, y_test, y_preds)

def train_huberregressor(X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
        X_train, X_test = scale_data(X_train, X_test)

        model = HuberRegressor(max_iter=1000)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_test, y_test, y_preds)