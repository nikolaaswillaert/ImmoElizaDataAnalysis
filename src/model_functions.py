import pandas as pd
from xgboost import XGBRegressor
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold

def linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title):
        print(f"General metrics for Linear models: ")
        print('--------------------------------------')
        print(f"Mean absolute error = {round(sm.mean_absolute_error(y_test, y_preds), 4)}")
        print(f"Mean squared error = {round(sm.mean_squared_error(y_test, y_preds), 4)}") 
        print(f"Median absolute error = {round(sm.median_absolute_error(y_test, y_preds), 4)}")
        print(f"Explain variance score = {round(sm.explained_variance_score(y_test, y_preds), 4)}")
        print(f"R2 score *coefficient of Determination = {round(sm.r2_score(y_test, y_preds), 4)}")
        print('--------------------------------------')
        print(f'TRAINING SCORE: {model.score(X_train, y_train)}')
        print(f'TESTING SCORE: {model.score(X_test, y_test)}')
        print('--------------------------------------')
        kfold = KFold(n_splits = 5)
        scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kfold)
        print(f'Cross validation scores: \n {scores}') 

        residuals = y_test - y_preds
        plt.scatter(y_test, residuals)
        plt.axhline(y=0, color='r', linestyle='--')  # Adding a horizontal line at y=0 for reference
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.title(f'{title} Residuals')
        plt.show()
        plt.tight_layout()
        plt.savefig(f'output/model_graphs/{title}.png', format='png')

        sns.regplot(x=y_test, y=y_preds, scatter_kws={"alpha": 0.45}, line_kws={"color": "red"})
        plt.axis('equal')
        plt.xlabel('Actual price values')
        plt.ylabel('Predicted Price Values')
        plt.title(f'{title} - R2 score: {round(sm.r2_score(y_test, y_preds), 4)}')
        print("--------------------------------------")
        print("Saving the graph in output/model_graphs")
        print("--------------------------------------")
        plt.tight_layout()
        plt.savefig(f'output/model_graphs/{title}.png', format='png')
        plt.show()

def neural_network_eval(y_test, y_preds, loss):
        print(f"R2 score *coefficient of Determination TEST = {round(sm.r2_score(y_test, y_preds), 4)}\n")
        print(f"loss = {loss}")

def scale_data(X_train, X_test):
        scaler = MinMaxScaler()
        norm_x_train = scaler.fit_transform(X_train)
        norm_x_test = scaler.transform(X_test)

        df_train_scaled = pd.DataFrame(norm_x_train, columns=X_train.columns)
        df_test_scaled = pd.DataFrame(norm_x_test, columns=X_test.columns)

        return df_train_scaled, df_test_scaled

def train_linear_regr(X, y, title):
        # Scale the Numerical Data with MinMax Scaler
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)

        X_train, X_test = scale_data(X_train, X_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)
        return y_preds, y_test

def train_knn_regr(X, y, title, **params):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        # Scale the Numerical Data with MinMax Scaler
        X_train, X_test = scale_data(X_train, X_test)

        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)

        return model, X_train, X_test, y_train, y_test


def train_polynomial_regr(X, y, degree, title):
        steps = [('polynomial', PolynomialFeatures(degree=degree)), ('linearregres', LinearRegression())]
        pipe = Pipeline(steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        pipe.fit(X_train, y_train)
        y_preds = pipe.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, pipe, title)

def train_huberregressor(X,y, title):
        model = HuberRegressor(max_iter=1000)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)
        
        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)

def train_decessiontree_regression(X, y, title):
        
        model = DecisionTreeRegressor()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)

def train_XGBoost_regression(X, y, title, **params):
        model = XGBRegressor(**params)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)

        return y_test, y_preds, model, X_train, y_train, X_test

def train_SGDregressor(X, y, title):
        model = SGDRegressor(max_iter=1000, tol=1e-3)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train)
        y_preds = model.predict(X_test)

        linear_evaluation(X_train, X_test, y_train, y_test, y_preds, model, title)

def train_neural_network(X,y, epochs, batch_size):
        log_dir = './log/' 
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        checkpoint_callback = ModelCheckpoint(filepath='./log/best_model.h5', monitor='loss', save_best_only=True)
        early_stopping_callback = EarlyStopping(monitor='loss', patience=10)

        num_features = len(X.columns)
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(num_features,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear')) 

        model.compile(loss='mean_absolute_error', optimizer='adam')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=22)
        X_train, X_test = scale_data(X_train, X_test)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback])

        loss = model.evaluate(X_test, y_test)
        y_preds = model.predict(X_test)
        neural_network_eval(y_test, y_preds, loss)