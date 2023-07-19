# :chart_with_upwards_trend: Model Price prediction :chart_with_upwards_trend:

We start from the cleaned.csv dataset we acquired from the Data cleaning (see data-exploration/Main_DataCleaning_notebook.ipynb)

## :pencil2: Adding Postalcode feature
using the data/zipcode-belgium.csv file, we insert the zipcodes for the different localities

## :pencil2: Check the correlation of the numerical columns
```df[numerical_cols].corr()['price'].sort_values(ascending=False)```
```
price             1.000000
living_area       0.520740
number_rooms      0.409525
surface_land      0.237257
number_facades    0.175345
fireplace         0.136150
garden            0.134182
terrace_area      0.117371
terrace           0.113510
swimming_pool     0.096904
garden_area       0.092524
latitude          0.007480
furnished        -0.004397
longitude        -0.068915
Name: price, dtype: float64
```

## :pencil2: Feature Selection

### :bookmark: Numerical Features :bookmark:
I make a calculated decision on what features to include into the model (using top 4 numerical columns of correlation with price)
```
numerical_cols = ['price','number_rooms', 'living_area', 'surface_land', 'number_facades', 'latitude', 'longitude']
```

### :bookmark: Categorical Features :bookmark:

```
cat_cols = ['property_type','property_subtype','kitchen','building_state','region','province',]
```

### Define the Features(X) and target(y)
X - all columns except for the price <br>
y - target column = price
```
X = df[numerical_cols].drop(columns=['price'], axis=1)
y = df['price']
```
## Overview of the results
The best model that was trained was the XGBoost model (R2 score of 0.8) <br>
![Pasted image](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c45ff30e-4b5e-4e52-8f88-9835500a6acd)

# Results
## :cyclone: Linear Regression model <br>
### Houses and Apartments combined <br>
![Linear Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/557a1539-09d7-424d-b6d2-8772d2d654c0)


### Houses only <br>
![Linear Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0e33b253-b933-407f-ae0c-975d8b7aa3e7)


### Apartments only <br>
![Linear Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/de73ac01-a592-443a-a0cb-4d218cbb44e9)


## :cyclone: K-Nearest Neighbour Regression model <br>
### Houses and Apartments combined <br>
Using GridSearchCV to get the best parameters:<br>
![KNN Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/212e60d9-4c80-40d1-a4e6-6696a5aee941)


### Houses only <br>
![KNN Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/81600109-bbc5-44cc-bab0-9a2679dac0f5)


### Apartments only <br>
![KNN Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c3201f08-32f2-4c6b-8d49-1343640d5d05)


## :cyclone: HuberRegression model <br>
### Houses and Apartments combined <br>
![Huber Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/21b0bc47-e6ae-452e-b9d9-ef8a559fbbd3)

### Houses only <br>
![Huber Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/b3605c7c-eb0c-4542-8ee0-59511e0f9ed5)

### Apartments only <br>
![Huber Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/ade691f5-d916-41d9-b2e5-65d03aa673d5)


## :cyclone: Decision Tree Regression model <br>
### Houses and Apartments combined <br>
![Decision Tree Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/af56184b-e9e2-4fb9-bb3c-6b04a3871e2c)

### Houses only <br>
![Decision Tree Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/9ccbb3aa-4bda-41a8-989b-14510eb178c5)


### Apartments only <br>
![Decision Tree Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c195ad28-dcc9-45a4-a3b7-10209754da15)


## :cyclone: XGBoost Regression model <br>
### Houses and Apartments combined <br>
![XGBoost Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/476e3522-97bc-4d37-a2a8-683697ddde82)


### Houses only <br>
![XGBoost Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/4999f126-ce89-4e27-b79a-1341ea05a5fe)

### Apartments only <br>
![XGBoost Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0c5efffe-76fe-4d59-899c-4af39e6c61cb)

## :cyclone: Stochastic Gradient Descent Model <br>
### Houses and Apartments combined <br>
![SGD Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/19c2d46f-8b4f-45f8-8af8-c4f4cf021427)

### Houses only <br>
![SGD Regression (House only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/4820df40-448b-4d90-be6e-533131cc700b)

### Apartments only <br>
![SGD Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0e095438-0821-49ff-9860-2b886c5e3101)

## :cyclone: Neural Network Regression <br>
run the following command in the terminal to load tensorboard:
```
tensorboard --logdir=/log/train
``` 
Each neural network was trained for 300 epochs with a batch_size 8 <br>
```
epochs = 300
batch_size = 8
train_neural_network(X,y, epochs, batch_size)
```
### Houses and Apartments combined <br>
R2 score= 0.62 <br>
loss: 74859.0938 <br>
![Tensorboard](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/654210ba-b050-4832-9d62-1a6b45892185)

### Houses only <br>
R2 score = 0.62 <br>
loss: 77604.0625 <br>
![Tensorboard 3](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/e00058e8-26e8-49f8-ab92-a4937cb7ca6e)

### Apartments only <br>
R2 score = 0.42 <br>
loss: 77865.6250 <br>
![Tensorboard 2](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/25085629-a1e6-49e8-9677-9a9f0138d3b1)

