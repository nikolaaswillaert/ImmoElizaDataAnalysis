# Machine learning

We start from the cleaned.csv dataset we acquired from the Data cleaning (see data-exploration/Main_DataCleaning_notebook.ipynb)

## Adding Postalcode feature
using the data/zipcode-belgium.csv file, we insert the zipcodes for the different localities

## Check the correlation of the numerical columns
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

## Feature Selection

### Numerical Features
I make a calculated decision on what features to include into the model (using top 4 numerical columns of correlation with price)
```
numerical_cols = ['price','number_rooms', 'living_area', 'surface_land', 'number_facades', 'latitude', 'longitude']
```

### Categorical Features

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

## Linear Regression model <br>
### Houses and Apartments combined <br>
![Linear Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/557a1539-09d7-424d-b6d2-8772d2d654c0)


### Houses only <br>
![Linear Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0e33b253-b933-407f-ae0c-975d8b7aa3e7)


### Apartments only <br>
![Linear Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/de73ac01-a592-443a-a0cb-4d218cbb44e9)


## K-Nearest Neighbour Regression model <br>
### Houses and Apartments combined <br>
![KNN Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/8d9a22f2-3ff5-4105-91f1-99cd06c603bd)

Using GridSearchCV to get the best parameters:

### Houses only <br>
![KNN Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/81600109-bbc5-44cc-bab0-9a2679dac0f5)


### Apartments only <br>
![KNN Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c3201f08-32f2-4c6b-8d49-1343640d5d05)


## HuberRegression model <br>
### Houses and Apartments combined <br>
![KNN Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/964e0f33-2298-4e07-b9c2-ecd521fbd12a)


### Houses only <br>
![Huber Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/d68d1833-ee54-4204-8352-b524556aeb8a)



### Apartments only <br>
![KNN Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/54a69601-4a4c-4814-95b2-b0b9bde5bea2)



## Decision Tree Regression model <br>
### Houses and Apartments combined <br>
![Decision Tree Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c0e76dea-4c01-4f28-86c5-2c2c5f8b041f)

### Houses only <br>
![Decision Tree Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/ca317511-e718-4fff-81a3-cbb380838397)


### Apartments only <br>
![Decision Tree Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/fbb694c8-e2fe-44c3-8495-ba7cd7f0c063)


## XGBoost Regression model <br>
### Houses and Apartments combined <br>
![XGBoost Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/476e3522-97bc-4d37-a2a8-683697ddde82)


### Houses only <br>
![XGBoost Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/4999f126-ce89-4e27-b79a-1341ea05a5fe)

### Apartments only <br>
![XGBoost Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0c5efffe-76fe-4d59-899c-4af39e6c61cb)

## Stochastic Gradient Descent Model <br>
### Houses and Apartments combined <br>
![SGD Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/51321729-e953-4e1c-a819-a1133a47aae6)


### Houses only <br>
![SGD Regression (House only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a46b55f5-ee21-4950-9303-65960fcd6227)


### Apartments only <br>
![SGD Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/42a85843-003e-4b8b-a945-db34d8fc4d61)


## Neural Network Regression <br>


