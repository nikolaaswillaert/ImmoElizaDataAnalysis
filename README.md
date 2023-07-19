# Linear models

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


## K-Nearest Neighbour Regression model <br>

## HuberRegression model <br>

## Decision Tree Regression model <br>

## XGBoost Regression model <br>

## Neural Network Regression <br>


