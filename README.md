# :chart_with_upwards_trend: Model Price prediction :chart_with_upwards_trend:

## :information_source: Installation
Install the required dependencies using the following command: <br>
```
pip install -r requirements.txt
```
Run the **main.py** file to get train best possible model and the results will be written in 'output/XGB_best_model_details.txt')
This model will be trained with the best hyperparameters (acquired through GridSearchCV)

## :information_source: Overview
We start from the **cleaned.csv** dataset we acquired from the Data cleaning (see data-exploration/Main_DataCleaning_notebook.ipynb)
The models that have been used to **predict pricing based on several features**:
 - Linear Regression
 - K-Nearest Neighbour Regression
 - Huber Regression
 - Decision Tree Regression
 - Xgboost Regression
 - Stochastic Gradient Descent Regression
 - Neural Network (Keras)

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

When doing the data visualisation of the dataset (see data-exploration/Visualisation_Notebook.ipynb), we quickly get a grasp on how the data is structured and what features might have the most influence on the total property price. Taking these visualisations and knowledge into account we select a select set of features

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
The best performing model is the XGBoost Regression model (R2 score of 0.8). With hyperparameter tuning the R2 result had increased to 0.81 <br>
**Note:** The Neural network has not been included in this table
![Pasted image](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c45ff30e-4b5e-4e52-8f88-9835500a6acd)

**XGBoost Regression**

:pushpin: **Mean Absolute Error (MAE)** is 54885.9907: This indicates that, on average, the model's predictions differ from the actual values by approximately 54042.1821 units. Lower MAE values indicate better accuracy.

:pushpin: **Explained Variance Score** is 0.8035: The explained variance score assesses the proportion of variance in the target variable that the model captures. A value of 0.8074 indicates that the model explains 80.74% of the total variance in the data, which is quite good.

:pushpin: **R-squared (R2)** Score is 0.8034: The R2 score, also known as the coefficient of determination, is the same as the explained variance score in this case. It indicates that 80.74% of the variance in the target variable is explained by the model.

:card_index: In summary, the model achieves a **high level of accuracy**, capturing a significant portion of the variance in the data and making predictions that are **close to the actual values**.

:card_index: Regarding its training and testing performance, the **model has been trained well with a training score of 0.9253**, suggesting that it has learned from the training data effectively. When evaluated on the testing data, it performs very well with a **testing score of 0.8074**, which is **consistent with the R2 score** mentioned earlier.

:card_index: Finally, **the cross-validation scores** further validate the model's **generalization ability**, as all the scores are relatively high and close to the testing score. This means the **model is robust and reliable**, and it is likely to perform well on new, unseen data.

# Results
## :cyclone: Linear Regression model <br>
### :house: :office: Houses and Apartments combined <br>
![Linear Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/d3b52656-0c8b-43ac-9c99-15bc8ac77b3f)


### :house: Houses only <br>
![Linear Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/8019805d-5393-4d97-97fc-34af6996b0ad)


### :office: Apartments only <br>
![Linear Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/09a789aa-5733-425f-8732-b8739296d9e4)



## :cyclone: K-Nearest Neighbour Regression model <br>
### :house: :office: Houses and Apartments combined <br>
Using GridSearchCV to get the best parameters:<br>
![KNN Regression (houses + apartments GridSearch)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/27fb3dce-87d4-465c-b6bf-a27501a6745a)



### :house: Houses only <br>
![KNN Regression (Houses only) - GridSearch](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/e27eff09-7eba-4ccc-a0dd-f805facb32d9)


### :office: Apartments only <br>
![KNN Regression (Apartments only) - GridSearch](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/8dd7af08-2fd3-42d1-bf7d-907b97d79e20)



## :cyclone: HuberRegression model <br>
### :house: :office: Houses and Apartments combined <br>
![Huber Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/2e4323f6-eb76-46a1-aae8-2b7271294b6f)


### :house: Houses only <br>
![Huber Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/2433d815-6e7d-457e-b6b3-2e95a06220da)


### :office: Apartments only <br>
![Huber Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/52ef1107-3160-4e17-9970-4f6d146b1720)



## :cyclone: Decision Tree Regression model <br>
### :house: :office: Houses and Apartments combined <br>
![Decision Tree Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/d964428a-eab8-4205-af88-14e9c057194f)

### :house: Houses only <br>
![Decision Tree Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a4934bf4-14f0-4420-b328-2eec670dd856)



### :office: Apartments only <br>
![Huber Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a78f5a67-9003-4809-91fa-8b2f8ebd962a)



## :cyclone: XGBoost Regression model <br>

**Note:** I have added the **GridSearchCV** feature to extract the **best hyperparameters** for (only) the **XGBoost model** as this regression model gave me the best initial model. After tuning a **R2 score of 0.81** was acquired.

### :house: :office: Houses and Apartments combined <br>
![XGBoost Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a84d6536-4d4c-4a7d-921c-8e4bfef38ec8)


### :house: Houses only <br>
![XGBoost Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/6e7f7ac2-cc84-462b-8f85-2cc15ccb5168)

### :office: Apartments only <br>
![XGBoost Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c36fcd17-f2e8-445c-8638-5951389dffdf)


## :cyclone: Stochastic Gradient Descent Model <br>
### :house: :office: Houses and Apartments combined <br>
![SGD Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/eef8ac8d-6834-4ba7-a951-03beb084c075)


### :house: Houses only <br>
![SGD Regression (House only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/545d8c5d-1a5a-46aa-8418-8d63bc55d960)


### :office: Apartments only <br>
![SGD Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/399d02ad-fb77-45de-8fcc-68a09e631a1e)


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
### :house: :office: Houses and Apartments combined <br>
R2 score= 0.62 <br>
loss: 74859.0938 <br>
![Tensorboard](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/654210ba-b050-4832-9d62-1a6b45892185)

### :house: Houses only <br>
R2 score = 0.62 <br>
loss: 77604.0625 <br>
![Tensorboard 3](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/e00058e8-26e8-49f8-ab92-a4937cb7ca6e)

### :office: Apartments only <br>
R2 score = 0.42 <br>
loss: 77865.6250 <br>
![Tensorboard 2](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/25085629-a1e6-49e8-9677-9a9f0138d3b1)

