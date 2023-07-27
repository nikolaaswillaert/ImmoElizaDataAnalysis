# :chart_with_upwards_trend: Model Price prediction :chart_with_upwards_trend:

## Model Selection

In the **'model_exploration.ipynb'** notebook I have trained a set of different Linear and non-linear models to see how they would perform with my dataset. After the benchmark of these models I select the best one and fine tune the hyperparameters of the model to get the best possible result. The model i chose is XGBoostRegression model (with finetuned parameters)

## Summary of the evaluation of the XGBoost Regression model

:pushpin: **Mean Absolute Error (MAE)** is 54885.9907: This indicates that, on average, the model's predictions differ from the actual values by approximately 54042.1821 units. Lower MAE values indicate better accuracy.

:pushpin: **Explained Variance Score** is 0.8035: The explained variance score assesses the proportion of variance in the target variable that the model captures. A value of 0.8074 indicates that the model explains 80.74% of the total variance in the data, which is quite good.

:pushpin: **R-squared (R2)** Score is 0.8034: The R2 score, also known as the coefficient of determination, is the same as the explained variance score in this case. It indicates that 80.74% of the variance in the target variable is explained by the model.

:card_index: In summary, the model achieves a **high level of accuracy**, capturing a significant portion of the variance in the data and making predictions that are **close to the actual values**.

:card_index: Regarding its training and testing performance, the **model has been trained well with a training score of 0.9253**, suggesting that it has learned from the training data effectively. When evaluated on the testing data, it performs very well with a **testing score of 0.8074**, which is **consistent with the R2 score** mentioned earlier.

:card_index: Finally, **the cross-validation scores** further validate the model's **generalization ability**, as all the scores are relatively high and close to the testing score. This means the **model is robust and reliable**, and it is likely to perform well on new, unseen data.

### :house: :office: Houses and Apartments combined <br>
![XGBoost Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a84d6536-4d4c-4a7d-921c-8e4bfef38ec8)


### :house: Houses only <br>
![XGBoost Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/6e7f7ac2-cc84-462b-8f85-2cc15ccb5168)

### :office: Apartments only <br>
![XGBoost Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c36fcd17-f2e8-445c-8638-5951389dffdf)

# :globe_with_meridians: Deployment :globe_with_meridians:
## API - Fastapi

You can access the api running on:
```
https://house-prediction-model-api.onrender.com
```
or go to the following link to send the house data directly through the portal:
```
https://house-prediction-model-api.onrender.com/docs
```

or send a curl request to receive the price prediction, where the -d flag is followed by a dictionary with the required house features.
```
curl -X 'POST' \
  'https://house-prediction-model-api.onrender.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "property_type": "HOUSE",
  "property_subtype": "VILLA",
  "kitchen": "SEMI_EQUIPPED",
  "building_state": "TO RENOVATE",
  "region": "Flanders",
  "province": "West Flanders",
  "number_rooms": 4,
  "living_area": 150,
  "surface_land": 200,
  "number_facades": 2,
  "latitude": 51.208887,
  "longitude": 3.445221
}'
```
