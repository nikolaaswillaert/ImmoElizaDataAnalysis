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
### Houses and Apartments combined <br>
![Linear Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/557a1539-09d7-424d-b6d2-8772d2d654c0)

```
--------------------------------------
Mean absolute error = 81351.5961
Mean squared error = 12297112277.8925
Median absolute error = 62936.0
Explain variance score = 0.5807
R2 score *coefficient of Determination = 0.5806
--------------------------------------
TRAINING SCORE: 0.5798523419274266
TESTING SCORE: 0.5806381257581426
--------------------------------------
Cross validation scores: 
 [ 5.42733466e-01  5.46934876e-01  5.89128376e-01  5.53159927e-01
  5.90421359e-01  5.63165105e-01 -2.16531184e+20  5.72792017e-01
  6.10080083e-01  5.62182127e-01]
```
### Houses only <br>
![Linear Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0e33b253-b933-407f-ae0c-975d8b7aa3e7)

```
--------------------------------------
Mean absolute error = 78656.5337
Mean squared error = 11344728018.8809
Median absolute error = 59584.0
Explain variance score = 0.632
R2 score *coefficient of Determination = 0.6309
--------------------------------------
TRAINING SCORE: 0.6716441120417997
TESTING SCORE: 0.6309056548707214
--------------------------------------
Cross validation scores: 
 [0.65742856 0.67212323 0.65248232 0.659429   0.64124224 0.67348972
 0.65805296 0.66971892 0.67797175 0.68864123]
```
### Apartments only <br>
![Linear Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/de73ac01-a592-443a-a0cb-4d218cbb44e9)
```
--------------------------------------
Mean absolute error = 82594.3468
Mean squared error = 13707318764.9881
Median absolute error = 60392.0
Explain variance score = 0.4293
R2 score *coefficient of Determination = 0.4279
--------------------------------------
TRAINING SCORE: 0.47898402514169414
TESTING SCORE: 0.427858991005727
--------------------------------------
Cross validation scores: 
 [-6.79368252e+19  4.85540404e-01  4.29287833e-01  4.05580740e-01
  4.05419514e-01  4.16702634e-01  4.52918926e-01  5.41011552e-01
  5.02146144e-01  4.69250912e-01]
```

## K-Nearest Neighbour Regression model <br>
### Houses and Apartments combined <br>
![KNN Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/8d9a22f2-3ff5-4105-91f1-99cd06c603bd)

Using GridSearchCV to get the best parameters:
```
--------------------------------------
Mean absolute error = 65841.3204
Mean squared error = 10162757294.6057
Median absolute error = 43768.4688
Explain variance score = 0.6541
R2 score *coefficient of Determination = 0.6534
--------------------------------------
TRAINING SCORE: 0.9993332932162652
TESTING SCORE: 0.6534248976328503
--------------------------------------
Cross validation scores: 
 [0.57740195 0.69823234 0.63280251 0.62696426 0.67221104 0.55382267
 0.59946802 0.58136225 0.63922162 0.58592388]

```
### Houses only <br>
![KNN Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/81600109-bbc5-44cc-bab0-9a2679dac0f5)


```
--------------------------------------
Mean absolute error = 70866.5079
Mean squared error = 11568731168.3563
Median absolute error = 47239.1164
Explain variance score = 0.6237
R2 score *coefficient of Determination = 0.6236
--------------------------------------
TRAINING SCORE: 0.999765394940186
TESTING SCORE: 0.6236178383955329
--------------------------------------
Cross validation scores: 
 [0.65520714 0.64413214 0.62356329 0.63494463 0.61880427 0.68727865
 0.63218958 0.62279167 0.65115438 0.68209554]
```
### Apartments only <br>
![KNN Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c3201f08-32f2-4c6b-8d49-1343640d5d05)
```
--------------------------------------
Mean absolute error = 79392.2574
Mean squared error = 13628868626.8734
Median absolute error = 55271.4286
Explain variance score = 0.4319
R2 score *coefficient of Determination = 0.4311
--------------------------------------
TRAINING SCORE: 0.4706776018972737
TESTING SCORE: 0.4311334855984523
--------------------------------------
Cross validation scores: 
 [0.46032972 0.27914073 0.25520822 0.42719012 0.30283208 0.4274735
 0.32258478 0.43152421 0.45872109 0.39951993]
```

## HuberRegression model <br>
### Houses and Apartments combined <br>
![KNN Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/964e0f33-2298-4e07-b9c2-ecd521fbd12a)

```
--------------------------------------
Mean absolute error = 79367.9613
Mean squared error = 12555310794.4747
Median absolute error = 59735.0762
Explain variance score = 0.5765
R2 score *coefficient of Determination = 0.5718
--------------------------------------
TRAINING SCORE: 0.5656109149823432
TESTING SCORE: 0.5718329191865958
--------------------------------------
Cross validation scores: 
 [0.53202622 0.53345899 0.5764177  0.54958728 0.56803282 0.54200913
 0.5562657  0.57200397 0.61051626 0.55383638]
```
### Houses only <br>
![Huber Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/d68d1833-ee54-4204-8352-b524556aeb8a)


```
Mean absolute error = 77494.2771
Mean squared error = 11556908912.9665
Median absolute error = 57065.9749
Explain variance score = 0.6244
R2 score *coefficient of Determination = 0.624
--------------------------------------
TRAINING SCORE: 0.6637142631223514
TESTING SCORE: 0.6240024688250836
--------------------------------------
Cross validation scores: 
 [0.65051602 0.65947206 0.64631469 0.65803661 0.63305784 0.66641404
 0.652217   0.6678057  0.6664528  0.6856284 ]
```
### Apartments only <br>
![KNN Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/54a69601-4a4c-4814-95b2-b0b9bde5bea2)

```
--------------------------------------
Mean absolute error = 79328.8198
Mean squared error = 15326031345.7778
Median absolute error = 52983.8193
Explain variance score = 0.3843
R2 score *coefficient of Determination = 0.3603
--------------------------------------
TRAINING SCORE: 0.41027530068386586
TESTING SCORE: 0.3602942203074433
--------------------------------------
Cross validation scores: 
 [0.39105602 0.43792702 0.33212374 0.41573569 0.30863224 0.42038837
 0.40026254 0.45858344 0.41495599 0.33925622]
```


## Decision Tree Regression model <br>
### Houses and Apartments combined <br>
![Decision Tree Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c0e76dea-4c01-4f28-86c5-2c2c5f8b041f)
```
--------------------------------------
Mean absolute error = 73385.2603
Mean squared error = 12642212660.1208
Median absolute error = 49000.0
Explain variance score = 0.5692
R2 score *coefficient of Determination = 0.5689
--------------------------------------
TRAINING SCORE: 0.9993332932162652
TESTING SCORE: 0.568869351120457
--------------------------------------
Cross validation scores: 
 [0.45064662 0.48189894 0.53340498 0.55782218 0.49195854 0.52114875
 0.54049307 0.52239161 0.47016358 0.52883236]
```
### Houses only <br>
![Decision Tree Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/ca317511-e718-4fff-81a3-cbb380838397)

```
--------------------------------------
Mean absolute error = 77981.47
Mean squared error = 14095867282.7102
Median absolute error = 50000.0
Explain variance score = 0.5423
R2 score *coefficient of Determination = 0.5414
--------------------------------------
TRAINING SCORE: 0.999765394940186
TESTING SCORE: 0.5413988863300778
--------------------------------------
Cross validation scores: 
 [0.50047088 0.5096556  0.49142754 0.56830718 0.56684488 0.5961688
 0.5322211  0.52161241 0.49503307 0.54802041]
```
### Apartments only <br>
![Decision Tree Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/fbb694c8-e2fe-44c3-8495-ba7cd7f0c063)
```
--------------------------------------
Mean absolute error = 67884.1449
Mean squared error = 11705335760.2278
Median absolute error = 36000.0
Explain variance score = 0.514
R2 score *coefficient of Determination = 0.5114
--------------------------------------
TRAINING SCORE: 0.9985756950301382
TESTING SCORE: 0.5114213999619304
--------------------------------------
Cross validation scores: 
 [0.31895303 0.406326   0.49734905 0.45867462 0.60475302 0.58927532
 0.44181489 0.39605371 0.69791638 0.55416734]
```

## XGBoost Regression model <br>
### Houses and Apartments combined <br>
![XGBoost Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/476e3522-97bc-4d37-a2a8-683697ddde82)

```
--------------------------------------
Mean absolute error = 54885.9907
Mean squared error = 5845267189.5407
Median absolute error = 39387.2656
Explain variance score = 0.8035
R2 score *coefficient of Determination = 0.8034
--------------------------------------
TRAINING SCORE: 0.9310402886305642
TESTING SCORE: 0.8034376727713815
--------------------------------------
Cross validation scores: 
 [0.74928108 0.75056096 0.76551827 0.75152203 0.80326452 0.75630356
 0.74491564 0.78125817 0.73967731 0.72692145]
```

### Houses only <br>
![XGBoost Regression (Houses only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/4999f126-ce89-4e27-b79a-1341ea05a5fe)

```
--------------------------------------
Mean absolute error = 59820.5286
Mean squared error = 7440554949.7951
Median absolute error = 40971.3281
Explain variance score = 0.7609
R2 score *coefficient of Determination = 0.7608
--------------------------------------
TRAINING SCORE: 0.9456457539296447
TESTING SCORE: 0.7608427061758977
--------------------------------------
Cross validation scores: 
 [0.7626534  0.78068538 0.74327949 0.77171942 0.79666323 0.77542566
 0.77471706 0.76724273 0.75806482 0.79102736]
```
### Apartments only <br>
![XGBoost Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/0c5efffe-76fe-4d59-899c-4af39e6c61cb)
```
--------------------------------------
Mean absolute error = 50910.9978
Mean squared error = 6467832544.3885
Median absolute error = 30830.8203
Explain variance score = 0.7109
R2 score *coefficient of Determination = 0.7109
--------------------------------------
TRAINING SCORE: 0.985156908241426
TESTING SCORE: 0.7108958894027468
--------------------------------------
Cross validation scores: 
 [0.75530291 0.5759087  0.68789167 0.65860443 0.80743173 0.7009887
 0.75347213 0.749243   0.76952    0.74819887]
```
## Stochastic Gradient Descent Model <br>
### Houses and Apartments combined <br>
![SGD Regression (houses + apartments)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/51321729-e953-4e1c-a819-a1133a47aae6)
```
--------------------------------------
Mean absolute error = 82060.2973
Mean squared error = 12287746168.3992
Median absolute error = 63675.2242
Explain variance score = 0.5882
R2 score *coefficient of Determination = 0.5868
--------------------------------------
TRAINING SCORE: 0.575385561001303
TESTING SCORE: 0.5867925443037172
--------------------------------------
Cross validation scores: 
 [0.53990928 0.5441909  0.5606337  0.55711293 0.58876985 0.5849176
 0.59024483 0.55664733 0.60995964 0.56804734]
```

### Houses only <br>
![SGD Regression (House only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/a46b55f5-ee21-4950-9303-65960fcd6227)


```
--------------------------------------
Mean absolute error = 79002.1504
Mean squared error = 11610521637.0955
Median absolute error = 58179.3639
Explain variance score = 0.6272
R2 score *coefficient of Determination = 0.6268
--------------------------------------
TRAINING SCORE: 0.6697646305238913
TESTING SCORE: 0.6268099686985804
--------------------------------------
Cross validation scores: 
 [0.6672244  0.6535932  0.65675876 0.65448881 0.63761482 0.67033001
 0.66530214 0.66517233 0.67953676 0.69130284]
```
### Apartments only <br>
![SGD Regression (Apartments only)](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/42a85843-003e-4b8b-a945-db34d8fc4d61)

```
--------------------------------------
Mean absolute error = 79998.9636
Mean squared error = 13007379082.238
Median absolute error = 57952.2054
Explain variance score = 0.4186
R2 score *coefficient of Determination = 0.4186
--------------------------------------
TRAINING SCORE: 0.4686476125677994
TESTING SCORE: 0.41858625204600197
--------------------------------------
Cross validation scores: 
 [0.38552034 0.50319951 0.42472477 0.5176386  0.43161022 0.39010255
 0.4454127  0.51397939 0.47524786 0.46602893]
```
## Neural Network Regression <br>


