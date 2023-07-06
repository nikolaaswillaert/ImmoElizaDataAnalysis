#  :hammer_and_wrench: Data Cleaning and Analysis 

## 1. :carpentry_saw: Instructions
I started off with a raw dataset obtained from the **ImmoEliza Datascraper** (https://github.com/nikolaaswillaert/ImmoEliza). This document is a csv-file.
From this **csv-dataset**, we start cleaning and interpreting the data based on several **assumptions**.

## 2. :gear: Installation

Program was written using python 3.11. Please make sure you have **python 3.11** installed. have added the **requirements.txt** file as wel if for some reason the code would not run:
```
pip install -r requirements.txt
```
## 3. :nazar_amulet: Goal 
The goal of this project is to effectively clean a dataset obtained by scraping an **Immo website (Immoweb)**. We want to have a **fully functional pipeline** to clean the data. After cleaning the dataset we want to **visualise and analyse the data**. We will be using **matplotlib and seaborn**. After the visualisation and getting to know the data we will train a **Machine Learning model **to **predict prices** on certain houses



----------------------------------------------------------------------------------------------------------
## 4. Detailed Overview of cleaning (and thought process)
There are 17 different categories (DataFrame columns) to assess.

### 4.0 Price
ASSUMPTION:  We have 475 rows with NaN as price - 2.38% of the total
Drop the NaN values (as (al)most (all) of the data is not filled out) 

### 4.1 locality
No dropping / replacing needed - All values are filled out
Adding Longitude and latitude based on the city name (using geocoding - opencage)

### 4.1.1 Heatmap of locality (generated at the end of the jupyter notebook - after cleaning the dataset)

![heatmap_example](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/6d1101cc-adc9-4bef-9d2b-f216a2efa5c1)


### 4.2 property_type
No dropping / replacing needed - All values are filled out

### 4.3 property_subtype
No dropping / replacing needed - All values are filled out

### 4.4 number_rooms
No dropping / replacing needed - All values are filled out

### 4.5 living_area
ASSUMPTION: We have 1054 rows with NaN - 5.41% of the total
Replacing the NaN values with the mean values of living_area grouped by property_subtype

### 4.6 kitchen
ASSUMPTION: A lot of NaN values. We can convert these NaN values AND the '0' values to 'NOT_DEFINED'

### 4.7 furnished
ASSUMPTION: We have 9743 rows with NaN - 49.97 % of the total
I assume that this NaN mean it is not furnished (especially Belgium not a lot of houses are sold furnished)
Replacing False/True with 0/1

### 4.8 fireplace
ASSUMPTION: We have 4878 rows as -1 - 25.0 % of the total
Replace the -1 with 0

ASSUMPTION: We have 13323 rows as NaN - 68.33 % of the total
Replace the NaN with 0

### 4.9 terrace
ASSUMPTION: We have 6612 rows as NaN - 33.91 % of the total
Replace the NaN values with 0
Replace True with 1

### 4.10 terrace_area
If there is an terrace_area, that means terrace should be 1 and vice versa
if terrace area is 0 and terrace is 1. We replace the 0 values with the mean terrace_area of that particular property_subtype

### 4.11 garden
No dropping / replacing needed - All values are filled out

### 4.12 garden_area
If there is an garden_area, that means garden should be 1 and vice versa

### 4.13 surface_land
ASSUMPTION: We have 1895 rows as 'UNKNOWN' - 9.7 % of the total
change the 'UNKNOWN' to the mean_value of surface_land grouped by property_subtype

### 4.14 number_facades
Replace the number of facades (if not filled in) by the mean 

### 4.15 swimming_pool
No dropping / replacing needed - All values are filled out
change False and True to 0 and 1

### 4.16 building_state
We have 3385 values as 0 (17.3%)
We have 293 values as 'UNKNOWN' (1.5%)
Setting both values to 'UNKNOWN' category

## 5. Timeline
Started on Monday 03/06/2023 - end on Friday 07/06/2023
