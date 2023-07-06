#  :hammer_and_wrench: Data Cleaning and Analysis 

## :carpentry_saw: Instructions
I started off with a raw dataset obtained from the **ImmoEliza Datascraper** (https://github.com/nikolaaswillaert/ImmoEliza). This document is a csv-file.
From this **csv-dataset**, we start cleaning and interpreting the data based on several **assumptions**.

## :gear: Installation

Program was written using python 3.11. Please make sure you have **python 3.11** installed. have added the **requirements.txt** file as wel if for some reason the code would not run:
```
pip install -r requirements.txt
```
## :nazar_amulet: Goal 
The goal of this project is to effectively clean a dataset obtained by scraping an **Immo website (Immoweb)**. We want to have a **fully functional pipeline** to clean the data. After cleaning the dataset we want to **visualise and analyse the data**. We will be using **matplotlib and seaborn**. After the visualisation and getting to know the data we will train a **Machine Learning model **to **predict prices** on certain houses



----------------------------------------------------------------------------------------------------------
## :star: Detailed Overview of cleaning (and thought process)
There are 17 different categories (DataFrame columns) to assess.

### :cyclone: Price
ASSUMPTION:  We have 475 rows with NaN as price - 2.38% of the total
Drop the NaN values (as (al)most (all) of the data is not filled out) 

### :cyclone: Locality
No dropping / replacing needed - All values are filled out
Adding Longitude and latitude based on the city name (using geocoding - opencage)

### Heatmap of locality (generated at the end of the jupyter notebook - after cleaning the dataset)

![heatmap_example](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/6d1101cc-adc9-4bef-9d2b-f216a2efa5c1)


### :cyclone: Property Type
No dropping / replacing needed - All values are filled out

### :cyclone: Property Subtype
No dropping / replacing needed - All values are filled out

### :cyclone: Number Rooms
No dropping / replacing needed - All values are filled out

### :cyclone: Living Area
ASSUMPTION: We have 1054 rows with NaN - 5.41% of the total
Replacing the NaN values with the mean values of living_area grouped by property_subtype

### :cyclone: Kitchen
ASSUMPTION: A lot of NaN values. We can convert these NaN values AND the '0' values to 'NOT_DEFINED'

### :cyclone: Furnished
ASSUMPTION: We have 9743 rows with NaN - 49.97 % of the total
I assume that this NaN mean it is not furnished (especially Belgium not a lot of houses are sold furnished)
Replacing False/True with 0/1

### :cyclone: Fireplace
ASSUMPTION: We have 4878 rows as -1 - 25.0 % of the total
Replace the -1 with 0

ASSUMPTION: We have 13323 rows as NaN - 68.33 % of the total
Replace the NaN with 0

### :cyclone: Terrace
ASSUMPTION: We have 6612 rows as NaN - 33.91 % of the total
Replace the NaN values with 0
Replace True with 1

### :cyclone: Terrace Area
If there is an terrace_area, that means terrace should be 1 and vice versa
if terrace area is 0 and terrace is 1. We replace the 0 values with the mean terrace_area of that particular property_subtype

### :cyclone: Garden
No dropping / replacing needed - All values are filled out

### :cyclone: Garden Area
If there is an garden_area, that means garden should be 1 and vice versa

### :cyclone: Surface Land
ASSUMPTION: We have 1895 rows as 'UNKNOWN' - 9.7 % of the total
change the 'UNKNOWN' to the mean_value of surface_land grouped by property_subtype

### :cyclone: Number of Facades
Replace the number of facades (if not filled in) by the mean 

### :cyclone: Swimming Pool
No dropping / replacing needed - All values are filled out
change False and True to 0 and 1

### :cyclone: Building State
We have 3385 values as 0 (17.3%)
We have 293 values as 'UNKNOWN' (1.5%)
Setting both values to 'UNKNOWN' category

## 5. Timeline
Started on Monday 03/06/2023 - end on Friday 07/06/2023
