# Data Cleaning and Analysis 

### 1. Instructions
I started off with a raw dataset obtained from the ImmoEliza Datascraper (https://github.com/nikolaaswillaert/ImmoEliza). This document is a csv-file.
From this dataset, I start with cleaning and interpreting the data based on several assumptions.

### 2. Installation

Program was written using python 3.11. Please make sure you have python 3.11 installed. have added the requirements.txt file as wel if for some reason the code would not run:
```
pip install -r requirements.txt
```
### 3. Overview of cleaning

There are 17 different categories (DataFrame columns) to assess.
#### 3.0 Price
ASSUMPTION:  We have 475 rows with NaN as price - 2.38% of the total
Drop the NaN values (as (al)most (all) of the data is not filled out) 

#### 3.1 locality
No dropping / replacing needed - All values are filled out

#### 3.2 property_type
No dropping / replacing needed - All values are filled out

#### 3.3 property_subtype
No dropping / replacing needed - All values are filled out

#### 3.4 number_rooms
No dropping / replacing needed - All values are filled out

#### 3.5 living_area
ASSUMPTION: We have 1054 rows with NaN - 5.41% of the total
Replacing the NaN values with the mean values of living_area grouped by property_subtype

#### 3.6 kitchen
ASSUMPTION: A lot of NaN values. We can convert these NaN values AND the '0' values to 'NOT_DEFINED'

#### 3.7 furnished
ASSUMPTION: We have 9743 rows with NaN - 49.97%% of the total
I assume that this NaN mean it is not furnished (especially Belgium not a lot of houses are sold furnished)
Replacing False/True with 0/1

#### 3.8 fireplace
ASSUMPTION: raw data has -1 values. These -1 values can be changed to 0.

#### 3.9 terrace

#### 3.10 terrace_area
#### 3.11 garden
#### 3.12 garden_area
#### 3.13 surface_land
#### 3.14 number_facades
#### 3.15 swimming_pool
#### 3.16 building_state

### Timeline

