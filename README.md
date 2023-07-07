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
The goal of this project is to effectively clean a dataset obtained by scraping an **Immo website (Immoweb)**. We want to have a **fully functional pipeline** to clean the data. After cleaning the dataset we want to **visualise and analyse the data**. We will be using **matplotlib and seaborn**. After the visualisation and getting to know the data we will train a **Machine Learning model** to **predict prices** on houses with specific characteristics. <br>
The cleaning and reasoning has been written out in the jupyernotebook.

----------------------------------------------------------------------------------------------------------
## :star: Detailed Overview Correlation
There are 21 different categories (DataFrame columns) to assess.<br>
'locality', 'property_type', 'property_subtype', 'price', 'number_rooms', 'living_area', 'kitchen', 'furnished', 'fireplace', 'terrace', 'terrace_area', 'garden', 'garden_area', 'surface_land', 'number_facades', 'swimming_pool', 'building_state', 'latitude', 'longitude', 'region', 'province'

**Correlations:** <br>
![correlationmatrix](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/3b2a3306-2700-4a6f-bb98-0e8493fc4451) <br>

### Analysis of the correlation matrix
- price is highly overall correlated with living_area <br>
![pricelivingarepertype](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/3444e6c7-cea5-4f0d-9754-1590eb77d5d6)

- price is highly overall correlated with number of rooms <br>
![pricenumberroomspertype](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/f78e40dc-2680-4b25-b7f0-128a8c462507)

- number_rooms is highly overall correlated with living_area <br>
![livingareavsnumberrooms_regression](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/2ddfafa8-3c00-499b-aef3-ea49c3098b16)

- living_area is highly overall correlated with surface_land <br>
  ![surfacelandvslivingarea_regressiojn](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/ec0f1c5c-b793-4ad5-b1ca-70a0e72ed0b4)

## :star: Detailed Overview of the cleaning process
### :cyclone: Price
ASSUMPTION:  We have 475 rows with NaN as price - 2.38% of the total
Drop the NaN values (as (al)most (all) of the data is not filled out) 

Below you can find the normal distribution of the Prices<br>
![price_normal_distribution](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/5ec7b8c2-b7c0-4eef-b2b2-8b60968f9c61)


### :cyclone: Locality
No dropping / replacing needed - All values are filled out
Adding Longitude and latitude based on the city name (using geocoding - opencage)

Heatmap of locality with **prices as weight** (more expensive gets more weight) <br>
**Note:** this heatmap will be generated as an interactive .html file (when running the notebook)
![heatmap_example](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/6d1101cc-adc9-4bef-9d2b-f216a2efa5c1)
<br>
A wordcloud to show the most common localities of listings: <br>
![wordlcloudlocalities](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/82070f7f-9070-461f-8dfa-53c805896832)

### :cyclone: Regions <br>
Price per region - barplot<br>
The prices in Brussels are generally higher than the ones in Flanders and Wallonia. Prices in Flanders are generally higher than the ones in Wallonia <br>
![price per region barplot](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/c85ce71a-08bc-4ce3-9123-1e82d7e904df)
<br>

### :cyclone: Province <br>
Price per Province - barplot<br>
![price per province barplot](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/72da4414-ca49-46e5-b64b-9a3a4712d1d2)


### :cyclone: Property Type <br>
No dropping / replacing needed - All values are filled out <br>
![property_type](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/15f6cafe-158b-46e3-a0e9-25c3a5fae3dd)


### :cyclone: Property Subtype
:exclamation: property_subtype is highly imbalanced (54.6%) :exclamation: <br>
No dropping / replacing needed - All values are filled out <br>
![property_subtype_vs_price](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/42183af5-66ed-471f-afc8-a950a7e5333d)

### :cyclone: Number Rooms <br>
No dropping / replacing needed - All values are filled out <br>
Used the IQR method to remove outlyers <br>

Below a distrubution plot the values of number of rooms in the datase
![number_rooms_histplot](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/11bfeee2-6c24-48b2-b0e2-6825427b109c)

Below the prices per number of rooms per region (split on property type)<br>
![Price per rooms per property type](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/893598a0-6fa5-4992-af1a-90f9c6bcc18a)

### :cyclone: Living Area
ASSUMPTION: We have 1054 rows with NaN - 5.41% of the total <br>
Replacing the NaN values with the mean values of living_area grouped by property_subtype
Used the IQR method to remove outlyers <br>

![Living_area_histplot](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/42c580d5-1d70-491b-97d6-ce573263d239)

Below you can find the price per living_area per region (filtered on property_type) <br>
![Price per Living Area per region](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/e282acc3-59ad-4541-bd0e-e27621ca2d6a)


### :cyclone: Kitchen
ASSUMPTION: A lot of NaN values. We can convert these NaN values AND the '0' values to 'NOT_DEFINED' <br>

![type_of_kitchen_count](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/86933170-8d24-43b1-8a5e-ba5f5528c225)


### :cyclone: Furnished
:exclamation: furnished is highly imbalanced (86.4%)	 :exclamation: <br>

ASSUMPTION: We have 9743 rows with NaN - 49.97 % of the total
I assume that this NaN mean it is not furnished (especially Belgium not a lot of houses are sold furnished). Replacing False/True with 0/1 <br>

![furnished_property_count](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/e0c43967-31ef-4bbb-b6e7-4a3a1e333937)


### :cyclone: Fireplace <br>
:exclamation: fireplace is highly imbalanced (88.0%)	 :exclamation: <br>
ASSUMPTION: We have 4878 rows as -1 - 25.0 % of the total
Replace the -1 with 0 <br>

ASSUMPTION: We have 13323 rows as NaN - 68.33 % of the total
Replace the NaN with 0 <br>

![fireplaces](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/2a9ca642-30fc-4c92-8652-e523251fa2dc)

### :cyclone: Terrace
ASSUMPTION: We have 6612 rows as NaN - 33.91 % of the total
Replace the NaN values with 0
Replace True with 1 <br>
![terrace](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/26b3bc74-170c-4a63-942d-85f1edd5400f)

### :cyclone: Terrace Area
If there is an terrace_area, that means terrace should be 1 and vice versa
if terrace area is 0 and terrace is 1. We replace the 0 values with the mean terrace_area of that particular property_subtype
Used the IQR method to remove outlyers <br>

### :cyclone: Garden
No dropping / replacing needed - All values are filled out <br>
![garden](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/75c4961b-740b-45c9-881c-43ee4fa4a5ca)

### :cyclone: Garden Area
If there is an garden_area, that means garden should be 1 and vice versa

### :cyclone: Surface Land
ASSUMPTION: We have 1895 rows as 'UNKNOWN' - 9.7 % of the total
change the 'UNKNOWN' to the mean_value of surface_land grouped by property_subtype <br>
Used the IQR method to remove outlyers <br>

![surfaceland](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/9092490d-d878-4142-9a80-c8ddbd9bb4f0)

### :cyclone: Number of Facades
Replace the rounded number of facades (if not filled in) by the mean 

### :cyclone: Swimming Pool
:exclamation: swimming_pool is highly imbalanced (89.8%) :exclamation: <br>
No dropping / replacing needed - All values are filled out
change False and True to 0 and 1 <br>
![swimmingpool](https://github.com/nikolaaswillaert/ImmoElizaDataAnalysis/assets/106211266/f648159e-60db-40a0-8ce8-6e28f1d23b1f)

### :cyclone: Building State
We have 3385 values as 0 (17.3%)
We have 293 values as 'UNKNOWN' (1.5%)
Setting both values to 'UNKNOWN' category

## 5. Main Take Aways



:exclamation: **Imbalances:** :exclamation:
- property_subtype is highly imbalanced (54.6%)
- furnished is highly imbalanced (86.4%)	
- fireplace is highly imbalanced (88.0%)	
- swimming_pool is highly imbalanced (89.8%)	
- garden_area data is highly skewed

## 6. Timeline
Started on Monday 03/06/2023 - end on Tuesday 11/06/2023
