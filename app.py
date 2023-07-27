# Importing Necessary modules
import pandas as pd
import json
from fastapi import FastAPI
from pydantic import BaseModel, validator
from src.predict import predict_price
from src.preprocessing import preprocess_new_data

# instantiate the Basemodel of pydantic
class House(BaseModel):
    property_type: str
    property_subtype: str
    kitchen: str
    building_state: str
    region: str
    province: str
    number_rooms: float
    living_area:float
    surface_land:float
    number_facades:float
    latitude:float
    longitude:float

    # add some validation checks
    @validator('property_type')
    def propertytype_cls(cls, value: str) -> str:
        if value not in ['HOUSE', 'APARTMENT']:
            raise ValueError("Wrong value: Please fill out 1 value from this list: ['HOUSE', 'APARTMENT']")
        return value

    @validator('property_subtype')
    def propertysubtype_cls(cls, value: str) -> str:
        if value not in ['HOUSE','VILLA','APARTMENT','MIXED_USE_BUILDING',
                                        'APARTMENT_BLOCK','DUPLEX','FLAT_STUDIO','MANSION',
                                        'EXCEPTIONAL_PROPERTY','GROUND_FLOOR','PENTHOUSE','TOWN_HOUSE',
                                        'TRIPLEX','SERVICE_FLAT','OTHER_PROPERTY','LOFT',
                                        'COUNTRY_COTTAGE','CHALET','BUNGALOW','FARMHOUSE',
                                        'MANOR_HOUSE','KOT']:
            raise ValueError('Wrong value: Please check out the data-format to see possible entries (/docs)')
        return value

    @validator('kitchen')
    def kitchen_cls(cls, value: float) -> float:
        if value not in ['USA_HYPER_EQUIPPED' , 'SEMI_EQUIPPED', 'HYPER_EQUIPPED',
                                'USA_INSTALLED', 'INSTALLED', 'NOT_DEFINED', 'USA_SEMI_EQUIPPED',
                                'NOT_INSTALLED', 'USA_UNINSTALLED']:
            raise ValueError('Wrong value: Please check out the data-format to see possible entries (/docs)')
        return value
    
    @validator('building_state')
    def buildingst_cls(cls, value: str) -> str:
        if value not in ["NEW" , "GOOD" , "TO RENOVATE" , "JUST RENOVATED" , "TO REBUILD"]:
            raise ValueError("Wrong value: Please fill out 1 value from this list: [NEW, GOOD, TO RENOVATE, JUST RENOVATED, TO REBUILD]")
        return value

    @validator('region')
    def region_cls(cls, value: str) -> str:
        if value not in ['Brussels' , 'Flanders' , 'Wallonie']:
            raise ValueError("Wrong value: Please fill out 1 value from this list: ['Brussels' , 'Flanders' , 'Wallonie']")
        return value
    
    @validator('province')
    def province_cls(cls, value: str) -> str:
        if value not in ['Brussels','Antwerp','Walloon Brabant','Flemish Brabant',
                                'East Flanders','Limburg','West Flanders','Liège','Hainaut',
                                'Namur','Luxembourg']:
            raise ValueError("Wrong value: Please fill out 1 value from the list mentioned in /docs")
        return value
    
    @validator('number_rooms')
    def number_rooms_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Number of rooms must be a positive number")
        return value
    
    @validator('living_area')
    def living_area_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Living area must be a positive number")
        return value

    @validator('surface_land')
    def surface_land_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Surface land cannot be less than 0")
        return value
    
    @validator('number_facades')
    def number_facades_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Number of facades cannot be less than 0")
        return value
    
    @validator('latitude')
    def latitude_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("latitude must be a positive number")
        return value
    
    @validator('longitude')
    def longitude_cls(cls, value: float) -> float:
        if value < 0:
            raise ValueError("longitude must be a positive number")
        return value
    
# Declaring our FastAPI instance
app = FastAPI()

# handling get request
@app.get('/')
async def main():
    return {'message': 'Welcome to my House price prediction model',
            'INFO': 'See the data section to check what features need to be sent - either float or one of the items in the lists',
            "data-format": {
                    "property_type": ['HOUSE', 'APARTMENT'],
                    "property_subtype": ['HOUSE','VILLA','APARTMENT','MIXED_USE_BUILDING',
                                        'APARTMENT_BLOCK','DUPLEX','FLAT_STUDIO','MANSION',
                                        'EXCEPTIONAL_PROPERTY','GROUND_FLOOR','PENTHOUSE','TOWN_HOUSE',
                                        'TRIPLEX','SERVICE_FLAT','OTHER_PROPERTY','LOFT',
                                        'COUNTRY_COTTAGE','CHALET','BUNGALOW','FARMHOUSE',
                                        'MANOR_HOUSE','KOT'],
                    "kitchen":['USA_HYPER_EQUIPPED' , 'SEMI_EQUIPPED', 'HYPER_EQUIPPED',
                                'USA_INSTALLED', 'INSTALLED', 'NOT_DEFINED', 'USA_SEMI_EQUIPPED',
                                'NOT_INSTALLED', 'USA_UNINSTALLED'],
                    "building-state": ["NEW" , "GOOD" , "TO RENOVATE" , "JUST RENOVATED" , "TO REBUILD"],
                    "region":['Brussels' , 'Flanders' , 'Wallonie'],
                    "province":['Brussels','Antwerp','Walloon Brabant','Flemish Brabant',
                                'East Flanders','Limburg','West Flanders','Liège','Hainaut',
                                'Namur','Luxembourg'],
                    "number_rooms":"float - f.e.: 2.0",
                    "living_area":"float -f.e.: 297.0",
                    "surface_land":"float -f.e.: 713.0",
                    "number_facades":"float - f.e.: 4.0",
                    "latitude":"float - f.e.: 50.846557",
                    "longitude":"float - f.e.: 4.351697",
                }
            }

# handling post request
@app.post('/predict')
async def predict_house_price(data: House):
    data = json.loads(data.json())

    df = pd.DataFrame(data, index=[0])
    df = preprocess_new_data(df)
    predictions = predict_price(df)
    
    preds = predictions.tolist()
    json_string = json.dumps(preds)

    response = {
        "PREDICTION (PRICE)": f"{preds[0]}",
        "status code": f'200'
    }
    return response
