# Importing Necessary modules
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_new_data
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

# Declaring our FastAPI instance
app = FastAPI()

# handling get request
@app.get('/')
def main():
    return {'message': 'Welcome to my House price prediction model',
            'INFO': 'See the data section to check what features need to be sent - either float or one of the items in the lists',
            "data-format": {
                    "property-type": ['HOUSE', 'APARTMENT'],
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
@app.post('/predict/')
def predict_price(data: House):
    data = data.dict()
