from lib.DataPipeline import DataPipeline
from lib.DataManager import DataManager
from lib.model.XGBoostModel import XGBoostModel
import pandas as pd

def train_xgboost_model(save=False,path_to_save="model/xgb_model.model"):

    # load the dataset
    df = DataManager.load_csv(path_to_csv="data/Kangaroo.csv",verbose=1)
    
    # merge columns from Giraffe dataset
    df = DataManager.merge_columnsFrom(main_df=df,
                                       path_to_csv="data/Giraffe.csv",
                                       id_col="id",
                                       from_id_col="propertyId",
                                       from_columns_to_merge=['latitude', 'longitude','cadastralIncome','primaryEnergyConsumptionPerSqm'],
                                       verbose=1)
    
    # Get lat/lng from the zip code with Nominatim
    df = DataManager.get_lat_lng_for_zipcode(df=df,verbose=1)

    print(f"{df.head(10)}")

    # Columns to drop
    unrevelant_columns = ["Unnamed: 0","id","url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 
                          'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 'kitchenSurface', 
                          'floorCount', 'hasDiningRoom', 'hasDressingRoom', 'hasAttic','diningRoomSurface',
                          'hasLivingRoom','livingRoomSurface','gardenOrientation','hasBasement',
                          'streetFacadeWidth','kitchenSurface']

    print("Pipeline creation")
    pipeline = DataPipeline(df,"price")
 
    print("Pipeline execution..")
    X_train_processed, X_test_processed, y_train, y_test, features_list = pipeline.execute_pipeline(
                                                                                unreavelant_columns=unrevelant_columns,
                                                                                k=3,
                                                                                debug=True)
    
    
    xgbmodel = XGBoostModel(verbose=1)

    print("Train XGBoost model...")
    model = xgbmodel.train(train_features=X_train_processed,
                           train_labels=y_train,
                           test_features=X_test_processed,
                           test_labels=y_test)
    
    if save and path_to_save:
        xgbmodel.save_model(path_to_save)

 

def predict_from_csv(path_to_model):
    df = DataManager.load_csv("data/Kangaroo.csv",True)

    df = DataManager.merge_columnsFrom(df,"data/Giraffe.csv","id","propertyId",['latitude', 'longitude','cadastralIncome','primaryEnergyConsumptionPerSqm'],True)

   
    df = DataManager.get_lat_lng_for_zipcode(df=df,verbose=True)

    unrevelant_columns = ["Unnamed: 0","url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 
                          'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 'kitchenSurface', 
                          'floorCount', 'hasDiningRoom', 'hasDressingRoom', 'hasAttic','diningRoomSurface',
                          'hasLivingRoom','livingRoomSurface','gardenOrientation','hasBasement',
                          'streetFacadeWidth','kitchenSurface']#

    print("Pipeline creation")
    pipeline = DataPipeline(df,"price")
 
    print("Pipeline execution..")#'Unnamed: 0', 'id', 'url','monthlyCost','hasBalcony','accessibleDisabledPeople'
    X_train_processed, X_test_processed, y_train, y_test, features_list = pipeline.execute_pipeline(
                                                                                unreavelant_columns=unrevelant_columns,
                                                                                k=3,
                                                                            debug=True)
     
    xgbmodel = XGBoostModel(verbose=1)

    xgbmodel.load_model(path_to_model=path_to_model)

    predictions = xgbmodel.predict(X_test_processed)

    print(f"prediction : {predictions}")
    

def test_predict(path_to_model="model/xgb_model.model"):
    #"price": 295000,
    data = {
            "type": "HOUSE",
            "subtype": "HOUSE",
            "bedroomCount": 3,
            "bathroomCount": 1,
            "province": "Brabant Wallon",
            "locality": "WAVRE",
            "postCode": 1300,
            "habitableSurface": 123,
            "buildingCondition": "GOOD",
            "buildingConstructionYear": 1950,
            "facedeCount": 4,
            "hasLift": False,
            "floodZoneType": "NON_FLOOD_ZONE",
            "heatingType": "GAS",
            "hasHeatPump": False,
            "hasPhotovoltaicPanels": False,
            "hasThermicPanels": False,
            "kitchenType": "INSTALLED",
            "landSurface": 603,
            "hasGarden": True,
            "gardenSurface": 480,
            "parkingCountIndoor": 1,
            "parkingCountOutdoor": 3,
            "hasAirConditioning": False,
            "hasArmoredDoor": False,
            "hasVisiophone": True,
            "hasOffice": False,
            "toiletCount": 2,
            "hasSwimmingPool": False,
            "hasFireplace": False,
            "hasTerrace": True,
            "terraceSurface": 5,
            "terraceOrientation": None,
            "epcScore": "E",
            "cadastralIncome": 1299,
            "primaryEnergyConsumptionPerSqm": 210,
            "latitude": 50.7127987,
            "longitude": 4.581166,
        }
    
     #"price": 295000,
    """
    data = { #199000
            "type": "HOUSE",
            "subtype": "HOUSE",
            "bedroomCount": 3,
            "bathroomCount": 1,
            "province": "Liège",
            "locality": "HERSTAL",
            "postCode": 4040,
            "habitableSurface": 179,
            "buildingCondition": "TO_RENOVATE",
            "buildingConstructionYear": 1900,
            "facedeCount": 3,
            "hasLift": False,
            "floodZoneType": "NON_FLOOD_ZONE",
            "heatingType": "GAS",
            "hasHeatPump": False,
            "hasPhotovoltaicPanels": False,
            "hasThermicPanels": False,
            "kitchenType": "INSTALLED",
            "landSurface": 395,
            "hasGarden": True,
            "gardenSurface": 262,
            "parkingCountIndoor": 1,
            "parkingCountOutdoor": 0,
            "hasAirConditioning": False,
            "hasArmoredDoor": False,
            "hasVisiophone": True,
            "hasOffice": False,
            "toiletCount": 2,
            "hasSwimmingPool": False,
            "hasFireplace": False,
            "hasTerrace": True,
            "terraceSurface": 21,
            "terraceOrientation": None,
            "epcScore": "D",
            "cadastralIncome": 714,
            "primaryEnergyConsumptionPerSqm": 305,
            "latitude": 50.670791,
            "longitude": 5.610183,
        }
    """
    df = pd.DataFrame([data])

    # Get lat/lng from the zip code with Nominatim
    df = DataManager.get_lat_lng_for_zipcode(df=df,verbose=1)

    print(f"df :\n{df.head()}")

    print("Pipeline creation")
    pipeline = DataPipeline(df=df,target_columns_name="price",path_to_save_pipeline="pipeline/xgboost_pipeline.pipeline")

    test_data = pipeline.prepare_data_for_prediction()


    
    xgbmodel = XGBoostModel(verbose=1)

    xgbmodel.load_model(path_to_model=path_to_model)

    predictions = xgbmodel.predict(test_data)

    print(f"prediction : {predictions}")
    



if __name__ == "__main__":

    # Set pandas to display all rows (if you want to print everything)
    pd.set_option('display.max_rows', None)  # Display all rows

    # Set pandas to display full content of each cell (if content is long)
    pd.set_option('display.max_colwidth', None)  # No truncation of cell content

    #train_xgboost_model(save=True,path_to_save="model/xgb_model.model")
    

    test_predict(path_to_model="model/xgb_model.model")
    #predict_from_csv(path_to_model="model/xgb_model.model")


    