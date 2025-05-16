from lib.DataPipeline import DataPipeline
from lib.DataManager import DataManager
from lib.model.XGBoostModel import XGBoostModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm

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
    #"price": 395000,  
    """
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
    """
     #"price": 295000,
    
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
    return predictions[0]
    

def dertimane_columns_importance():
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
    xgbmodel = XGBoostModel(verbose=1)

    xgbmodel.dertermine_features_importance(df)


def predict_on_test_data(path_to_model="model/xgb_model.model"):
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
    

    xgb_model = XGBoostModel(verbose=1)
    xgb_model.load_model(path_to_model=path_to_model)

    predictions = xgb_model.predict(X_test_processed)
    predicted_price = test_predict(path_to_model=path_to_model)

    pred_arr = np.array(predictions)

    true_values_arr = np.array(y_test)

    errors_test = pred_arr - true_values_arr
    print(f"Error: {errors_test}")

    # Calculate mean and standard deviation of errors
    mean_error_test = np.mean(errors_test)
    std_error_test = np.std(errors_test)


    # Choose a confidence level
    
    z_score = norm.ppf(1 - (1 - 0.46) / 2)  # Get Z-score

    lower_bound = predicted_price - (z_score * std_error_test)
    upper_bound = predicted_price + (z_score * std_error_test)
    z_value = (predicted_price - lower_bound) / std_error_test
    confidence_level = 2 * norm.cdf(z_value) - 1

    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"We are approximately {confidence_level * 100:.0f}% confident that the actual price falls within:")
    print(f"  ${lower_bound:,.2f} and ${upper_bound:,.2f}")

    # Generate x-values for the normal distribution curve
    x = np.linspace(min(errors_test), max(errors_test), 100)
    y = norm.pdf(x, mean_error_test, std_error_test)  # Normal distribution probability density function
    # Percentage error
    percentage_error = (std_error_test / predicted_price) * 100

    # Plotting
    # Create a range of price deviations (x-axis) around predicted price
    x = np.linspace(predicted_price - 4*std_error_test, predicted_price + 4*std_error_test, 1000)

    # Generate normal distribution centered around predicted price
    pdf = norm.pdf(x, loc=predicted_price + mean_error_test, scale=std_error_test)

    # Plot normal distribution
    # Define sigma levels
    sigma_levels = [1, 2]


    # Confidence intervals
    ci_68_mask = (x >= predicted_price - std_error_test) & (x <= predicted_price + std_error_test)
    ci_95_mask = (x >= predicted_price - 2*std_error_test) & (x <= predicted_price + 2*std_error_test)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Prediction Uncertainty', line=dict(color='skyblue')))
    fig.add_trace(go.Scatter(
        x=x[ci_95_mask], y=pdf[ci_95_mask], fill='tozeroy', mode='none',
        name='Very likely range (~95%)', fillcolor='rgba(135, 206, 235, 0.2)'
    ))
    fig.add_trace(go.Scatter(
        x=x[ci_68_mask], y=pdf[ci_68_mask], fill='tozeroy', mode='none',
        name='Likely range (~68%)', fillcolor='rgba(135, 206, 235, 0.4)'
    ))
    fig.add_trace(go.Scatter(
        x=[predicted_price, predicted_price], y=[0, max(pdf)],
        mode='lines', line=dict(color='red', dash='dash'),
        name=f'Predicted Price: €{predicted_price:0,}'
    ))

    fig.update_layout(
        title="Predicted Real Estate Price with Estimated Uncertainty",
        xaxis_title="Price",
        yaxis_title="Probability",
        xaxis=dict(range=[0, predicted_price + 4 * std_error_test]),
        template='simple_white'
    )
    fig.update_layout(
        yaxis=dict(range=[0, None]),
        xaxis=dict(range=[0, predicted_price + 4 * std_error_test]),
        title="Predicted Real Estate Price with Estimated Uncertainty",
        xaxis_title="Price",
        yaxis_title="Probability",
        template='simple_white'
    )
    fig.write_html("price_prediction.html")
    fig.show()                      


    # Plot normal distribution
    """
    # Plot the normal distribution
    plt.figure(figsize=(10, 5))
    plt.plot(x, pdf, label='Prediction Uncertainty (Normal Distribution)', color='skyblue')

    # Plot the predicted price
    plt.axvline(round(predicted_price), color='red', linestyle='--', linewidth=2, label='Predicted Price')

    # Confidence intervals
    plt.fill_between(x, 0, pdf, where=(x >= round(predicted_price - std_error_test)) & (x <= round(predicted_price + std_error_test)),
                    color='skyblue', alpha=0.3, label='Likely range (~68% chance)')
    plt.fill_between(x, 0, pdf, where=(x >= round(predicted_price - 2*std_error_test)) & (x <= round(predicted_price + 2*std_error_test)),
                    color='skyblue', alpha=0.15, label='Very likely range (~95% chance)')

    # Add vertical lines and labels
    for i in [1, 2]:
        for sign in [-1, 1]:
            offset = sign * i * std_error_test
            plt.axvline(round(predicted_price) + round(offset), color='gray', linestyle=':', linewidth=1)
            label = f"{'+' if sign > 0 else '-'} ${abs(offset):,}"
            y_pos = max(pdf) * (0.6 if i == 1 else 0.4)
            plt.text(round(predicted_price) + round(offset), y_pos, label, ha='center', fontsize=9, color='gray')

    plt.text(round(predicted_price), round(max(pdf) * 0.95), "Predicted Price", ha='center', color='red', fontsize=10)

    # Formatting
    plt.title("Predicted Price with Estimated Uncertainty")
    plt.xlabel("Price")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    """    """
    # 5. Plotting
    plt.figure(figsize=(12, 6))

    # Subplot 1: Predicted Price Distribution with Uncertainty Range
    plt.subplot(1, 2, 1)
    price_range = np.linspace(
        max(0, predicted_price - 3 * std_error_test),  # Ensure >= 0
        predicted_price + 3 * std_error_test,
        100
    )
    price_distribution = norm.pdf(price_range, predicted_price, std_error_test)
    plt.plot(price_range, price_distribution, color='blue', label='Predicted Price Distribution')
    plt.fill_between(
        price_range,
        price_distribution,
        alpha=0.3,
        color='lightblue',
        label='95% Confidence Interval'
    )
    plt.xlabel("Price", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title("Predicted Price Distribution", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(bottom=0)

    # Add predicted price as a vertical line
    plt.axvline(x=predicted_price, color='green', linestyle='--', label='Predicted Price')

    # Add lower and upper bounds as vertical lines
    plt.axvline(x=lower_bound, color='red', linestyle='--', label='Lower Bound (95% CI)')
    plt.axvline(x=upper_bound, color='red', linestyle='--')

    # Subplot 2: Distribution of Errors
    plt.subplot(1, 2, 2)
    plt.hist(errors_test, bins=10, density=True, alpha=0.6, color='blue')
    x = np.linspace(min(errors_test), max(errors_test), 100)
    y = norm.pdf(x, mean_error_test, std_error_test)
    plt.plot(x, y, 'r-', linewidth=2, label='Normal Distribution')
    plt.xlabel("Error", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Errors (Test Set)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)

    plt.tight_layout()
    plt.show()

    # 6.  Output Price and Confidence
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"We are 95% confident that the actual price falls within ${lower_bound:.2f} and ${upper_bound:.2f}")
   print(f"We are approximately {confidence_level * 100:.0f}% confident that the actual price falls within:")
   print(f"  ${lower_bound:,.2f} and ${upper_bound:,.2f}")
    """



if __name__ == "__main__":

    # Set pandas to display all rows (if you want to print everything)
    pd.set_option('display.max_rows', None)  # Display all rows

    # Set pandas to display full content of each cell (if content is long)
    pd.set_option('display.max_colwidth', None)  # No truncation of cell content

    train_xgboost_model(save=False,path_to_save="model/xgb_model.model")
    

    #test_predict(path_to_model="model/xgb_model.model")

    #predict_from_csv(path_to_model="model/xgb_model.model")

    #dertimane_columns_importance()

    #predict_on_test_data()


    