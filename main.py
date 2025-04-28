from lib.DataCleaner import DataCleaner
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def split_data(df, test_size=0.2, random_state=42):
    """
    Split the DataFrame into training and testing sets.
    
    Parameters:
    df (DataFrame): The DataFrame to split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: A tuple containing the training and testing DataFrames.
    """
    return train_test_split(df, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    # Initialize the DataCleaner with the path to the dataset
    cleaner = DataCleaner("data/Kangaroo.csv")
    # Clean the data
    df_cleaned_data = cleaner.clean_data(["Unnamed: 0", "url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 'diningRoomSurface', 
                      'streetFacadeWidth', 'gardenOrientation', 'kitchenSurface', 'floorCount', 'hasDiningRoom', 
                      'hasDressingRoom', 'hasAttic','terraceOrientation','province','locality'])

    df_cleaned_data.head()
    df_cleaned_data = df_cleaned_data[df_cleaned_data['price'] >= 50000]
    df_cleaned_data = df_cleaned_data[df_cleaned_data['price'] <= 1000000]
    
    # Print the cleaned data
    print(df_cleaned_data)
    # Labels are the values we want to predict
    prices = np.array(df_cleaned_data['price'])
    df_cleaned_data = df_cleaned_data.drop(['price'], axis=1)

    # Saving feature names for later use
    feature_list = list(df_cleaned_data.columns)

    # Convert to numpy array
    features = np.array(df_cleaned_data)  # Uncommented this line to convert features to numpy array

    train_features, test_features, train_labels, test_labels = train_test_split(features, prices, test_size = 0.25, random_state = 42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'euro.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')