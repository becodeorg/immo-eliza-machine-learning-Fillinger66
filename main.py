from lib.DataCleaner import DataCleaner
from lib.DataClustering import DataClustering
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, make_scorer,mean_squared_error # Need make_scorer for custom metrics
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error


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

def train_random_forest_regressor(train_features, train_labels, test_features, test_labels):
    """
    Train a Random Forest Regressor model.
    
    Parameters:
    train_data (DataFrame): The training data.
    train_labels (Series): The labels for the training data.
    test_data (DataFrame): The testing data.
    test_labels (Series): The labels for the testing data.
    Returns:
    model: The trained model.
    """
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 50, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('RANDOM_FOREST::Mean Absolute Error:', round(np.mean(errors), 2), 'euro.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('RANDOM_FOREST::Accuracy:', round(accuracy, 2), '%.')

def train_xgboost(train_features, train_labels, test_features, test_labels):
    #n_estimator=1500 learning_rate=0.11 -> 83.25% accuracy -- Error 53016.44€
    #n_estimator=2000 learning_rate=0.11 -> 83.3% accuracy -- Error 52876.48€ -> no lat/lng for each sells
    n_esitmator=2800 # The number of trees in the ensemble. More trees can lead to better performance but also longer training time.
    learning_rate=0.031 # The step size shrinkage used in update to prevent overfitting. Lower values make the model more robust but require more trees.
    random_state=42
    max_depth=8 # The maximum depth of the tree. Increasing this value will make the model more complex and more likely to overfit.
    subsample=1 # These control the proportion of data and features used to build each individual tree. This is a form of regularization.
    colsample_bytree=0.55 # The subsample ratio of columns when constructing each tree. This is another form of regularization.
    tree_method='hist' # The tree construction algorithm used in XGBoost. 'hist' is faster and more memory efficient for large datasets.
    model = xgb.XGBRegressor(n_estimators=n_esitmator, random_state=random_state, 
                             learning_rate=learning_rate, max_depth=max_depth,
                             subsample=subsample, colsample_bytree=colsample_bytree)

    model.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = model.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    print('XGBOOST::Mean Absolute Error:', round(np.mean(errors), 2), 'euro.')
     # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('XGBOOST::Accuracy:', round(accuracy, 2), '%.')
    print(f"XGBoost:: model trained with\n {n_esitmator} estimators,\n learning rate {learning_rate},\n max depth {max_depth},\n subsample {subsample},\n colsample_bytree {colsample_bytree},\n tree_method {tree_method} .")

    mae = mean_absolute_error(test_labels, predictions)
    accuracy = 100 - mean_absolute_percentage_error(test_labels, predictions) * 100
    print("-"*50)
    print('XGBOOST::Accuracy:', round(accuracy, 2), '%.')
    print('XGBOOST::mae:', round(mae, 2), '%.')

    return model



def train_mlp(train_features, train_labels, test_features, test_labels):
    # The pipeline chains the preprocessing step (Scaler) and the model (MLPRegressor)
    pipeline = Pipeline([
        ('scaler', StandardScaler()), # Step 1: Scale the features
        ('mlp', MLPRegressor(random_state=42)) # Step 2: The MLP model (initial params don't matter much here)
    ])

    print("\nPipeline created.")


    # --- 3. Define the Hyperparameter Grid for the Pipeline ---
    # You define a dictionary where keys are parameter names in the pipeline
    # using the format 'stepname__parametername'
    # Values are lists of the specific values you want to try

    param_grid = {
        # Parameters for the 'mlp' step in the pipeline
        'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)], # Try different network structures
        'mlp__activation': ['relu', 'tanh'],                            # Try different activation functions
        'mlp__solver': ['adam', 'sgd'],                                 # Try different optimizers
        'mlp__alpha': [0.0001, 0.001, 0.01],                            # Try different regularization strengths
        'mlp__learning_rate_init': [0.001, 0.01],                       # Try different initial learning rates
        'mlp__max_iter': [300, 500]                                     # Try different max iterations (start lower for speed)
        # Note: You would typically use more values or ranges in a real scenario
    }

    print("\nHyperparameter grid defined:")
    print(param_grid)


    # --- 4. Define the scoring metric (Optional but Recommended) ---
    # By default, GridSearchCV for regressors uses R-squared.
    # If you want to optimize for MAE or RMSE, you need to specify a scorer.
    # Let's optimize for negative MAE (since GridSearchCV maximizes scores)
    # Or you can optimize for negative MSE (which is equivalent to optimizing for RMSE)
    # You can use make_scorer if you want to use MAE directly.
    # For simplicity, let's use 'neg_mean_squared_error' as the scoring, which GridSearchCV understands.
    # The 'neg_' prefix means it maximizes the negative value, which is equivalent to minimizing the positive value.
    # You could also create a custom MAE scorer: mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    scoring_metric = 'neg_mean_squared_error' # Optimize for minimizing MSE (equivalent to RMSE)

    print(f"\nOptimization metric set to: {scoring_metric}")


    # --- 5. Create the GridSearchCV object ---
    # It takes the pipeline, the parameter grid, the scoring metric, and the number of cross-validation folds (cv)
    grid_search = GridSearchCV(estimator=pipeline, # The pipeline to tune
                            param_grid=param_grid, # The grid of parameters
                            scoring=scoring_metric, # The metric to optimize
                            cv=5, # Number of cross-validation folds
                            n_jobs=-1, # Use all available CPU cores
                            verbose=2) # Print progress updates

    print("\nGridSearchCV object created.")


    # --- 6. Fit the GridSearchCV object on the training data ---
    # This starts the tuning process. GridSearchCV will train the pipeline
    # multiple times internally using cross-validation for each parameter combination.
    print("\nStarting Grid Search hyperparameter tuning...")
    grid_search.fit(train_features, train_labels) # **Fit on the original, UNSCALED training data!**
    print("Grid Search tuning complete.")


    # --- 7. Get the best parameters and best score ---
    print("\nBest hyperparameters found:")
    print(grid_search.best_params_)

    # The best_score_ is the average cross-validation score across the folds
    print(f"\nBest cross-validation score ({scoring_metric}): {grid_search.best_score_:.4f}")
    # Note: This score is negative because we used 'neg_mean_squared_error'.
    # The actual RMSE would be sqrt(-grid_search.best_score_)


    # --- 8. Evaluate the best model on the test set ---
    # grid_search.best_estimator_ is the trained pipeline with the best found parameters
    best_model = grid_search.best_estimator_

    # Predict on the test set (the pipeline handles scaling automatically)
    test_predictions = best_model.predict(test_features)

    # Calculate MAE on the test set using the best model
    mae_test = mean_absolute_error(test_labels, test_predictions)
    print(f"\nMean Absolute Error (MAE) on the test set with best model: {mae_test:.4f}")

    # You can also calculate RMSE if needed:
    rmse_test = mean_squared_error(test_labels, test_predictions, squared=False)
    print(f"Root Mean Squared Error (RMSE) on the test set with best model: {rmse_test:.4f}")

    # Calculate MAPE as you did before (be mindful of zero values in y_test)
    errors_test = abs(test_predictions - test_labels)
    mape_test = 100 * (errors_test / test_labels)
    mape_test = mape_test[np.isfinite(mape_test)] # Handle potential inf/NaN

    accuracy_test = 100 - np.mean(mape_test)
    print(f"Accuracy (100 - Mean MAPE) on the test set with best model: {accuracy_test:.2f} %")



def train_mlp_regressor(train_features, train_labels, test_features, test_labels):
    """
    Train a Multi-Layer Perceptron Regressor model.
    
    Parameters:
    train_data (DataFrame): The training data.
    train_labels (Series): The labels for the training data.
    test_data (DataFrame): The testing data.
    test_labels (Series): The labels for the testing data.
    
    Returns:
    model: The trained model.
    """
    # Instantiate model
    mlp = MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(200, 200,100), 
                       learning_rate_init=0.001 ,
                       learning_rate='adaptive',
                       activation='relu',
                        solver='adam',
                        alpha=0.0001)
    # Train the model on training data
    mlp.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = mlp.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero if test_labels can be zero
    # Using a small epsilon for stability if needed, or filter zeros
    mape = 100 * (errors / test_labels)
    # Handle potential infinities or large values if test_labels are close to zero
    mape = mape[np.isfinite(mape)] # Remove non-finite values

    # Calculate and display accuracy (100 - Mean MAPE)
    # Note: MAPE can be misleading if actual values are close to zero
    accuracy = 100 - np.mean(mape)
    print('MLPRegressor::Accuracy:', round(accuracy, 2), '%.')


if __name__ == "__main__":
    # Initialize the DataCleaner with the path to the dataset
    cleaner = DataCleaner("data/Kangaroo.csv")
    # Clean the data
    #df_cleaned_data = cleaner.clean_data(["Unnamed: 0", "url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 'diningRoomSurface', 
    #                  'streetFacadeWidth', 'gardenOrientation', 'kitchenSurface', 'floorCount', 'hasDiningRoom', 
    #                  'hasDressingRoom', 'hasAttic','terraceOrientation','province','locality'],True)

    #df_cleaned_data.head()

    #cluster_util = DataClustering()
    #cluster_util.cluster_data(df_cleaned_data)

    
    # Print the cleaned data
    #print(df_cleaned_data)
    # Labels are the values we want to predict
    
    
    #prices = np.array(df_cleaned_data['price'])
    #df_cleaned_data = cleaner.preprocess_data()
    

    #df_cleaned_data = df_cleaned_data.drop(['price'], axis=1)

    # Saving feature names for later use
    #feature_list = list(df_cleaned_data.columns)

    # Convert to numpy array
    #features = np.array(df_cleaned_data)  # Uncommented this line to convert features to numpy array

    train_features, test_features, train_labels, test_labels,features_list = cleaner.clean_data(["Unnamed: 0", "url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 'roomCount', 'diningRoomSurface', 
                      'streetFacadeWidth', 'kitchenSurface', 'floorCount', 'hasDiningRoom', 
                      'hasDressingRoom', 'hasAttic'],False)#train_test_split(features, prices, test_size = 0.25, random_state = 42)



    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)
    
    # Train the model
    #model = train_random_forest_regressor(train_features, train_labels, test_features, test_labels)
    # BEST RESULTS:
    model = train_xgboost(train_features, train_labels, test_features, test_labels)
    
    #model = train_mlp_regressor(train_features, train_labels, test_features, test_labels)
    #model = train_mlp(train_features, train_labels, test_features, test_labels)