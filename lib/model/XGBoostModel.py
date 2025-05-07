from lib.DataMetrics import DataMetrics
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import scipy.stats as st
import xgboost as xgb

class ModelNotInitialized(Exception):
    pass

class XGBoostModel:
    """
        Class embbedding XGBoost model.
        Featrures: 
            train a model
            train a model with early stopping
            predict with the current trained model
            save a trained model
            load a trained model

            

        the method predict(..) can be called on an loaded model.
    """
    def __init__(self,verbose=0):
        self.model = None
        self.verbose=verbose

    def log(self,text):
        print("XGBoostModel::"+text)

    def load_model(self, path_to_model):
        """
        Load a previously saved model from a file using Python's pickle module.
        Ensures that the file and its directory exist before attempting to load.

        Parameters:
            path_to_model (str): The file path from which to load the model.

        Raises:
            FileNotFoundError: If the specified file or its directory does not exist.
        """

        # --- 1. Check if directory exists ---
        dir_path = os.path.dirname(path_to_model)
        if dir_path and not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory does not exist: {dir_path}")

        # --- 2. Check if model file exists ---
        if not os.path.isfile(path_to_model):
            raise FileNotFoundError(f"Model file not found: {path_to_model}")

        # --- 3. Load the model using pickle ---
        with open(path_to_model, 'rb') as file:
            self.model = pickle.load(file)
        if self.verbose:
            self.log(f"Model loaded successful from {dir_path}")

    def save_model(self, path_to_model, model=None):
        """
        Saves the trained model to a file using Python's pickle module.
        Ensures the target directory exists before saving.

        Parameters:
            path_to_model (str): Full file path where the model will be saved.
            model (object, optional): The model object to save. If None, saves self.model.

        Raises:
            ModelNotInitialized: If no model is provided and self.model is not set.
        """

        # --- 1. Ensure the directory exists ---
        # Extract directory from full path
        dir_path = os.path.dirname(path_to_model)
        if self.verbose:
            self.log(f"Extracted path : {dir_path}")
        # If directory path is not empty and doesn't exist, create it
        if dir_path and not os.path.exists(dir_path):
            self.log(f"directory path doesn't exist, create it : {dir_path}")
            os.makedirs(dir_path, exist_ok=True)  # Create intermediate directories if needed

        # --- 2. Save the model using pickle ---
        with open(path_to_model, 'wb') as file:
            if model:
                pickle.dump(model, file)  # Save the provided model
            elif self.model:
                pickle.dump(self.model, file)  # Save internal model
            else:
                # Raise an error if neither model is provided nor internal model is set
                raise ModelNotInitialized("ERROR -> model is not initialized")
        self.log(f"Model saved successful under : {path_to_model}")

    def train_with_earlystop(self, train_features, train_labels, test_features, test_labels,hyper_params,num_boost_round = 50000,early_stopping_rounds = 100):
        """
        Trains an XGBoost regression model with early stopping using a validation set.

        Parameters:
            train_features (array-like): Features for training the model.
            train_labels (array-like): Labels corresponding to training features.
            test_features (array-like): Features for evaluating the model.
            test_labels (array-like): Labels corresponding to test features.
            hyper_params (dict): Dictionary of hyperparameters for XGBoost.
            num_boost_round (int): Maximum number of boosting rounds (default is 50000).
            early_stopping_rounds (int): Early stopping if no improvement after these rounds.
            verbose (int): If 1, prints detailed logs and evaluation metrics.

        Returns:
            model (xgb.Booster): Trained XGBoost model.
        """

        # --- 1. Split Training Data into Training and Validation Sets ---
        # This allows monitoring validation performance for early stopping.
        X_train, X_valid, y_train, y_valid = train_test_split(
            train_features, train_labels, test_size=0.15, random_state=42
        )

        # --- 2. Convert data to XGBoost DMatrix format ---
        # DMatrix is optimized for performance during training.
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest  = xgb.DMatrix(test_features, label=test_labels)

        # --- 3. Define Hyperparameters for the XGBoost Model ---
        if hyper_params:
            params = hyper_params
        else:
            params = {
                "objective": "reg:squarederror",  # Use squared error for regression
                "learning_rate": 0.0195,          # Small learning rate for gradual learning
                "max_depth": 5,                   # Depth of each tree
                "subsample": 0.7970,              # Fraction of data used per tree
                "colsample_bytree": 0.2,          # Features sampled per tree
                "colsample_bylevel": 0.4,         # Features sampled per tree level
                "reg_alpha": 1.1,                 # L1 regularization term
                "reg_lambda": 2.25,               # L2 regularization term
                "seed": 42                        # Reproducibility
            }


        # --- 4. Train the Model with Early Stopping ---
        if self.verbose==1:
            self.log(f"\n[INFO] Training XGBoost with early stopping on validation set...")
        evals = [(dtrain, "train"), (dvalid, "valid")]

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50
        )

        # --- 5. Make Predictions on Test and Training Sets using Best Iteration ---
        y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
        train_pred = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))

        # --- 6. Optional: Print Performance Metrics if Verbose ---
        if self.verbose == 1:
            # Compute R2, RMSE, MAE for train and test sets
            train_r2, train_rmse, train_mae = self.get_score(y_train, train_pred)
            test_r2, test_rmse, test_mae = self.get_score(test_labels, y_pred)  # FIXED: previously used undefined y_test

            # Print Model Parameters
            self.log("-" * 42)
            self.log(f"| \t  XGBOOST::PARAMETERS\t\t |")
            self.log("-" * 42)
            self.log(f"| Estimators (used) \t{model.best_iteration + 1}\t\t |")
            self.log("|" + "-" * 40 + "|")
            self.log(f"| Learning rate\t\t{params['learning_rate']}\t\t |")
            self.log(f"| Max depth\t\t{params['max_depth']}\t\t |")
            self.log(f"| Subsample\t\t{params['subsample']}\t\t |")
            self.log(f"| Colsample bytree\t{params['colsample_bytree']}\t\t |")
            self.log(f"| Colsample bylevel\t{params['colsample_bylevel']}\t\t |")
            self.log(f"| Reg alpha\t\t{params['reg_alpha']}\t\t |")
            self.log(f"| Reg lambda\t\t{params['reg_lambda']}\t\t |")
            self.log(f"| Objective\t\t{params['objective']} |")
            self.log("-" * 66)

            # Print Performance Scores
            self.log("-"*68)
            self.log('| XGBOOST::SCORE   \tTRAIN\t\tTEST\t\tDIFF     |')
            self.log("|"+"-"*66+"|")
            self.log(f"| Model R2        \t{train_r2:.4f}\t\t{test_r2:.4f}\t\t{(train_r2-test_r2):.4f}   |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model R2 %      \t{(train_r2*100):.2f}\t\t{(test_r2*100):.2f}\t\t{(train_r2-test_r2) * 100:.2f}     |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model RMSE     \t\t{train_rmse:.2f}\t{test_rmse:.2f}\t{abs(train_rmse-test_rmse):.2f} |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model MAE      \t\t{train_mae:.2f}\t{test_mae:.2f}\t{abs(train_mae-test_mae):.2f} |")
            self.log("-"*68)
            self.log(f"Best iteration used: {model.best_iteration}")

        # --- 7. Return the Trained Model ---
        return model    

    def train(self,train_features, train_labels, test_features, test_labels,
                n_esitmator = 20000,
                learning_rate = np.float64(0.0195),
                random_state = 42,
                max_depth = 5 ,
                subsample = np.float64(0.7970),
                colsample_bytree = np.float64(0.2),
                colsample_bylevel = np.float64(0.4),
                reg_alpha = np.float64(1.1),
                reg_lambda = np.float64(2.25),
                tree_method = 'hist',
                objective = 'reg:squarederror'):
        """
        Train an XGBoost regressor model using the specified training data and evaluate it on test data.

        Parameters:
            train_features (array-like): Features for training.
            train_labels (array-like): Target labels for training.
            test_features (array-like): Features for testing.
            test_labels (array-like): Target labels for testing.
            verbose (int): If 1, print model parameters and performance metrics.

        Returns:
            model (xgb.XGBRegressor): Trained XGBoost model.
        """
        # if model is none, create one 
        if self.model == None:
            # Hyperparameters for XGBoost model
            n_esitmator = 20000  # Number of boosting rounds (trees). Larger numbers increase accuracy but also training time.
            learning_rate = np.float64(0.0195)  # Learning rate (shrinkage). Lower values improve robustness but slow down learning.
            random_state = 42  # Ensures reproducibility by controlling the randomness.
            max_depth = 5  # Maximum depth of a tree. Higher values can lead to overfitting.
            subsample = np.float64(0.7970)  # Fraction of training samples used for each tree (row subsampling).
            colsample_bytree = np.float64(0.2)  # Fraction of features used when building each tree (column subsampling).
            colsample_bylevel = np.float64(0.4)  # Fraction of features used per tree level.
            reg_alpha = np.float64(1.1)  # L1 regularization term (encourages sparsity).
            reg_lambda = np.float64(2.25)  # L2 regularization term (prevents large weights).
            tree_method = 'hist'  # Histogram-based tree building method for speed and memory efficiency.
            objective = 'reg:squarederror'  # Objective function for regression (squared error loss).

            # Initialize XGBoost Regressor with specified hyperparameters
            self.model = xgb.XGBRegressor(objective=objective,n_estimators=n_esitmator, random_state=random_state, 
                                    learning_rate=learning_rate, max_depth=max_depth,
                                    subsample=subsample, colsample_bytree=colsample_bytree,
                                    colsample_bylevel=colsample_bylevel,reg_alpha=reg_alpha,reg_lambda=reg_lambda)
        # Fit model to training data
        self.model.fit(train_features, train_labels)

        # Predict outputs on both training and testing data using a class-defined prediction method
        predictions = self.predict(test_features)
        y_train_pred = self.predict(train_features)

        if self.verbose==1:
            # Compute metrics (R², RMSE, MAE) for both training and test predictions
            train_r2,train_rmse,train_mae = DataMetrics.get_score(train_labels,y_train_pred)
            test_r2,test_rmse,test_mae = DataMetrics.get_score(test_labels,predictions)
            self.log("-"*44)
            self.log(f"| \t  XGBOOST::PARAMETERS\t\t |")
            self.log("-"*44)
            self.log(f"| Estimator\t\t{n_esitmator}\t\t |")
            self.log("|"+"-"*42+"|")
            self.log(f"| Learning rate\t\t{learning_rate}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Random state\t\t{random_state}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Max depth\t\t{max_depth}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Subsample\t\t{subsample}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Colsample bytree\t{colsample_bytree}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Colsample bylevel\t{colsample_bylevel}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Reg alpha\t\t{reg_alpha}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Reg lambda\t\t{reg_lambda}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Tree method\t\t{tree_method}\t\t |") 
            self.log("|"+"-"*42+"|")
            self.log(f"| Objective\t\t{objective} |") 
            self.log("|"+"-"*42+"|")

            
            self.log("-"*68)
            self.log('| XGBOOST::SCORE   \tTRAIN\t\tTEST\t\tDIFF     |')
            self.log("|"+"-"*66+"|")
            self.log(f"| Model R2        \t{train_r2:.4f}\t\t{test_r2:.4f}\t\t{(train_r2-test_r2):.4f}   |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model R2 %      \t{(train_r2*100):.2f}\t\t{(test_r2*100):.2f}\t\t{(train_r2-test_r2) * 100:.2f}     |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model RMSE     \t\t{train_rmse:.2f}\t{test_rmse:.2f}\t{abs(train_rmse-test_rmse):.2f} |")
            self.log("|"+"-"*66+"|")
            self.log(f"| Model MAE      \t\t{train_mae:.2f}\t{test_mae:.2f}\t{abs(train_mae-test_mae):.2f} |")
            self.log("-"*68)
        
        return self.model


    def predict(self,features):
        """
        Make predictions using the trained XGBoost model.

        Parameters:
            features (array-like): The input features for which predictions are to be made.

        Returns:
            predictions (array-like): The predicted values from the model.

        Raises:
            ModelNotInitialized: If the model has not been initialized or trained before calling this method.
        """
        # Check if the model has been initialized (i.e., trained or loaded)
        if self.model:
            # Use the model's predict method to generate predictions for the given features
            predictions = self.model.predict(features)
            return predictions
        else:
            # Raise a custom exception if the model hasn't been initialized
            raise ModelNotInitialized("XGBoostModel::ERROR -> XGBoost model not initialized...")
        
    def affine_xgboost(self,X_train_processed, y_train, param_xgb, n_iter=50, cv=5, scoring='neg_root_mean_squared_error', 
                   objective='reg:squarederror', tree_method='hist'):
        """
        Performs hyperparameter tuning for an XGBoost Regressor using RandomizedSearchCV.

        Example for param_xgb:
            param_dist_xgb = {
                'n_estimators': st.randint(2500, 10000),  # Number of boosting rounds
                'learning_rate': st.loguniform(0.01, 0.5),  # Learning rate (shrinkage)
                'max_depth': st.randint(3, 68),  # Tree depth
                'subsample': st.uniform(0.6, 0.4),  # Row sampling
                'colsample_bytree': st.uniform(0.4, 0.6),  # Feature sampling per tree
                'reg_alpha': st.loguniform(0.01, 1),  # L1 regularization
                'reg_lambda': st.loguniform(0.01, 1),  # L2 regularization
                'colsample_bylevel': st.uniform(0.4, 0.6),  # Feature sampling per level
                'colsample_bynode': st.uniform(0.5, 0.5)  # Feature sampling per node
            }

        Parameters:
            X_train_processed (array-like): Preprocessed training features.
            y_train (array-like): Target values for training.
            param_xgb (dict or None): If provided, custom hyperparameter search space; otherwise defaults are used.
            n_iter (int): Number of parameter combinations to try in RandomizedSearchCV.
            cv (int): Number of folds in cross-validation.
            scoring (str): Scoring metric to optimize ('neg_root_mean_squared_error' by default).
            objective (str): XGBoost objective function (default is regression).
            tree_method (str): Tree growing algorithm; 'hist' is efficient for large datasets.
            verbose (int): Controls verbosity of output (0 = silent, 1 = detailed output).

        Returns:
            best_params_ (dict): Best hyperparameters found.
            best_score_ (float): Best cross-validation score achieved (negative RMSE).
            best_xgb_model (XGBRegressor): Trained model with best parameters.
        """

        # --- 1. Optional: Verbose Logging ---
        if self.verbose:
            self.log("\nAffinage de XGBoost...")  # French: "XGBoost tuning..."

        # --- 2. Define Search Space ---
        # If custom parameter grid is not provided, define a default one
        if param_xgb:
            param_dist_xgb = param_xgb
        else:
            param_dist_xgb = {
                'n_estimators': st.randint(2500, 10000),  # Number of boosting rounds
                'learning_rate': st.loguniform(0.01, 0.5),  # Learning rate (shrinkage)
                'max_depth': st.randint(3, 68),  # Tree depth
                'subsample': st.uniform(0.6, 0.4),  # Row sampling
                'colsample_bytree': st.uniform(0.4, 0.6),  # Feature sampling per tree
                'reg_alpha': st.loguniform(0.01, 1),  # L1 regularization
                'reg_lambda': st.loguniform(0.01, 1),  # L2 regularization
                'colsample_bylevel': st.uniform(0.4, 0.6),  # Feature sampling per level
                'colsample_bynode': st.uniform(0.5, 0.5)  # Feature sampling per node
            }

        # --- 3. Configure RandomizedSearchCV ---
        random_search_xgb = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(
                objective=objective,  # Regression objective
                random_state=42,  # For reproducibility
                tree_method=tree_method  # Use 'hist' for large datasets
            ),
            param_distributions=param_dist_xgb,  # Hyperparameter space
            n_iter=n_iter,  # Number of random combinations to test
            cv=cv,  # Cross-validation folds
            scoring=scoring,  # Metric to optimize (negative RMSE)
            random_state=42,  # Ensures reproducible search
            n_jobs=-1,  # Parallelize across all CPU cores
            verbose=2 if self.verbose == 1 else 0  # Conditional logging
        )

        # --- 4. Perform Hyperparameter Search ---
        random_search_xgb.fit(X_train_processed, y_train)

        # --- 5. Extract the Best Model ---
        best_xgb_model = random_search_xgb.best_estimator_

        # --- 6. Optionally Print the Results ---
        if self.verbose == 1:
            print(f"\nMeilleurs paramètres pour XGBoost: {random_search_xgb.best_params_}")
            print(f"Meilleur score de validation croisée (RMSE): {-random_search_xgb.best_score_:.2f}")  # Negated for readability

        # --- 7. Return Best Params, Score, and Model ---
        return random_search_xgb.best_params_, random_search_xgb.best_score_, best_xgb_model    