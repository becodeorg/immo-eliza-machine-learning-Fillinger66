import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from lib.DataPipeline import DataPipeline
from lib.DataManager import DataManager
from sklearn.model_selection import train_test_split,KFold,RandomizedSearchCV

from sklearn.metrics import mean_absolute_error, make_scorer,mean_squared_error # Need make_scorer for custom metrics
import numpy as np
import xgboost as xgb
from xgboost import callback
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.linear_model import Ridge # Utilisé comme méta-modèle simple
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping



import scipy.stats as st


def train_stacked_models(X_train_processed,y_train,X_test_processed,y_test):
    print("Entraînement du modèle XGBoost...")
    # Définir et entraîner le modèle XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', # Objectif pour la régression
                                n_estimators=100,
                                learning_rate=0.1,
                                random_state=42)
    xgb_model.fit(X_train_processed, y_train)
    print("Entraînement XGBoost terminé.")

    print("Entraînement du modèle de Réseau de Neurones...")
    # Définir un modèle de Réseau de Neurones simple (architecture basique pour l'exemple)
    # Le nombre d'unités d'entrée correspond au nombre de caractéristiques après prétraitement
    nn_model = Sequential([
        Input(shape=(X_train_processed.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1) # Couche de sortie pour la régression (une seule unité)
    ])

    nn_model.compile(optimizer='adam', loss='mse') # Optimiseur Adam et fonction de perte MSE pour la régression

    # Ajouter un Early Stopping pour éviter le sur-apprentissage (utile sur petits datasets)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Entraîner le réseau de neurones
    # Utiliser une petite partie de l'ensemble d'entraînement comme validation pour l'early stopping
    history = nn_model.fit(X_train_processed, y_train,
                        epochs=100, # Nombre maximum d'époques
                        batch_size=8, # Taille du batch
                        verbose=1, # Supprimer la sortie détaillée de l'entraînement
                        validation_split=0.2, # Utiliser 20% de l'entraînement pour la validation
                        callbacks=[early_stopping],
                        ) # Appliquer l'early stopping
    print("Entraînement Réseau de Neurones terminé.")

    # --- Implémentation du Stacking ---

    print("Implémentation du Stacking...")
    # Utiliser la validation croisée K-Fold pour générer les prédictions out-of-fold
    # Ces prédictions serviront d'ensemble d'entraînement pour le méta-modèle
    kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5 folds

    # Initialiser les tableaux pour stocker les prédictions out-of-fold
    oof_xgb_preds = np.zeros(X_train_processed.shape[0])
    oof_nn_preds = np.zeros(X_train_processed.shape[0])

    # Initialiser les tableaux pour stocker les prédictions sur l'ensemble de test (moyennées sur les folds)
    test_xgb_preds = np.zeros(X_test_processed.shape[0])
    test_nn_preds = np.zeros(X_test_processed.shape[0])

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_processed)):
        print(f"  Traitement du Fold {fold+1}/{kf.n_splits}")
        X_train_fold, X_val_fold = X_train_processed[train_index], X_train_processed[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index] # Utiliser iloc pour l'indexation

        # Entraîner les modèles de base sur les données d'entraînement du fold
        xgb_fold = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42,verbose=1)
        xgb_fold.fit(X_train_fold, y_train_fold)

        nn_fold = Sequential([
            Input(shape=(X_train_fold.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        nn_fold.compile(optimizer='adam', loss='mse')
        # Entraîner le NN du fold, en utilisant l'ensemble de validation du fold pour l'early stopping
        nn_fold.fit(X_train_fold, y_train_fold, epochs=100, batch_size=8, verbose=1,
                    validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping],)


        # Générer les prédictions out-of-fold
        oof_xgb_preds[val_index] = xgb_fold.predict(X_val_fold)
        oof_nn_preds[val_index] = nn_fold.predict(X_val_fold).flatten() # Aplatir la sortie du NN

        # Générer les prédictions sur l'ensemble de test et les additionner pour faire une moyenne plus tard
        test_xgb_preds += xgb_fold.predict(X_test_processed) / kf.n_splits
        test_nn_preds += nn_fold.predict(X_test_processed).flatten() / kf.n_splits # Aplatir la sortie du NN

    # Préparer les données d'entraînement pour le méta-modèle
    X_meta_train = np.column_stack((oof_xgb_preds, oof_nn_preds))
    y_meta_train = y_train # La cible pour le méta-modèle est la cible originale

    # Préparer les données de test pour le méta-modèle
    X_meta_test = np.column_stack((test_xgb_preds, test_nn_preds))

    # Entraîner le méta-modèle
    print("Entraînement du Méta-modèle (Ridge)...")
    meta_model = Ridge() # Un modèle linéaire simple fonctionne bien comme méta-modèle
    meta_model.fit(X_meta_train, y_meta_train)
    print("Entraînement du Méta-modèle terminé.")

    # Faire les prédictions finales avec le modèle empilé
    stacked_preds = meta_model.predict(X_meta_test)
    print("Stacking terminé.")

    # --- Évaluation ---

    print("\n--- Évaluation des Modèles ---")

    # Évaluer les modèles individuels sur l'ensemble de test
    xgb_test_preds = xgb_model.predict(X_test_processed)
    nn_test_preds = nn_model.predict(X_test_processed).flatten() # Aplatir la sortie du NN

    # Évaluation XGBoost
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_preds))
    xgb_r2 = r2_score(y_test, xgb_test_preds)
    print(f"XGBoost Test RMSE: {xgb_rmse:.2f}")
    print(f"XGBoost Test R2 Score: {xgb_r2:.2f}")

    # Évaluation Réseau de Neurones
    nn_rmse = np.sqrt(mean_squared_error(y_test, nn_test_preds))
    nn_r2 = r2_score(y_test, nn_test_preds)
    print(f"Réseau de Neurones Test RMSE: {nn_rmse:.2f}")
    print(f"Réseau de Neurones Test R2 Score: {nn_r2:.2f}")

    # Évaluation du Modèle Empilé (Stacked)
    stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_preds))
    stacked_r2 = r2_score(y_test, stacked_preds)
    print(f"Modèle Stacked Test RMSE: {stacked_rmse:.2f}")
    print(f"Modèle Stacked Test R2 Score: {stacked_r2:.2f}")


def affine_xgboost(X_train_processed,y_train,X_test_processed,y_test):
    # --- 1. Affinage de XGBoost ---
    print("\nAffinage de XGBoost...")

    # Définir l'espace de recherche des paramètres pour Randomized Search
    # Ces distributions et valeurs sont des exemples, ajustez-les en fonction de vos connaissances et ressources
    param_dist_xgb = {
        # Nombre d'arbres : Plage raisonnable pour débuter. L'early stopping est aussi une bonne pratique
        # si vous mettez une valeur max plus élevée comme 5000+.
        'n_estimators': st.randint(2500, 10000), # Augmenté la plage max pour mieux exploiter les données
        # Taux d'apprentissage : Plage log-uniforme classique. 1 est souvent trop haut, 0.3 est plus sûr comme max.
        'learning_rate': st.loguniform(0.01, 0.5), # Resserrement de la plage supérieure
        # Profondeur max : Limite la complexité de chaque arbre. Crucial avec bcp de features pour éviter le sur-apprentissage.
        # Des arbres trop profonds sur 5000+ features peuvent sur-apprendre le bruit.
        'max_depth': st.randint(3, 68), # Resserrement vers des profondeurs légèrement plus faibles
        # Subsample : Fraction des échantillons par arbre. Ajoute de la robustesse. La plage 0.6-1.0 est standard.
        'subsample': st.uniform(0.6, 0.4), # Reste identique, plage [0.6, 1.0]
        # Colsample_bytree : Fraction des features par arbre. ESSENTIEL avec 5100 colonnes pour éviter le sur-apprentissage.
        # Il est souvent bénéfique d'utiliser une fraction plus faible que 1.0. Explorer des valeurs < 0.6 est pertinent ici.
        'colsample_bytree': st.uniform(0.4, 0.6), # Plage élargie et déplacée vers [0.4, 1.0] pour explorer des fractions plus petites
        # Régularisation L1 et L2 : Très importantes pour contrôler la complexité et le sur-apprentissage avec bcp de features.
        # La plage 1e-3 à 1 est un bon point de départ.
        'reg_alpha': st.loguniform(0.01, 1),
        'reg_lambda': st.loguniform(0.01, 1),
        
        'colsample_bylevel': st.uniform(0.4, 0.6), # 
        'colsample_bynode': st.uniform(0.5, 0.5) # 
    }


    # Configurer Randomized Search pour XGBoost
    # n_iter: nombre de combinaisons à tester (plus c'est grand, plus c'est long)
    # cv: nombre de folds pour la validation croisée
    # scoring: métrique à optimiser (RMSE dans ce cas, sklearn optimise par défaut la métrique pour le modèle)
    # random_state: pour la reproductibilité
    # verbose: niveau de verbosité pendant la recherche
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, tree_method='hist'), # tree_method='hist' est recommandé pour les grands datasets/bcp de features
        param_distributions=param_dist_xgb,
        n_iter=50, # Augmenté de 50 à 200 (Minimum suggéré pour cette taille/dimensionnalité. Plus = potentiellement mieux, mais plus long)
        cv=5, # Augmenté de 3 à 5 folds pour une meilleure estimation de la performance
        scoring='neg_root_mean_squared_error', # Optimiser le négatif du RMSE (sklearn minimise)
        random_state=42,
        n_jobs=-1, # Utiliser tous les cœurs CPU disponibles pour accélérer la recherche
        verbose=2 # Afficher les détails de la recherche pendant l'exécution
    )

    # Lancer la recherche
    random_search_xgb.fit(X_train_processed, y_train)

    # Meilleur modèle trouvé
    best_xgb_model = random_search_xgb.best_estimator_
    print(f"\nMeilleurs paramètres pour XGBoost: {random_search_xgb.best_params_}")
    print(f"Meilleur score de validation croisée (RMSE) pour XGBoost: {-random_search_xgb.best_score_:.2f}")

def train_neural_network(X_train_processed,y_train,X_test_processed,y_test):
   

    # 1. Création du modèle de réseau de neurones
    model = models.Sequential([
        layers.Input(shape=(X_train_processed.shape[1],)),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # 2. Compilation du modèle
    model.compile(
        optimizer='adam',
        loss='mse',   # Erreur quadratique moyenne
        metrics=['mae']  # Erreur absolue moyenne (plus interprétable en €)
    )

    

    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(X_train_processed, y_train,
          validation_split=0.2,
          epochs=200,
          batch_size=32,
          callbacks=[early_stop],
          verbose=1)

    # 3. Entraînement du modèle
    #history = model.fit(
    #    X_train_processed, y_train,
    #    validation_split=0.2,
    #    epochs=20,
    #    batch_size=32,
    #    verbose=1
    #)

    # 4. Évaluation sur les données de test
    test_loss, test_mae = model.evaluate(X_test_processed, y_test)
    test_loss, test_mae

def train_xgboost_earlystop(train_features, train_labels, test_features, test_labels):

    X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_labels, test_size=0.15, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(test_features, label=test_labels)

    # Parameters
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.0155,
        "max_depth": 5,
        "subsample": 0.7970,
        "colsample_bytree": 0.2,#0.3092,
        "colsample_bylevel": 0.4,#0.0649,
        "reg_alpha": 1.1,#0.04384,
        "reg_lambda": 2.25,#0.0711,
        "seed": 42
    }


    num_boost_round = 20000
    early_stopping_rounds = 100

    # Train Model with Early Stopping
    print(f"\n[INFO] Training XGBoost with early stopping on validation set...")
    evals = [(dtrain, "train"), (dvalid, "valid")]

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50
    )

    # --- 4. Evaluate on Test Set ---
    y_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    train_pred = model.predict(dtrain, iteration_range=(0, model.best_iteration + 1))

    train_r2, train_rmse, train_mae = get_score(y_train, train_pred)
    test_r2, test_rmse, test_mae = get_score(y_test, y_pred)

    print("-" * 42)
    print(f"| \t  XGBOOST::PARAMETERS\t\t |")
    print("-" * 42)
    print(f"| Estimators (used) \t{model.best_iteration + 1}\t\t |")
    print("|" + "-" * 40 + "|")
    print(f"| Learning rate\t\t{params['learning_rate']}\t\t |")
    print(f"| Max depth\t\t{params['max_depth']}\t\t |")
    print(f"| Subsample\t\t{params['subsample']}\t\t |")
    print(f"| Colsample bytree\t{params['colsample_bytree']}\t\t |")
    print(f"| Colsample bylevel\t{params['colsample_bylevel']}\t\t |")
    print(f"| Reg alpha\t\t{params['reg_alpha']}\t\t |")
    print(f"| Reg lambda\t\t{params['reg_lambda']}\t\t |")
    print(f"| Objective\t\t{params['objective']} |")
    print("-" * 66)
    print('| XGBOOST::SCORE   \tTRAIN\t\tTEST\t\tDIFF\t |')
    print("|" + "-" * 64 + "|")
    print(f"| Model R2        \t{train_r2:.4f}\t\t{test_r2:.4f}\t\t{(train_r2 - test_r2):.4f}\t |")
    print(f"| Model R2 %      \t{(train_r2 * 100):.2f}\t\t{(test_r2 * 100):.2f}\t\t{(train_r2 - test_r2) * 100:.2f}\t |")
    print(f"| Model RMSE      \t{train_rmse:.2f}\t{test_rmse:.2f}\t{abs(train_rmse - test_rmse):.2f} |")
    print(f"| Model MAE       \t{train_mae:.2f}\t{test_mae:.2f}\t{abs(train_mae - test_mae):.2f} |")
    print("-" * 66)
    print(f"Best iteration used: {model.best_iteration}")
    return model
    
def train_xgboost(train_features, train_labels, test_features, test_labels):
    
    early_stopping = xgb.callback.EarlyStopping(rounds=20,save_best=True, metric_name="rmse")
 
    n_esitmator=2800#2875 # The number of trees in the ensemble. More trees can lead to better performance but also longer training time.
    learning_rate=np.float64(0.0355) # The step size shrinkage used in update to prevent overfitting. Lower values make the model more robust but require more trees.
    random_state=42
    max_depth=6 # The maximum depth of the tree. Increasing this value will make the model more complex and more likely to overfit.
    subsample=np.float64(0.7970) # These control the proportion of data and features used to build each individual tree. This is a form of regularization.
    colsample_bytree=np.float64(0.3092) # The subsample ratio of columns when constructing each tree. This is another form of regularization.
    reg_alpha=np.float64(0.04384)
    reg_lambda=np.float64(0.8711)
    colsample_bylevel=np.float64(0.0649)
    tree_method='hist' # The tree construction algorithm used in XGBoost. 'hist' is faster and more memory efficient for large datasets.
    objective='reg:squarederror'

    model = xgb.XGBRegressor(objective=objective,n_estimators=n_esitmator, random_state=random_state, 
                             learning_rate=learning_rate, max_depth=max_depth,
                             subsample=subsample, colsample_bytree=colsample_bytree,
                             colsample_bylevel=colsample_bylevel,reg_alpha=reg_alpha,reg_lambda=reg_lambda)
    """
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             colsample_bylevel= np.float64(0.7669918962929685), 
                             colsample_bynode= np.float64(0.5035331526098588), 
                             colsample_bytree= np.float64(0.4138374550248495), 
                             learning_rate= np.float64(0.05958758776628972), 
                             max_depth= 5, 
                             n_estimators=2705, 
                             reg_alpha= np.float64(0.012397420340784145),
                            reg_lambda= np.float64(0.8861577452533068),
                            subsample= np.float64(0.6931085361721216))
    """
    eval_set = [(train_features, test_features), (train_labels, test_labels)]
    
    model.fit(train_features, train_labels)


    #model.fit(train_features, train_labels)
    # Use the forest's predict method on the test data
    predictions = model.predict(test_features)

    # --- Calcul des Métriques sur l'Ensemble d'Entraînement ---
    y_train_pred = model.predict(X_train_processed)


    train_r2,train_rmse,train_mae = get_score(y_train,y_train_pred)
    test_r2,test_rmse,test_mae = get_score(test_labels,predictions)

    print("-"*42)
    print(f"| \t  XGBOOST::PARAMETERS\t\t |")
    print("-"*42)
    print(f"| Estimator\t\t{n_esitmator}\t\t |")
    print("|"+"-"*40+"|")
    print(f"| Learning rate\t\t{learning_rate}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Random state\t\t{random_state}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Max depth\t\t{max_depth}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Subsample\t\t{subsample}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Colsample bytree\t{colsample_bytree}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Colsample bylevel\t{colsample_bylevel}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Reg alpha\t\t{reg_alpha}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Reg lambda\t\t{reg_lambda}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Tree method\t\t{tree_method}\t\t |") 
    print("|"+"-"*40+"|")
    print(f"| Objective\t\t{objective} |") 
    print("|"+"-"*40+"|")

    
    print("-"*65)
    print('| XGBOOST::SCORE   \tTRAIN\t\tTEST\t\tDIFF\t|')
    print("|"+"-"*63+"|")
    print(f"| Model R2        \t{train_r2:.4f}\t\t{test_r2:.4f}\t\t{(train_r2-test_r2):.4f}\t|")
    print("|"+"-"*63+"|")
    print(f"| Model R2 %      \t{(train_r2*100):.2f}\t\t{(test_r2*100):.2f}\t\t{(train_r2-test_r2) * 100:.2f}\t|")
    print("|"+"-"*63+"|")
    print(f"| Model RMSE     \t{train_rmse:.2f}\t{test_rmse:.2f}\t{abs(train_rmse-test_rmse):.2f} |")
    print("|"+"-"*63+"|")
    print(f"| Model MAE      \t{train_mae:.2f}\t{test_mae:.2f}\t{abs(train_mae-test_mae):.2f} |")
    print("-"*65)
    
    return model


def get_score(test_labels, predictions,to_print=False):
    # 1. Calculate R2 Score
    r2 = r2_score(test_labels, predictions)
    # 2. Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    # 3. Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(test_labels, predictions)
    # --- Affichage des Résultats ---
    if to_print:
        print(f"Model Test R2 Score : {r2:.4f}") 
        print(f"Model Test R2 (en %) : {r2 * 100:.2f} %")
        print(f"Model Test RMSE : {rmse:.2f} €")
        print(f"Model Test MAE : {mae:.2f} €")

    return r2,rmse,mae

if __name__ == "__main__":


    df = DataManager.load_csv("data/Kangaroo.csv",True)

    df = DataManager.merge_columnsFrom(df,"data/Giraffe.csv","id","propertyId",['latitude', 'longitude','cadastralIncome','primaryEnergyConsumptionPerSqm'],True)

   
    df = DataManager.get_lat_lng_for_zipcode(df=df,debug=True)

    unrevelant_columns = ["Unnamed: 0", "id","url",'monthlyCost', 'hasBalcony', 'accessibleDisabledPeople', 
                          'roomCount', 'diningRoomSurface', 'streetFacadeWidth', 'kitchenSurface', 
                          'floorCount', 'hasDiningRoom', 'hasDressingRoom', 'hasAttic','diningRoomSurface',
                          'hasLivingRoom','livingRoomSurface','gardenOrientation','hasBasement',
                          'streetFacadeWidth','kitchenSurface']

    print("Pipeline creation")
    pipeline = DataPipeline(df,"price")


    #pipeline.determine_kmean(unrevelant_columns)

    #pipeline.determine_kmean_by_elbow(unrevelant_columns)

    
    print("Pipeline execution..")#'Unnamed: 0', 'id', 'url','monthlyCost','hasBalcony','accessibleDisabledPeople'
    X_train_processed, X_test_processed, y_train, y_test, features_list = pipeline.execute_pipeline(
                                                                                unreavelant_columns=unrevelant_columns,
                                                                                k=3,
                                                                                debug=True)
    
    

    print("Train XGBoost model...")
    #model = train_xgboost(X_train_processed,y_train,X_test_processed,y_test)
    model = train_xgboost_earlystop(X_train_processed,y_train,X_test_processed,y_test)


    #print("Train neural network")
    #neuronal_model = train_neural_network(X_train_processed,y_train,X_test_processed,y_test)


    #print("Train stacked models")
    #stacked_model = train_stacked_models(X_train_processed,y_train,X_test_processed,y_test)

    #print("Affine Xgboost ")
    #affine_xgboost(X_train_processed,y_train,X_test_processed,y_test)
    