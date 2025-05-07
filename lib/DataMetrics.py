from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, make_scorer,mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

class DataMetrics:

    @staticmethod
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