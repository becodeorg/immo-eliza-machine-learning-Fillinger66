import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans # Re-import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score # Importation du score de silhouette
import matplotlib.pyplot as plt
import geopy
from geopy.distance import geodesic

class DataClustering:

    # __init__ remains without city_center_coords
    def __init__(self):
        """
        Initializes the DataClustering class.
        """
        pass # No longer needs city_center_coords

    # cluster_data remains without city_center_coords parameter
    def cluster_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds distance and cluster features based on location, including:
        - Distance to zipcode centroid
        - Latitude/longitude interaction
        - K-Means location cluster ID
        - Distance to the assigned K-Means cluster centroid.

        Parameters:
        df (pd.DataFrame): DataFrame containing property data with latitude,
                           longitude, zipcode_Latitude, and zipcode_Longitude.

        Returns:
        pd.DataFrame: DataFrame with additional columns for distances and cluster labels.
        """

        print("--- Starting Location Feature Engineering ---")

        # Ensure necessary columns exist and are numeric before calculations
        location_cols_zip = ['latitude', 'longitude', 'zipcode_Latitude', 'zipcode_Longitude']
        if not all(col in df.columns for col in location_cols_zip):
             print(f"Warning: Missing one or more required zipcode location columns: {location_cols_zip}. Skipping zipcode distance calculation.")
             # Add NaN column as placeholder if columns are missing
             df['distance_to_zipcode_km'] = np.nan
        else:
            # Calculate distance between property coordinates and zipcode coordinates
            # This is the distance to the "centroid" of the zipcode "cluster"
            print("Calculating property-zipcode distance (distance to zipcode centroid)...")
            df['distance_to_zipcode_km'] = df.apply(self.calculate_distance_to_zipcode, axis=1)
            print("Distance to zipcode centroid feature created.")
            # Optional: Display head only if columns exist
            # print(df[['latitude', 'longitude', 'zipcode_Latitude', 'zipcode_Longitude', 'distance_to_zipcode_km']].head())


        # Ensure lat/lon exist for interaction term
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Handle potential NaNs in the product (multiplication with NaN results in NaN)
            df['lat_lon_interaction'] = df['latitude'] * df['longitude']
            print("Latitude * Longitude interaction feature created.")
            # Optional: Display head only if columns exist
            # print(df[['latitude', 'longitude', 'lat_lon_interaction']].head())
        else:
            print("Skipping lat*lon interaction: 'latitude' or 'longitude' columns missing.")

        # Removed: Distance to city center calculation

        # --- Re-introducing K-Means Clustering ---
        cluster_cols = ['latitude', 'longitude']
        # K-Means cannot handle missing values. Need to impute or remove.
        # Imputing is generally preferred to keep all rows.
        # Let's create a temporary DataFrame for clustering and imputation
        clustering_data = df[cluster_cols].copy()

        # Check if there are any non-NaN values to cluster
        if clustering_data.dropna().empty:
             print(f"Warning: No non-missing data in {cluster_cols} for clustering. Skipping K-Means and distance to centroid.")
             # Add placeholder columns with NaNs if clustering is skipped
             df['location_cluster'] = np.nan
             df['distance_to_cluster_centroid_km'] = np.nan
        else:
            # Use a pipeline for imputation and scaling before clustering
            # Scaling is important for K-Means as it's distance-based
            clustering_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')), # Impute NaNs with the mean
                ('scaler', StandardScaler()),
                ('kmeans', KMeans(n_clusters=20, # <--- Choose the number of clusters (K) - Tune this!
                                 random_state=42,
                                 n_init=10)) # Number of initializations to run
            ])

            print(f"Performing K-Means clustering on {cluster_cols}...")
            # Fit the pipeline and get cluster labels
            # The pipeline handles fitting imputer, scaler, and kmeans on the data
            df['location_cluster'] = clustering_pipeline.fit_predict(clustering_data)

            print("Location cluster feature created.")
            # Convert cluster labels to object/category type if you want to treat them as nominal categories
            # (Often better for tree models than treating them as ordered numbers)
            df['location_cluster'] = df['location_cluster'].astype('category')
            # Optional: Display head only if columns exist
            # print(df[['latitude', 'longitude', 'location_cluster']].head())
            print(f"Number of unique clusters: {df['location_cluster'].nunique()}")

            # --- Calculate Distance to K-Means Cluster Centroid ---
            print("Calculating distance to assigned K-Means cluster centroid...")

            # Get the cluster centers from the fitted KMeans model
            # These are in the scaled space if StandardScaler was used
            # cluster_centers_scaled = clustering_pipeline.named_steps['kmeans'].cluster_centers_

            # To calculate distance in original lat/lon space, we need the centroids
            # in the original scale. We can get these by grouping the imputed data
            # (before scaling) by the assigned cluster ID and calculating the mean.

            # Get the imputed data from the pipeline's imputer step
            imputed_data = clustering_pipeline.named_steps['imputer'].transform(clustering_data)
            # Convert imputed data back to DataFrame to easily group by cluster
            imputed_df = pd.DataFrame(imputed_data, columns=cluster_cols, index=df.index)
            imputed_df['location_cluster'] = df['location_cluster'] # Add cluster labels

            # Calculate centroids on the imputed data (original scale)
            # Group by cluster and calculate the mean lat/lon for each cluster
            #centroids_in_original_scale = imputed_df.groupby('location_cluster')[cluster_cols].mean()
            centroids_in_original_scale = imputed_df.groupby('location_cluster', observed=False)[cluster_cols].mean()
            # Define a function to get the centroid for a given row's cluster
            def get_centroid_coords(row, centroids_df):
                 cluster_id = row['location_cluster']
                 # Handle cases where cluster_id might be NaN if clustering was skipped for some rows
                 if pd.isna(cluster_id):
                     return np.nan, np.nan

                 # Ensure the cluster_id exists in the centroids_df index
                 # This shouldn't happen if clustering was successful, but as a safeguard
                 if cluster_id not in centroids_df.index:
                     # print(f"Warning: Cluster ID {cluster_id} not found in centroids.") # Too verbose
                     return np.nan, np.nan

                 centroid_lat = centroids_df.loc[cluster_id, 'latitude']
                 centroid_lon = centroids_df.loc[cluster_id, 'longitude']
                 return centroid_lat, centroid_lon

            # Apply a function to calculate distance for each row
            def calculate_distance_to_assigned_centroid(row, centroids_df):
                 prop_coords = (row['latitude'], row['longitude'])
                 centroid_lat, centroid_lon = get_centroid_coords(row, centroids_df)

                 # Handle missing property coordinates or missing centroid coordinates
                 if pd.isna(prop_coords[0]) or pd.isna(prop_coords[1]) or pd.isna(centroid_lat) or pd.isna(centroid_lon):
                     return np.nan

                 try:
                     return geodesic(prop_coords, (centroid_lat, centroid_lon)).km
                 except ValueError:
                     # Handle potential errors with invalid coordinates
                     return np.nan

            # Apply the distance calculation
            df['distance_to_cluster_centroid_km'] = df.apply(
                lambda row: calculate_distance_to_assigned_centroid(row, centroids_in_original_scale),
                axis=1
            )

            print("Distance to K-Means cluster centroid feature created.")
            # Optional: Display head
            # print(df[['latitude', 'longitude', 'location_cluster', 'distance_to_cluster_centroid_km']].head())


        print("--- Finished Location Feature Engineering ---")
        # Display info to see the new columns and potential NaNs
        # print(df.info())
        # Look at descriptive statistics for new numerical columns
        # print(df[['distance_to_zipcode_km', 'lat_lon_interaction', 'distance_to_cluster_centroid_km']].describe())
        # Look at value counts for the new categorical cluster feature
        # if 'location_cluster' in df.columns:
        #      print(df['location_cluster'].value_counts().head())


        # Return the modified DataFrame
        return df

    # Removed: calculate_distance_to_city_center method

    # Define a method to calculate distance between property and zipcode coordinates
    # Renamed from calculate_distance for clarity
    def calculate_distance_to_zipcode(self, row) -> float:
        """
        Calculates the geodesic distance between property coordinates and zipcode coordinates for a row.
        """
        # Ensure required columns exist in the row before accessing
        required_cols = ['latitude', 'longitude', 'zipcode_Latitude', 'zipcode_Longitude']
        if not all(col in row.index for col in required_cols):
             # print("Warning: Missing required columns in row for zipcode distance calculation.") # Too verbose
             return np.nan # Return NaN if columns are missing in the row

        prop_coords = (row['latitude'], row['longitude'])
        zip_coords = (row['zipcode_Latitude'], row['zipcode_Longitude'])

        # Handle cases where coordinates are missing (will return NaN)
        if pd.isna(prop_coords[0]) or pd.isna(prop_coords[1]) or pd.isna(zip_coords[0]) or pd.isna(zip_coords[1]):
            return np.nan

        try:
            # Distance in kilometers
            return geodesic(prop_coords, zip_coords).km
        except ValueError:
            # Handle potential errors with invalid coordinates (e.g., 0,0 or out of range)
            return np.nan

    
    def clusterize_zipcode_sqrt_meter(self, X_train, X_test, k, debug=False):
        """
        Calculate and cluster average price per square meter (€/m²) by postal code.

        This function:
        - Computes the average price per square meter for each postal code in the training set.
        - Applies K-Means clustering on these averages to group postal codes by similar price levels.
        - Merges the resulting cluster labels and average prices back into both X_train and X_test.

        Parameters:
        ----------
        X_train : pd.DataFrame
            Training dataset containing 'postCode', 'price', and 'habitableSurface' columns.

        X_test : pd.DataFrame
            Test dataset to be enriched with the same clustering logic as the training set.

        k : int
            Number of clusters to use in the K-Means algorithm.

        debug : bool, optional (default=False)
            If True, print debug information including sample outputs from the enriched datasets.

        Returns:
        -------
            X_train_enriched : pd.DataFrame
                The training dataset enriched with 'avg_price_per_sqm' and 'cluster_label'.

            X_test_enriched : pd.DataFrame
                The test dataset enriched with 'avg_price_per_sqm' and 'cluster_label'.
        """
        print(f"\nDataClustering::AVG_PRICE_PER_SQM -> X_Train : {X_train.shape}")
        print(f"\nDataClustering::AVG_PRICE_PER_SQM -> X_Test : {X_test.shape}")
        # Copy relevant columns from training data to avoid modifying original data
        df_price_sqm_calc = X_train[['postCode', 'price', 'habitableSurface']].copy()

        # Drop rows with missing critical values
        df_price_sqm_calc.dropna(subset=['postCode', 'price', 'habitableSurface'], inplace=True)

        # Filter out rows with 0 surface to avoid division by zero
        df_price_sqm_calc = df_price_sqm_calc[df_price_sqm_calc['habitableSurface'] > 0].copy()

        # Ensure correct data types
        df_price_sqm_calc['price'] = pd.to_numeric(df_price_sqm_calc['price'])
        df_price_sqm_calc['habitableSurface'] = pd.to_numeric(df_price_sqm_calc['habitableSurface'])
        df_price_sqm_calc['postCode'] = df_price_sqm_calc['postCode'].astype(str)

        # --- 2. Calculate average price per square meter per postcode ---
        price_sqm_by_postcode = df_price_sqm_calc.groupby('postCode').agg(
            total_price=('price', 'sum'),
            total_surface=('habitableSurface', 'sum')
        ).reset_index()

        price_sqm_by_postcode['avg_price_per_sqm'] = price_sqm_by_postcode['total_price'] / price_sqm_by_postcode['total_surface']

        # --- 3. Prepare data for clustering (K-Means) ---

        # Extract the feature to cluster (avg_price_per_sqm) as a 2D NumPy array
        X_clustering = price_sqm_by_postcode[['avg_price_per_sqm']].values

        # Standardize the feature for K-Means (important for clustering quality)
        scaler_clustering = StandardScaler()
        X_scaled_clustering = scaler_clustering.fit_transform(X_clustering)

        # Apply K-Means clustering with k clusters (value passed as parameter)
        print(f"\nDataClustering::AVG_PRICE_PER_SQM -> Application de K-Means sur les codes postaux avec {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        price_sqm_by_postcode['cluster_label'] = kmeans.fit_predict(X_scaled_clustering)
        print("DataClustering::AVG_PRICE_PER_SQM -> Clustering des codes postaux terminé.")

        # --- 4. Merge clustering results back into original datasets ---

        # Select only relevant columns to merge back
        postcode_enrichment_map = price_sqm_by_postcode[['postCode', 'avg_price_per_sqm', 'cluster_label']].copy()

        # Ensure postcode columns are strings for a safe merge
        postcode_enrichment_map['postCode'] = postcode_enrichment_map['postCode'].astype(str)
        X_train['postCode'] = X_train['postCode'].astype(str)
        X_test['postCode'] = X_test['postCode'].astype(str)

        # Merge the enriched information separately into training and test sets
        X_train_enriched = pd.merge(X_train, postcode_enrichment_map, on='postCode', how='left')
        X_test_enriched = pd.merge(X_test, postcode_enrichment_map, on='postCode', how='left')

        # Optional debug prints
        if debug:
            print(f"DataClustering::AVG_PRICE_PER_SQM -> X_train_enriched head: {X_train_enriched.shape}")
            print(X_train_enriched[['postCode', 'avg_price_per_sqm', 'cluster_label']].head(50))
            
            print(f"\nDataClustering::AVG_PRICE_PER_SQM ->X_test_enriched head: {X_test_enriched.shape}")
            print(X_test_enriched[['postCode', 'avg_price_per_sqm', 'cluster_label']].head(50))

        # Return enriched datasets
        return X_train_enriched, X_test_enriched


    def calculate_price_pr_sqm(self,df):
        """
        Calculates the average price per square meter (price/sqm) for each postcode
        based on available property data and merges this information back into the
        original DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing property data
                               with at least 'postCode', 'price', and 'habitableSurface' columns.

        Returns:
            pd.DataFrame: The original DataFrame with an added column 'avg_price_per_sqm'.
        """
        # Create a copy of the relevant columns to perform calculations without modifying the original df yet.
        # This is good practice to avoid unintended side effects.
        df_price_sqm_calc = df[['postCode', 'price', 'habitableSurface']].copy()

        # Drop rows where any of the essential columns for calculation ('postCode', 'price', 'habitableSurface') are missing (NaN).
        # Using inplace=True modifies df_price_sqm_calc directly.
        df_price_sqm_calc.dropna(subset=['postCode', 'price', 'habitableSurface'], inplace=True)

        # Filter out rows where 'habitableSurface' is zero or less, as these cannot be used to calculate price per square meter.
        # Using .copy() after filtering ensures this is a new DataFrame slice to prevent SettingWithCopyWarning later.
        df_price_sqm_calc = df_price_sqm_calc[df_price_sqm_calc['habitableSurface'] > 0].copy()

        # Convert the 'price' and 'habitableSurface' columns to numeric types.
        # This is necessary to perform mathematical operations (summation and division).
        # pd.to_numeric is robust and can handle potential non-numeric entries by raising errors or coercing to NaN if specified (though dropna already handled NaNs).
        df_price_sqm_calc['price'] = pd.to_numeric(df_price_sqm_calc['price'])
        df_price_sqm_calc['habitableSurface'] = pd.to_numeric(df_price_sqm_calc['habitableSurface'])

        # Convert the 'postCode' column to string type.
        # This ensures consistent data types for the grouping and merging operations,
        # preventing potential mismatches if postcodes were originally stored as numbers or mixed types.
        df_price_sqm_calc['postCode'] = df_price_sqm_calc['postCode'].astype(str)

        # --- Calculate the average price per square meter for each postcode ---
        # Group the filtered data by 'postCode'.
        # Use the .agg() method to perform multiple aggregations simultaneously for each group.
        # 'total_price': Calculate the sum of 'price' for all properties within each postcode.
        # 'total_surface': Calculate the sum of 'habitableSurface' for all properties within each postcode.
        # .reset_index() converts the grouped output (which has 'postCode' as index) back into a regular DataFrame column.
        price_sqm_by_postcode = df_price_sqm_calc.groupby('postCode').agg(
            total_price=('price', 'sum'),
            total_surface=('habitableSurface', 'sum')
        ).reset_index()

        # Calculate the weighted average price per square meter for each postcode.
        # This is done by dividing the total price by the total habitable surface for all properties in that postcode.
        # This gives a more representative average price per sqm than just averaging the individual property price/sqm.
        price_sqm_by_postcode['avg_price_per_sqm'] = price_sqm_by_postcode['total_price'] / price_sqm_by_postcode['total_surface']

        # Select only the 'postCode' and the calculated 'avg_price_per_sqm' columns.
        # Use .copy() to create a separate DataFrame to prevent SettingWithCopyWarning during the merge preparation.
        postcode_enrichment = price_sqm_by_postcode[['postCode', 'avg_price_per_sqm']].copy()

        # --- Prepare for Merging ---
        # Ensure the 'postCode' column in the dataframe containing the calculated averages is of string type.
        # This is a redundant step if already done above, but harmless and ensures type consistency before merge.
        postcode_enrichment['postCode'] = postcode_enrichment['postCode'].astype(str)

        # Ensure the 'postCode' column in the ORIGINAL input dataframe (df) is also of string type.
        # This is CRUCIAL for the merge operation to correctly match postcodes between the two dataframes.
        df['postCode'] = df['postCode'].astype(str)

        # --- Merge the calculated average price per sqm back into the original DataFrame ---
        # Use pd.merge() to combine the original dataframe (df) with the postcode_enrichment dataframe.
        # on='postCode': Specifies the column to use for matching rows between the two dataframes.
        # how='left': Performs a left merge. This means all rows from the original 'df' are kept.
        #             If a postcode in 'df' does not have a corresponding entry in 'postcode_enrichment'
        #             (e.g., because all properties for that postcode had missing data or zero surface),
        #             the 'avg_price_per_sqm' column for that row will be NaN.
        df_enriched = pd.merge(df, postcode_enrichment, on='postCode', how='left')

        # Return the original DataFrame now enriched with the 'avg_price_per_sqm' column.
        return df_enriched


    def determine_kmean_by_silhouete(self,df):
        
        df = self.calculate_price_pr_sqm(df)

        # Préparer les données pour le clustering (utiliser le prix moyen/m²)
        # Utiliser .values pour obtenir un tableau numpy 2D
        X_clustering = df[['avg_price_per_sqm']].values

        # Mettre à l'échelle les données pour K-Means (important pour les distances)
        scaler_clustering = StandardScaler()
        X_scaled_clustering = scaler_clustering.fit_transform(X_clustering)

        # --- Application de la méthode du Score de Silhouette ---

        # Définir une plage de valeurs pour k à tester
        # k doit être au moins 2. Le maximum raisonnable est (nombre d'échantillons - 1).
        # Pour des milliers de codes postaux, tester jusqu'à 20 ou 30 clusters est souvent suffisant pour voir une tendance.
        range_n_clusters = list(range(2, 15)) # Tester de k=2 à k=20

        silhouette_scores = [] # Liste pour stocker les scores de silhouette

        print("Calcul des scores de silhouette pour différentes valeurs de k...")

        for n_clusters in range_n_clusters:
            # Appliquer K-Means pour le nombre de clusters actuel
            # n_init=10 est recommandé pour une initialisation plus robuste
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled_clustering)

            # Calculer le score de silhouette moyen
            # Nécessite au moins 2 échantillons dans les données
            if len(np.unique(cluster_labels)) > 1: # S'assurer qu'il y a plus d'un cluster formé
                score = silhouette_score(X_scaled_clustering, cluster_labels)
                silhouette_scores.append(score)
                print(f"  k={n_clusters}, Score de silhouette = {score:.2f}")
            else:
                # Ce cas est rare si n_clusters > 1 et qu'il y a assez de données
                silhouette_scores.append(0) # Ou NaN, selon comment vous voulez gérer


        # --- Visualisation des résultats ---

        plt.figure(figsize=(10, 6))
        plt.plot(range_n_clusters, silhouette_scores, marker='o')
        plt.title("Score de Silhouette en fonction du nombre de clusters (k)")
        plt.xlabel("Nombre de clusters (k)")
        plt.ylabel("Score de Silhouette Moyen")
        plt.xticks(range_n_clusters)
        plt.grid(True)
        plt.show()

        # --- Interprétation ---
        # La valeur de k qui correspond au point le plus élevé sur ce graphique
        # est considérée comme le nombre optimal de clusters selon la méthode de la silhouette.
        # Regardez le graphique pour identifier le pic.
        optimal_k_silhouette = range_n_clusters[np.argmax(silhouette_scores)]
        print(f"\nSelon la méthode de la silhouette, le nombre optimal de clusters (k) est : {optimal_k_silhouette}")

    def determine_kmean_elbow_method(self,df):
        """
        Determines the optimal number of clusters (k) for K-Means clustering
        using the Elbow Method on the 'avg_price_per_sqm' feature.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing property data.
                               It is expected to contain at least 'postCode', 'price', and 'habitableSurface'
                               to be able to calculate 'avg_price_per_sqm'.

        Returns:
            None: This function primarily displays a plot and prints results
                  to help choose k. It does not return a modified DataFrame.
        """

        print("--- Starting determination of k using the Elbow Method ---")

        # Ensure the 'avg_price_per_sqm' column exists by calling the previous function.
        # This function assumes 'avg_price_per_sqm' is the feature to cluster on.
        # It's important that calculate_price_pr_sqm handles missing values and invalid surfaces correctly.
        df = self.calculate_price_pr_sqm(df)

        # Filter out rows where 'avg_price_per_sqm' is NaN, as these cannot be used for clustering.
        # Clustering algorithms (like K-Means) cannot handle missing values.
        df_clustering_data = df.dropna(subset=['avg_price_per_sqm']).copy()

        # Check if there's enough data left for clustering after dropping NaNs.
        if len(df_clustering_data) < 2:
             print("Error: Not enough non-missing data for clustering after calculating average price/sqm.")
             print("--- Ending determination of k ---")
             return # Exit the function if not enough data

        # Prepare the data for clustering.
        # Select the feature to be used for clustering ('avg_price_pr_sqm').
        # Use .values to get a 2D NumPy array, which is the expected input format for scikit-learn models.
        X_clustering = df_clustering_data[['avg_price_per_sqm']].values

        # Scale the data for K-Means.
        # Scaling is crucial for K-Means because it is a distance-based algorithm.
        # Features with larger scales would disproportionately influence the distance calculations without scaling.
        scaler_clustering = StandardScaler()
        X_scaled_clustering = scaler_clustering.fit_transform(X_clustering)


        # --- Applying the Elbow Method ---

        # Define a range of values for k (number of clusters) to test.
        # k must be at least 1. The reasonable maximum is generally well below the number of samples.
        # Testing up to a certain number (e.g., 20 or 30) is often sufficient to see the "elbow".
        # The range from 2 to 20 (21 is exclusive) is a good starting point.
        range_n_clusters = list(range(2, 21)) # Test from k=2 up to k=20

        inertia = [] # List to store the inertia values for each k

        print(f"Calculating inertia values for different k values in the range {range_n_clusters} (Elbow Method)...")

        # Loop through each potential number of clusters (k).
        for n_clusters in range_n_clusters:
            # Apply K-Means for the current number of clusters.
            # random_state ensures reproducibility of the K-Means initialization.
            # n_init=10 is recommended to run the algorithm 10 times with different centroid seeds
            # and choose the best result (in terms of inertia).
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_scaled_clustering) # Train the K-Means model on the scaled data.

            # Store the inertia for this K-Means model.
            # Inertia is the sum of squared distances of samples to their closest cluster center.
            # It measures the internal cohesion of clusters. Lower inertia is better.
            inertia.append(kmeans.inertia_)
            print(f"   k={n_clusters}, Inertia = {kmeans.inertia_:.2f}")


        # --- Visualizing the Results ---
        # Plot the curve of inertia versus the number of clusters.
        # The "elbow" (point where the rate of decrease in inertia slows significantly)
        # suggests the optimal number of clusters.
        plt.figure(figsize=(10, 6)) # Define the figure size
        plt.plot(range_n_clusters, inertia, marker='o') # Plot the line with markers for each k point
        plt.title("Elbow Method") # Plot title
        plt.xlabel("Number of Clusters (k)") # X-axis label
        plt.ylabel("Inertia") # Y-axis label
        plt.xticks(range_n_clusters) # Ensure X-axis ticks match the tested k values
        plt.grid(True) # Display a grid for easier reading
        plt.show() # Display the plot

        print("\nElbow Method plot displayed.")
        print("Inspect the plot to find the point where the curve forms an 'elbow'.")
        print("--- Ending determination of k ---")