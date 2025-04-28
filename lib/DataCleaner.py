import pandas as pandas
import pgeocode as pgeocode


class DataCleaner:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pandas.read_csv(dataset_path)

    def drop_useless_columns(self, columns_to_drop):
        """Drop specified columns from the dataset."""
        self.df = self.df.drop(columns=columns_to_drop)

    def remove_duplicates(self):
        """Remove duplicate entries from the data."""
        self.df.duplicated(subset=["id"]).sum()
        dups = self.df.pivot_table(index = ['id'], aggfunc ='size')
        print(dups[dups > 1]) 
        self.df = self.df.drop_duplicates(subset=["id"], keep="first")

    def fill_missing_values(self):
        # bedroomCount : lets assume that they have at least one so fill nan by 1
        self.df['bedroomCount'] = self.df['bedroomCount'].fillna(1).astype(float)

        # bathroomCount same as bedrooms
        self.df['bathroomCount'] = self.df['bathroomCount'].fillna(1).astype(float)

        # toiletCount same as bedrooms
        self.df['toiletCount'] = self.df['toiletCount'].fillna(1).astype(float)

        # habitableSurface : replace by median 
        self.df['habitableSurface'] = self.df['habitableSurface'].fillna(self.df['habitableSurface'].median())

        # buildingCondition : replace by 'NOT_MENTIONED
        self.df['buildingCondition'] = self.df['buildingCondition'].fillna('NOT_MENTIONED')

        # buildingConstructionYear
        self.df['buildingConstructionYear'] = self.df['buildingConstructionYear'].fillna(self.df['buildingConstructionYear'].median()).astype(int)


        # floodZoneType lts assume that missing values are NON_FLOOD_ZONE
        self.df['floodZoneType'] = self.df['floodZoneType'].fillna('NON_FLOOD_ZONE')

        # heatingType
        self.df['heatingType'] = self.df['heatingType'].fillna(self.df['heatingType'].mode()[0])

        # hasThermicPanels lets assume that if its not precised, there are not
        self.df['hasThermicPanels'] = self.df['hasThermicPanels'].fillna(0).astype(float)

        # kitchenType
        self.df['kitchenType'] = self.df['kitchenType'].fillna(self.df['kitchenType'].mode()[0])

        # landSurface
        self.df['landSurface'] = self.df['landSurface'].fillna(self.df['landSurface'].median())

        # livingRoomSurface
        self.df['livingRoomSurface'] = self.df['livingRoomSurface'].fillna(self.df['livingRoomSurface'].median())

        # terraceSurface
        median_terrace = self.df.loc[(self.df['hasTerrace'] == 1) & (self.df['terraceSurface'].notnull()), 'terraceSurface'].median()
        self.df.loc[(self.df['hasTerrace'] == 1) & (self.df['terraceSurface'].isna()), 'terraceSurface'] = median_terrace
        self.df.loc[(self.df['hasTerrace'] != 1) & (self.df['terraceSurface'].isna()), 'terraceSurface'] = 0

        # terraceOrientation
        mode_terrace = self.df.loc[(self.df['hasTerrace'] == 1), 'terraceOrientation'].mode()[0]
        self.df.loc[(self.df['hasTerrace'] == 1) & (self.df['terraceOrientation'].isna()), 'terraceOrientation'] = mode_terrace
        self.df.loc[(self.df['hasTerrace'] != 1) & (self.df['terraceOrientation'].isna()), 'terraceOrientation'] = 'NO_TERRACE'

        # epcScore
        epc_order = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.df = self.df[self.df['epcScore'].isin(epc_order)]
        self.df['epcScore'] = self.df['epcScore'].fillna(self.df['epcScore'].mode()[0])

    def convert_boolean_to_int(self):
        """Convert boolean values to integers."""
        binary_cols = [
        'hasBasement', 'hasLift', 'hasHeatPump', 'hasPhotovoltaicPanels', 
        'hasAirConditioning', 'hasArmoredDoor', 'hasVisiophone', 'hasOffice', 
        'hasSwimmingPool', 'hasFireplace', 'parkingCountIndoor', 'parkingCountOutdoor'
        ]

        for col in binary_cols:
            self.df[col] = self.df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

        # Colonnes dépendantes d'autres colonnes
        self.df['hasLivingRoom'] = self.df['hasLivingRoom'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        self.df.loc[self.df['hasLivingRoom'].isna(), 'hasLivingRoom'] = self.df['livingRoomSurface'].notnull().astype(int)

        self.df['hasGarden'] = self.df['hasGarden'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        self.df.loc[self.df['hasGarden'].isna(), 'hasGarden'] = self.df['gardenSurface'].notnull().astype(int)

        self.df['hasTerrace'] = self.df['hasTerrace'].map({True: 1, False: 0, 'True': 1, 'False': 0})
        self.df.loc[self.df['hasTerrace'].isna(), 'hasTerrace'] = self.df['terraceSurface'].notnull().astype(int)

        self.df['hasGarden'] = self.df['hasGarden'].map({True: 1, False: 0, 'True': 1, 'False': 0})

        # When hasLivingRoom = 0 ; livingRoomSurface = 0
        self.df.loc[self.df['hasLivingRoom'] == 0, 'livingRoomSurface'] = 0

        # When hasGarden = 0 ; gardenSurface = 0
        self.df.loc[self.df['hasGarden'] == 0, 'gardenSurface'] = 0

        # When hasTerrace = 0 ; terraceSurface = 0 and terraceOrientation = 0
        self.df.loc[self.df['hasTerrace'] == 0, 'terraceSurface'] = 0
        self.df.loc[self.df['hasTerrace'] == 0, 'terraceOrientation'] = 0

        #drop number of facade bigger than 4 and transform "facedeCount" into "facadeCount"
        self.df['facadeCount'] = self.df['facedeCount']
        self.df = self.df.drop(columns='facedeCount')
        self.df['facadeCount'] = self.df['facadeCount'].fillna(2)
        self.df = self.df[self.df['facadeCount'] <= 4]

        print(f"DataCleaner::convert_boolean_to_int -> {self.df.shape}")

    def transform_columns_type(self):
        """Transform the type of specified columns."""
        col_types = {'id': 'int', 'type': 'str', 'subtype': 'str', 'bedroomCount': 'int', 'bathroomCount': 'int',
             'province': 'str', 'locality': 'str', 'postCode': 'int', 'habitableSurface': 'float', 
             'hasBasement': 'int', 'buildingCondition': 'str',
             'buildingConstructionYear': 'int', 'hasLift': 'int', 'floodZoneType': 'str',
             'heatingType': 'str', 'hasHeatPump': 'int', 'hasPhotovoltaicPanels': 'int', 'hasThermicPanels': 'int',
             'kitchenType': 'str', 'landSurface': 'float', 'hasLivingRoom': 'int', 'livingRoomSurface': 'float',
             'hasGarden': 'int', 'gardenSurface': 'float', 'parkingCountIndoor': 'int', 'parkingCountOutdoor': 'int',
             'hasAirConditioning': 'int', 'hasArmoredDoor': 'int', 'hasVisiophone': 'int', 'hasOffice': 'int', 
             'toiletCount': 'int', 'hasSwimmingPool': 'int', 'hasFireplace': 'int', 'hasTerrace': 'int', 'terraceSurface': 'float',
             'terraceOrientation': 'str', 'epcScore': 'str', 'price': 'float', 'facadeCount': 'int'}

        for col, dtype in col_types.items():
            self.df[col] = self.df[col].astype(dtype)

    def transform_label_data(self):
        """Transform the label data to match the main dataframe."""
        self.df["epcScore"] = self.df["epcScore"].map({
            'A++': 0, 'A+': 1, 'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8
        }).astype(int)

    def get_lat_lng_from_postcode(self):
        nomi = pgeocode.Nominatim('BE')
        self.df["Latitude"] = (nomi.query_postal_code(list(map(str,self.df["postCode"].tolist()))).latitude)
        self.df["Longitude"] = (nomi.query_postal_code(list(map(str,self.df["postCode"].tolist()))).longitude)

    def drop_missing_price(self):
        """Drop rows with missing price."""
        self.df = self.df.dropna(subset="price") 

    def dummies(self):
        self.df = pandas.get_dummies(self.df)

    def clean_data(self,columns_to_drop:list) -> pandas.DataFrame:
        """Perform all cleaning operations."""
        print("DataCleaner::clean_data - >Percentage of nan values : ")
        print(f"DataCleaner::clean_data -> \n{(self.df.isna().mean() * 100).round(2).astype(str)} %")
        print(f"DataCleaner::clean_data -> Shape before cleaning : {self.df.shape}")
        print("-" * 50)

        self.get_lat_lng_from_postcode()
        
        self.convert_boolean_to_int()
        self.remove_duplicates()
        self.fill_missing_values()
        self.drop_missing_price()
        self.transform_columns_type()
        self.transform_label_data()
        self.drop_useless_columns(columns_to_drop)
        self.dummies()
        print(f"DataCleaner::clean_data -> Percentage of nan values after cleaning : ")
        print(f"DataCleaner::clean_data -> \n{(self.df.isna().mean() * 100).round(2).astype(str)} %")
        print(f"DataCleaner::clean_data -> Shape after cleaning : {self.df.shape}")
        return self.df
    

