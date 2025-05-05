import pandas as pd
import pgeocode

class DataManager:

    @staticmethod
    def load_csv(path_to_csv,debug=False) -> pd.DataFrame:
       if debug:
           print(f"Load csv file : {path_to_csv}")
       return pd.read_csv(path_to_csv)
    
    @staticmethod
    def get_lat_lng_for_zipcode(df,debug=False):

        if debug:
            print(f"DataManager::Get lat/Lng for poastal code from Nominatim")
        nomi = pgeocode.Nominatim('BE')
        df["zipcode_Latitude"] = (nomi.query_postal_code(list(map(str,df["postCode"].tolist()))).latitude)
        df["zipcode_Longitude"] = (nomi.query_postal_code(list(map(str,df["postCode"].tolist()))).longitude)
        return df
    
    @staticmethod   
    def merge_columnsFrom(main_df,path_to_csv,id_col,from_id_col,from_columns_to_merge,debug=False):
        """Load columns from other dataset."""
        if debug:
            print(f"DataManager::Merge columns : {from_columns_to_merge}")
        main_df[id_col] = pd.to_numeric(main_df[id_col], errors='coerce').astype('Int64')

        from_df = pd.read_csv(path_to_csv)
        from_df = from_df[[from_id_col]+from_columns_to_merge]
        from_df = from_df.rename(columns={from_id_col: 'from_id'})
        from_df = from_df.drop_duplicates(subset=['from_id'], keep='first')
      
        main_df = main_df.merge(from_df, left_on=id_col, right_on='from_id', how='left')
        main_df = main_df.drop(columns=['from_id'])

        return main_df
