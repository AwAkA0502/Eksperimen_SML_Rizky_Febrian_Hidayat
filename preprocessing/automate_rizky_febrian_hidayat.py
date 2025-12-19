import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# --- FUNGSI-FUNGSI PREPROCESSING (SAMA SEPERTI KODE ANDA) ---

def load_data(file_path):
    print("="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil di-load dari: {file_path}")
        print(f"   Total baris: {len(df)} | Total kolom: {df.shape[1]}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return None

def remove_duplicates(df):
    print("\n" + "="*70)
    print("STEP 2: MENGHAPUS DUPLIKASI")
    print("="*70)
    duplicates = df.duplicated().sum()
    df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
    print(f"   {duplicates} duplikasi berhasil dihapus")
    return df_clean

def impute_engine_size(df):
    print("\n" + "="*70)
    print("STEP 3: IMPUTASI ENGINE SIZE = 0")
    print("="*70)
    zero_count = (df['engineSize'] == 0).sum()
    if zero_count > 0:
        for model in df[df['engineSize'] == 0]['model'].unique():
            median_engine = df[(df['model'] == model) & (df['engineSize'] > 0)]['engineSize'].median()
            df.loc[(df['model'] == model) & (df['engineSize'] == 0), 'engineSize'] = median_engine
        print(f" Imputasi selesai untuk {zero_count} baris")
    else:
        print(" Tidak ada engineSize = 0, skip imputasi")
    return df

def remove_outliers(df, columns):
    print("\n" + "="*70)
    print("STEP 4: MENGHAPUS OUTLIERS (IQR METHOD)")
    print("="*70)
    df_clean = df.copy()
    for col in columns:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    print(f" Outliers berhasil dibersihkan. Sisa data: {len(df_clean)} baris")
    return df_clean.reset_index(drop=True)

def encode_categorical(df, columns):
    print("\n" + "="*70)
    print("STEP 5: ENCODING DATA KATEGORIKAL")
    print("="*70)
    for col in columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        print(f"   {col} encoded.")
    return df

def scale_features(df, numerical_columns):
    print("\n" + "="*70)
    print("STEP 6: FEATURE SCALING")
    print("="*70)
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print(f" Scaling selesai untuk: {numerical_columns}")
    return df

def save_data(df, output_path):
    print("\n" + "="*70)
    print("STEP 7: MENYIMPAN DATA BERSIH")
    print("="*70)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Data berhasil disimpan ke: {output_path}")

# --- PIPELINE UTAMA ---

def preprocess_pipeline(input_path, output_path):
    print("\n" + "="*70)
    print("AUTOMATE PREPROCESSING - BMW CAR PRICE")
    print("="*70)
    
    df = load_data(input_path)
    if df is None: return
    
    df = remove_duplicates(df)
    df = impute_engine_size(df)
    df = remove_outliers(df, ['price', 'mileage', 'mpg', 'tax'])
    df = encode_categorical(df, ['model', 'transmission', 'fuelType'])
    df = scale_features(df, ['year', 'mileage', 'tax', 'mpg', 'engineSize'])
    
    save_data(df, output_path)
    print("\nPIPELINE SELESAI!")

if __name__ == "__main__":
    # DISESUAIKAN UNTUK GITHUB ACTIONS
    # bmw.csv harus berada di root repository
    # output akan disimpan di folder preprocessing/namadataset_preprocessing/
    INPUT = 'bmw.csv' 
    OUTPUT = 'preprocessing/namadataset_preprocessing/bmw_preprocessing.csv'
    
    preprocess_pipeline(INPUT, OUTPUT)