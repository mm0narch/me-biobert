import pandas as pd
import os

def load_data(file_path:str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    return pd.read_csv(file_path)

def save_file(df:pd.DataFrame,file_path:str) -> None:

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f'Data saved to {file_path}')