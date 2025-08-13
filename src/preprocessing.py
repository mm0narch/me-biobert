import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_text(text:str) -> str:
    if not isinstance(text,str):
        return ''
    
    text=text.lower()
    text=re.sub(r'[^a-z0-9\s]', '', text)
    text=re.sub(r'\s+', '', text)
    return text

def clean_chapter(df:pd.DataFrame):
    df=df[['Kategori', 'Nama_Penyakit']].copy()
    df['Nama_Penyakit']=df['Nama_Penyakit'].str.lower()
    df['Nama_Penyakit']=df['Nama_Penyakit'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['Nama_Penyakit']=df['Nama_Penyakit'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['Nama_Penyakit']=df['Nama_Penyakit'].str.strip()

    le=LabelEncoder()
    df['Label']=le.fit_transform(df['Kategori'])

    return df, le