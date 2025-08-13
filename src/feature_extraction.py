from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

_model=None

def load_model(model_name: str='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb') -> SentenceTransformer: 
    global _model
    if _model is None: 
        _model=SentenceTransformer(model_name)
    return _model

def get_biobert_embeddings(text: List[str], batch_size: int=16, save_path: Optional[str]=None) -> np.ndarray:
    model=load_model()
    embeddings=model.encode(
        text, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embeddings)
    
    return embeddings

def get_tfidf_features(text: List[str], save_path: Optional[str]=None):
    vectorizer=TfidfVectorizer()
    features=vectorizer.fit_transform(text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sparse.save_npz(save_path, features)

    return features, vectorizer