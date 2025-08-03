import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

def feature_engineering(X):
    X = X.copy()
    X['rooms_per_household'] = X['total_rooms'] / X['households']
    X['bedrooms_per_room'] = X['total_bedrooms'] / X['total_rooms']
    X['population_per_household'] = X['population'] / X['households']
    X['location'] = X['longitude'] + X['latitude']
    return X

def preprocess(X, preprocessing_pipeline, use_streamlit=False):
    print("--- preprocess.py: preprocess function called ---")
    try:
        # Apply feature engineering manually in case not part of pipeline
        X = feature_engineering(X)

        num_features = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]
        cat_features = [col for col in X.columns if X[col].dtype == 'object']
        
        X_prepared = preprocessing_pipeline.transform(X)
        print("Transformed shape:", X_prepared.shape)

        # Extract column names
        num_col_names = num_features
        cat_encoder = preprocessing_pipeline.named_transformers_['cat'].named_steps['encoder']
        cat_col_names = cat_encoder.get_feature_names_out(cat_features)
        all_col_names = list(num_col_names) + list(cat_col_names)

        X_dense = X_prepared.toarray() if hasattr(X_prepared, "toarray") else X_prepared
        return pd.DataFrame(X_dense, columns=all_col_names)

    except Exception as e:
        if use_streamlit:
            import streamlit as st
            st.error(f"Preprocessing error: {e}")
        else:
            print("Preprocessing error:", e)
        return None
