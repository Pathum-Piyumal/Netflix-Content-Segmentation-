import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path="data/netflix_titles.csv"):
    """Load the Netflix dataset from CSV."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Clean data by removing duplicates, handling missing values, and parsing date_added."""
    df = df.copy()
    
    # Remove duplicate records
    df = df.drop_duplicates()
    
    # Fill important missing categorical values
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Unknown")
    df["listed_in"] = df["listed_in"].fillna("Unknown")
    
    # Convert date_added to datetime and extract year_added
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    
    # Fill missing year_added values with median
    df["year_added"] = df["year_added"].fillna(df["year_added"].median())
    
    return df

def create_features(df):
    """Extract and encode key features from raw columns."""
    df = df.copy()
    
    # Encode content type (Movie: 0, TV Show: 1)
    df["type_encoded"] = df["type"].map({"Movie": 0, "TV Show": 1})
    
    # Extract numerical duration value (minutes or number of seasons)
    df["duration_value"] = df["duration"].astype(str).str.extract(r"(\d+)").astype(float)
    df["duration_value"] = df["duration_value"].fillna(df["duration_value"].median())
    
    # Extract main genre (first listed genre)
    df["main_genre"] = df["listed_in"].astype(str).str.split(",").str[0].str.strip()
    
    # Extract main country and group less common ones into 'Other'
    df["main_country"] = df["country"].astype(str).str.split(",").str[0].str.strip()
    top_countries = df["main_country"].value_counts().head(10).index
    df["country_group"] = df["main_country"].apply(lambda x: x if x in top_countries else "Other")
    
    return df

def prepare_feature_matrix(df):
    """Build the final feature matrix X (numerical + one-hot encoded categorical) and return scaled X."""
    # Numerical features
    numerical_features = ["release_year", "year_added", "duration_value", "type_encoded"]
    X_num = df[numerical_features].copy()
    X_num = X_num.fillna(X_num.median())
    
    # Categorical features with One-Hot Encoding
    categorical_features = ["rating", "main_genre", "country_group"]
    X_cat = pd.get_dummies(df[categorical_features], drop_first=False)
    
    # Combine numerical and categorical features
    X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, scaler
