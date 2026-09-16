import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from src.preprocessing import clean_data, create_features, prepare_feature_matrix
from src.clustering import safe_silhouette, find_optimal_k, fit_kmeans, fit_dbscan, apply_pca, profile_clusters


def sample_df():
    return pd.DataFrame({
        'type': ['Movie', 'TV Show', 'Movie', 'TV Show'],
        'country': ['USA', None, 'India', 'USA'],
        'rating': ['PG', None, 'TV-MA', 'PG'],
        'listed_in': ['Drama', 'Comedy', 'Drama', 'Documentaries'],
        'date_added': ['2020-01-01', None, '2022-02-03', 'bad date'],
        'release_year': [2019, 2018, 2020, 2021],
        'duration': ['90 min', '1 Season', None, '2 Seasons'],
    })


def test_preprocessing_complete_and_finite():
    df = create_features(clean_data(sample_df()))
    X, scaled, _ = prepare_feature_matrix(df)
    assert len(df) == len(X) == len(scaled) == 4
    assert np.isfinite(scaled).all()
    assert df['year_added'].notna().all()


def test_safe_silhouette_edge_cases():
    X = np.array([[0.], [1.], [2.], [3.]])
    assert safe_silhouette(X, [-1] * 4, ignore_noise=True) is None
    assert safe_silhouette(X, [0] * 4) is None
    assert safe_silhouette(X, [0, 1, 2, 3]) is None


def test_models_and_profiles():
    X, _ = make_blobs(n_samples=80, centers=3, random_state=42)
    ks, inertia, scores = find_optimal_k(X, max_k=5)
    assert len(ks) == len(inertia) == len(scores) == 4
    model, labels, score = fit_kmeans(X, 3)
    assert len(labels) == len(X) and score is not None
    assert apply_pca(X, 2)[1].shape == (80, 2)
    assert fit_dbscan(X, eps=0.001)[2] is None
    df = create_features(clean_data(sample_df()))
    df['cluster'] = [0, 0, 1, 1]
    summary, types, profiles = profile_clusters(df)
    assert len(summary) == 2 and len(profiles) == 2


def test_too_few_rows():
    with pytest.raises(ValueError):
        find_optimal_k(np.ones((2, 2)))
