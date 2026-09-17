"""Clustering utilities with bounded-cost evaluation and reproducible seeds."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

RANDOM_STATE = 42


def safe_silhouette(X, labels, sample_size=1000, ignore_noise=False):
    """Return None when a silhouette score is undefined; omit DBSCAN noise if requested."""
    labels = np.asarray(labels)
    if ignore_noise:
        valid = labels != -1
        X, labels = X[valid], labels[valid]
    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    if n < 3 or len(unique) < 2 or len(unique) >= n:
        return None
    # Sampling can accidentally omit small clusters; include stratified representatives.
    if n > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        required = np.array([rng.choice(np.flatnonzero(labels == c)) for c in unique])
        pool = np.setdiff1d(np.arange(n), required, assume_unique=True)
        rest = rng.choice(pool, size=min(sample_size - len(required), len(pool)), replace=False)
        idx = np.concatenate([required, rest])
        X, labels = X[idx], labels[idx]
    if len(np.unique(labels)) >= len(labels):
        return None
    return float(silhouette_score(X, labels))


def find_optimal_k(X_scaled, max_k=10):
    n = len(X_scaled)
    if n < 3:
        raise ValueError('At least three rows are needed to search for clusters.')
    k_range = list(range(2, min(int(max_k), n - 1) + 1))
    if not k_range:
        raise ValueError('max_k must be at least 2.')
    inertia, scores = [], []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        inertia.append(float(model.inertia_))
        scores.append(safe_silhouette(X_scaled, labels))
    return k_range, inertia, scores


def fit_kmeans(X_scaled, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels, safe_silhouette(X_scaled, labels)


def fit_hierarchical(X_scaled, n_clusters=4):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X_scaled)
    return model, labels, safe_silhouette(X_scaled, labels)


def fit_dbscan(X_scaled, eps=1.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    return model, labels, safe_silhouette(X_scaled, labels, ignore_noise=True)


def apply_pca(X_scaled, n_components=2):
    model = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return model, model.fit_transform(X_scaled)


def profile_clusters(df, cluster_col='cluster'):
    summary = df.groupby(cluster_col)[['release_year', 'year_added', 'duration_value']].mean()
    type_dist = pd.crosstab(df[cluster_col], df['type'], normalize='index')
    profiles = {}
    for c, sub_df in df.groupby(cluster_col):
        profiles[c] = {
            'count': len(sub_df),
            'top_genres': sub_df['main_genre'].value_counts().head(3).to_dict(),
            'top_ratings': sub_df['rating'].value_counts().head(3).to_dict(),
            'top_countries': sub_df['country_group'].value_counts().head(3).to_dict(),
        }
    return summary, type_dist, profiles
