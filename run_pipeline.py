import os
import sys
from pathlib import Path
import pandas as pd

# Add current directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.preprocessing import load_data, clean_data, create_features, prepare_feature_matrix
from src.clustering import find_optimal_k, fit_kmeans, fit_hierarchical, fit_dbscan, apply_pca, profile_clusters
from src.visualization import (
    plot_type_distribution, plot_top_genres, plot_elbow_curve,
    plot_silhouette_scores, plot_pca_clusters, plot_cluster_profiling
)

def main():
    print("Step 1: Loading raw dataset...")
    df = load_data(BASE_DIR / "data" / "netflix_titles.csv")
    print(f"Dataset shape: {df.shape}")
    
    print("\nStep 2: Cleaning data...")
    df_clean = clean_data(df)
    
    print("\nStep 3: Creating engineered features...")
    df_feat = create_features(df_clean)
    
    print("\nStep 4: Preparing feature matrix & scaling...")
    X, X_scaled, scaler = prepare_feature_matrix(df_feat)
    print(f"Feature matrix shape: {X.shape}")
    
    print("\nStep 5: Visualizing EDA plots...")
    plot_type_distribution(df_feat, str(BASE_DIR / "images" / "type_distribution.png"))
    plot_top_genres(df_feat, 10, str(BASE_DIR / "images" / "top_genres.png"))
    print("EDA plots saved in images/")
    
    print("\nStep 6: Finding optimal number of clusters (K)...")
    k_range, inertia, silhouette_scores = find_optimal_k(X_scaled, max_k=10)
    for k, score in zip(k_range, silhouette_scores):
        print(f"K={k}: Silhouette Score = {score if score is not None else 'N/A'}, Inertia = {inertia[k-2]:.2f}")
        
    plot_elbow_curve(k_range, inertia, str(BASE_DIR / "images" / "elbow_curve.png"))
    plot_silhouette_scores(k_range, silhouette_scores, str(BASE_DIR / "images" / "silhouette_scores.png"))
    print("Elbow and Silhouette curves saved in images/")
    
    # Pick optimal K (highest silhouette score or elbow point)
    valid_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) if score is not None]
    best_k = max(valid_scores, key=lambda p: p[1])[0] if valid_scores else k_range[0]
    print(f"\nBest K based on Silhouette Score: K={best_k}")
    
    print(f"\nStep 7: Fitting K-Means with K={best_k}...")
    kmeans_model, kmeans_labels, kmeans_score = fit_kmeans(X_scaled, n_clusters=best_k)
    df_feat["cluster"] = kmeans_labels
    print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
    
    print(f"\nStep 8: Fitting Hierarchical Clustering with K={best_k}...")
    hier_model, hier_labels, hier_score = fit_hierarchical(X_scaled, n_clusters=best_k)
    print(f"Hierarchical Silhouette Score: {hier_score:.4f}")
    
    print("\nStep 9: Fitting DBSCAN experiment...")
    db_model, db_labels, db_score = fit_dbscan(X_scaled, eps=2.0, min_samples=5)
    db_clusters = pd.Series(db_labels).value_counts().to_dict()
    print(f"DBSCAN Cluster Counts (including -1 for noise): {db_clusters}")
    if db_score is not None:
        print(f"DBSCAN Silhouette Score: {db_score:.4f}")
        
    print("\nStep 10: PCA 2D Dimensionality Reduction...")
    pca, X_pca = apply_pca(X_scaled, n_components=2)
    df_feat["PCA1"] = X_pca[:, 0]
    df_feat["PCA2"] = X_pca[:, 1]
    
    plot_pca_clusters(df_feat, "cluster", str(BASE_DIR / "images" / "cluster_visualization.png"))
    plot_cluster_profiling(df_feat, "cluster", str(BASE_DIR / "images" / "cluster_profiling.png"))
    print("PCA Cluster plot and Profiling chart saved in images/")
    
    print("\nStep 11: Cluster Profiling & Summary:")
    summary, type_dist, profiles = profile_clusters(df_feat, "cluster")
    print("\nNumerical Averages per Cluster:")
    print(summary)
    print("\nMovie vs TV Show ratio per Cluster:")
    print(type_dist)
    
    print("\nDetailed Cluster Profiles:")
    for c, info in profiles.items():
        print(f"\n--- Cluster {c} (Count: {info['count']}) ---")
        print(f"Top Genres: {info['top_genres']}")
        print(f"Top Ratings: {info['top_ratings']}")
        print(f"Top Countries: {info['top_countries']}")

if __name__ == "__main__":
    main()
