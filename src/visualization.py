import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Set styling
sns.set_theme(style="darkgrid")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

def plot_type_distribution(df, save_path="images/type_distribution.png"):
    """Plot distribution of Movies vs TV Shows."""
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x="type", hue="type", palette="Set2", legend=False)
    plt.title("Netflix Content Distribution: Movies vs TV Shows", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Content Type", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_top_genres(df, top_n=10, save_path="images/top_genres.png"):
    """Plot top N Netflix main genres."""
    plt.figure(figsize=(10, 6))
    top_genres = df["main_genre"].value_counts().head(top_n)
    sns.barplot(x=top_genres.values, y=top_genres.index, hue=top_genres.index, palette="viridis", legend=False)
    plt.title(f"Top {top_n} Main Genres on Netflix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Titles", fontsize=12)
    plt.ylabel("Main Genre", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_elbow_curve(k_range, inertia, save_path="images/elbow_curve.png"):
    """Plot Elbow method curve for K-Means."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    plt.title("Elbow Method for Optimal K Selection", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_silhouette_scores(k_range, scores, save_path="images/silhouette_scores.png"):
    """Plot Silhouette scores across K values."""
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
    plt.title("Silhouette Score vs. Number of Clusters", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Clusters (K)", fontsize=12)
    plt.ylabel("Silhouette Score", fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pca_clusters(df, cluster_col="cluster", save_path="images/cluster_visualization.png"):
    """Plot 2D PCA cluster visualization."""
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue=cluster_col,
        palette="tab10",
        style=cluster_col,
        alpha=0.7,
        s=60
    )
    plt.title("Netflix Content Clusters (2D PCA Projection)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Principal Component 1 (PCA1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PCA2)", fontsize=12)
    plt.legend(title="Cluster Segment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cluster_profiling(df, cluster_col="cluster", save_path="images/cluster_profiling.png"):
    """Plot bar chart comparing duration and release year per cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average Duration per Cluster
    sns.barplot(data=df, x=cluster_col, y="duration_value", ax=axes[0], hue=cluster_col, palette="Blues_d", legend=False)
    axes[0].set_title("Average Duration / Seasons per Cluster", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Duration Value")
    
    # Average Release Year per Cluster
    sns.barplot(data=df, x=cluster_col, y="release_year", ax=axes[1], hue=cluster_col, palette="Greens_d", legend=False)
    axes[1].set_title("Average Release Year per Cluster", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Release Year")
    axes[1].set_ylim(df["release_year"].min() - 5, df["release_year"].max() + 1)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
