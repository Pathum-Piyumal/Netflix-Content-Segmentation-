# Netflix Content Intelligence — Content Segmentation

A reproducible Netflix title segmentation project with a Streamlit dashboard, K-Means, agglomerative clustering, DBSCAN, PCA projections, and static chart generation. Includes the supplied dataset (8,790 titles) and notebook.

## Quick start

Use Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard is available at the local address printed by Streamlit (usually `http://localhost:8501`). Launch it from any directory using the full path to `app.py` if needed; data paths are resolved relative to the project.

Generate six static charts and run all three clustering methods:

```bash
python run_pipeline.py
```

Run automated regression tests (requires `pytest`, a development dependency):

```bash
python -m pip install pytest
python -m pytest -q
```

## Project contents

- `app.py` — interactive, multi-page Streamlit dashboard
- `src/preprocessing.py` — cleaning, metadata extraction, categorical encoding, standard scaling
- `src/clustering.py` — reproducible algorithms, safe sampled silhouette metrics, cluster profiles
- `src/visualization.py` — static analysis figures (under `images/`)
- `run_pipeline.py` — end-to-end model + visualization execution
- `notebooks/netflix_clustering.ipynb` — original exploratory notebook (not synchronized with every new code change)
- `tests/test_project.py` — preprocessing, clustering, edge-case regression checks
- `QA_REPORT.md` — changes, verification and remaining limitations

## Interpreting results

The full-data pipeline currently selects K=5 based on its sampled silhouette evaluation (fixed seed 42). The cluster labels are **unsupervised descriptive groups**, not verified editorial genre classes or recommendation personalization predictions. Silhouette scores depend on encoding, scaling, sample choice, and parameter settings. DBSCAN's silhouette score excludes noise points, so it is **not directly comparable** to all-title K-Means or hierarchical scores.

**Important:** `duration_value` is measured in minutes for movies but seasons for TV shows; mean duration for mixed-type clusters is not a comparable quantity. Ratings are metadata categories, not user star ratings. This is historical catalog data, not a live Netflix feed.

K-Means assigns each title to a segment, while DBSCAN may label observations `-1` (noise). Long-running runs can be resource-intensive: hierarchical clustering and DBSCAN use substantially more memory than K-Means for large datasets.

## Limitations

The dashboard and CLI share the same core `src/` processing and algorithms, but the archival notebook is preserved independently. The dashboard needs Streamlit and a browser to execute. The included test suite validates core functionality; it is not a substitute for browser-based UI acceptance testing.

The dataset and project source may have separate upstream licensing requirements. Verify source rights before public or commercial distribution.
