# Amazon Music Clustering

An **unsupervised machine learning project** that groups songs based on
their audio characteristics. The project applies K-Means as the primary
clustering algorithm and explores DBSCAN as an alternative method. The
resulting clusters are profiled and visualized to understand their
musical characteristics and demonstrate a possible application in
playlist generation.

------------------------------------------------------------------------

## 📋 Table of Contents

-   [Problem Statement]
-   [Business use cases]
-   [Features]
-   [Project Objectives]
-   [Dataset]
-   [Tech Stack]
-   [Project Structure]
-   [Methodology]
-   [Clustering Results]
-   [Visualizations]
-   [DBSCAN Analysis]
-   [Genre Analysis]
-   [Playlist Generation Use Case]
-   [Conclusion]

------------------------------------------------------------------------

## 🎯 Problem Statement

With millions of songs available on platforms like Amazon Music, manually categorizing tracks into genres is impractical. This project automatically groups similar songs based on their audio characteristics using clustering techniques, organizing songs into meaningful clusters representing different musical genres or moods—**without any prior labels**.

------------------------------------------------------------------------

## 💼 Business Use Cases

- **Personalized Playlist Curation**: Automatically group songs that sound similar to enhance playlist generation
- **Improved Song Discovery**: Suggest similar tracks to users based on their preferred audio profile
- **Artist Analysis**: Help artists and producers identify competitive songs in the same audio cluster
- **Market Segmentation**: Streaming platforms can analyze user listening patterns and optimize recommendations

---
## ✨ Features

- **Data Exploration & Preprocessing**: Comprehensive EDA, handling missing values, and feature scaling
- **Multiple Clustering Algorithms**: K-Means and DBSCAN implementation
- **Optimal Cluster Selection**: Elbow method and Silhouette score analysis
- **Dimensionality Reduction**: PCA visualization for 2D cluster representation
- **Model Persistence**: Trained models saved for production use
- **Interactive Web App**: Streamlit-based predictor for real-time cluster prediction
- **Comprehensive Analysis**: Cluster profiling and genre inference

------------------------------------------------------------------------

## 🎯 Project Objectives

-   Explore and understand the music dataset.
-   Select relevant audio features for clustering.
-   Scale the selected features before distance-based clustering.
-   Use the Elbow Method, Silhouette Score and Davies-Bouldin Index to
    evaluate different K-Means cluster counts.
-   Apply K-Means to group songs with similar audio characteristics.
-   Use PCA and other plots to visualize and interpret the clusters.
-   Explore DBSCAN as an alternative clustering algorithm.
-   Discuss how the resulting clusters can support playlist generation
    or song discovery.

------------------------------------------------------------------------

## 📂 Dataset

The project uses `single_genre_artists.csv`.

The dataset contains **95,837 rows and 23 columns**.

The 10 audio features selected for clustering are:

  Feature              Description
  -------------------- ---------------------------------------------------
  `danceability`       How suitable a track is for dancing
  `energy`             Perceptual measure of intensity and activity
  `loudness`           Overall loudness of a track, measured in decibels
  `speechiness`        Presence of spoken words in a track
  `acousticness`       Confidence measure of whether a track is acoustic
  `instrumentalness`   Likelihood that a track contains no vocals
  `liveness`           Likelihood that a track was performed live
  `valence`            Musical positiveness conveyed by a track
  `tempo`              Estimated tempo in beats per minute (BPM)
  `duration_ms`        Track duration in milliseconds

The `genres` column is retained for genre analysis and interpretation.
It is **not used as an input feature** for K-Means or DBSCAN.

------------------------------------------------------------------------

## 🛠️ Tech Stack

**Languages & Libraries:**
- Python 3.8+
- Pandas, NumPy - Data manipulation
- Scikit-learn - Machine learning algorithms
- Matplotlib - Data visualization
- Joblib - Model serialization
- Streamlit - Web application framework

**Algorithms:**
- K-Means Clustering
- DBSCAN (Density-Based Spatial Clustering)
- Principal Component Analysis (PCA)
- StandardScaler for feature normalization

------------------------------------------------------------------------

## 📁 Project Structure

``` text
Amazon-Music-Clustering/
│
├── amc_datahandle.ipynb
├── single_genre_artists.csv
├── amazon_music_clustered_new1.csv
└── Amazon_music_clusturing_README.md
|___amazon_streamlit.py
```

-   `amc_datahandle.ipynb` -- notebook containing data exploration,
    preprocessing, clustering, evaluation and visualizations.
-   `single_genre_artists.csv` -- input dataset.
-   `amazon_music_clustered_new.csv` -- exported dataset containing the
    clustering result columns.
-   `Amazon_music_clusturing_README` -- project documentation.
-   'amzon_streamlit.py'- project representing website

------------------------------------------------------------------------

## 🔬 Methodology

### 1. Data Exploration and Preprocessing

The dataset was loaded into a Pandas DataFrame and inspected.

The following checks and operations were performed:

-   Examined the first few rows, column names, data types and dataset
    shape.
-   Checked for missing values.
-   Checked for duplicate rows. The dataset contained **0 exact
    duplicate rows**.
-   Excluded song and artist identifiers/names from the clustering
    feature input.
-   Converted the `release_date` values into year values.
-   Plotted histograms for the selected audio features to understand
    their distributions, spread and skewness.

### 2. Feature Selection

The following 10 audio features were selected:

``` python
features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

X = amc_df[features]
```

### 3. Feature Scaling

`StandardScaler` was used because the selected features have different
numerical scales and K-Means and DBSCAN use distance calculations.

``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

The scaled features have approximately a mean of 0 and a standard
deviation of 1.

### 4. Choosing K for K-Means

Different values of K were evaluated before selecting the final K-Means
model.

**Elbow Method**

-   Tested K values from 1 to 10.
-   Used WCSS (Within-Cluster Sum of Squares), also known as K-Means
    inertia.
-   Plotted K against WCSS to identify the elbow region.
-   The elbow was observed around **K = 3**.

**Silhouette Score**

The Silhouette Score was evaluated for K values from 2 to 8.

    K   Silhouette Score
  --- ------------------
    2              0.203
    3          **0.242**
    4              0.231
    5              0.186
    6              0.186
    7              0.165
    8              0.172

The highest Silhouette Score among the tested values was obtained at **K
= 3**.

**Davies-Bouldin Index**

    K   Davies-Bouldin Index
  --- ----------------------
    2                 1.9355
    3                 1.5699
    4             **1.5297**
    5                 1.6886
    6                 1.6309
    7                 1.6231
    8                 1.5973

A lower Davies-Bouldin Index is preferred. The lowest value in the
tested range occurred at **K = 4**.

The Elbow Method and Silhouette Score supported **K = 3**, while the
Davies-Bouldin Index reached its minimum at K = 4. Considering the
results together, **K = 3 was selected for the final K-Means model**.

### 5. K-Means Clustering

The final K-Means model was created with three clusters:

``` python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
amc_df["cluster"] = kmeans.fit_predict(X_scaled)
```

Each song receives a cluster label based on its audio features.

### 6. Cluster Profiling

The mean of each audio feature was calculated for every cluster:

``` python
cluster_means = amc_df.groupby("cluster")[features].mean()
```

The feature means were compared across clusters to identify their
distinctive characteristics. These patterns were then used to give the
clusters descriptive interpretations.

------------------------------------------------------------------------

## 📊 Clustering Results

The final K-Means model produced **3 clusters**.

### Cluster Distribution (K-Means with k=3)

| Cluster | Description | Key Characteristics |
|---------|-------------|---------------------|
| **Cluster 0** | Chill Acoustic | Low energy, high acousticness, calm vibe |
| **Cluster 1** | Party Tracks | High danceability, high energy, high tempo, high valence |
| **Cluster 2** | Rap/Live Recordings | High danceability, medium energy, high speechiness, high liveness |
------------------------------------------------------------------------

## 📈 Visualizations

The project includes the following visualizations:

### PCA Cluster Scatter Plot

PCA was applied to reduce the original 10-dimensional feature space to 3
principal components.

``` python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
```

The scatter plot displays PC1 against PC2, with points colored according
to their K-Means cluster labels.

The three selected principal components together explain approximately
**57.45% of the variance** in the scaled features.

### Average Feature Values Bar Chart

A bar chart compares the average audio-feature values across the
clusters. `duration_ms` was excluded from this particular raw-value
chart because its numerical scale makes the other bars difficult to
compare. It remains included in the clustering features.

### Feature Comparison Heatmap

A heatmap displays the average feature values for each cluster using
color intensity and annotated values.

### Feature Distribution Plots

Histograms with density curves compare the distributions of features
such as danceability, energy, loudness, speechiness, acousticness,
instrumentalness, liveness, valence and tempo across the clusters.

------------------------------------------------------------------------

## 🔎 DBSCAN Analysis

DBSCAN was explored as an alternative clustering algorithm.

``` python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
```

Parameters used:

-   `eps = 1`
-   `min_samples = 5`

**Observed result:**

-   Number of clusters: **77**
-   Number of noise points: **9,514**

DBSCAN does not require the number of clusters to be specified in
advance. It identifies dense regions as clusters and labels points that
do not belong to a cluster as noise (`-1`).

The DBSCAN results were also visualized using the PCA representation.

------------------------------------------------------------------------

## 🎵 Genre Analysis

The `genres` column was explored separately to understand the genre
information available in the dataset.

-   Number of unique genre entries: **3,153**
-   Some records have an empty genre entry (`[]`).
-   Empty genre entries were retained because genre was not used as a
    clustering input feature.

The most frequent entries included:

  Genre entry                   Count
  --------------------------- -------
  `['hoerspiel']`               8,027
  `['kleine hoerspiel']`        2,081
  `[]` (empty genre)            1,876
  `['classic israeli pop']`     1,180
  `['vintage taiwan pop']`      1,097

Genre information can be compared with the discovered clusters to help
interpret the groups. The clustering itself is based on audio features,
not these genre entries.

------------------------------------------------------------------------

## 🎧 Playlist Generation Use Case

The clusters can support a simple **cluster-based playlist generation**
concept.

Songs assigned to the same cluster share similar patterns across the
selected audio features. A music platform could group songs from a
chosen cluster into a playlist.

Example:

``` text
Cluster 0 → Chill / Acoustic playlist
Cluster 1 → Party / Energetic playlist
Cluster 2 → Speech-heavy / Rap-oriented playlist
```

This project demonstrates how clustering can support playlist curation
and song discovery. It does not implement a complete, user-personalized
production recommendation system.

------------------------------------------------------------------------

## 🧾 Conclusion

This project applies unsupervised machine learning to a dataset of
**95,837 songs** to discover groups based on audio characteristics.

K-Means was selected as the primary clustering method with **K = 3**,
supported by the Elbow Method and Silhouette Score. Cluster profiling
and visualizations helped interpret the groups as chill/acoustic,
party/energetic and speech-heavy/rap-oriented characteristics.

DBSCAN was also explored as an alternative and identified 77 clusters
and 9,514 noise points with the selected parameters.

Overall, the project demonstrates how audio-feature clustering can help
organize music and provide a foundation for playlist generation and song
discovery.
