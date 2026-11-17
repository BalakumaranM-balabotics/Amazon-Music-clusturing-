
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# --------------------------------
# 1. Load saved model and scaler
# --------------------------------

@st.cache_resource
def load_model():
    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    return scaler, kmeans


scaler, kmeans = load_model()


# --------------------------------
# 2. Load dataset
# --------------------------------

@st.cache_data
def load_data():

    possible_files = [
        "final_clustered_songs.csv",
        "amazon_music_clustered_new.csv",
        "single_genre_artists.csv"
    ]

    for file in possible_files:
        if os.path.exists(file):
            return pd.read_csv(file)

    st.error("Dataset CSV file not found.")
    st.stop()


df = load_data()


# --------------------------------
# 3. Select audio features
# --------------------------------

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


# Check required columns

if not all(feature in df.columns for feature in features):
    st.error("The dataset is missing one or more required audio features.")
    st.stop()


# --------------------------------
# 4. Streamlit application
# --------------------------------

st.title("Amazon Music Song Cluster Predictor")

st.write(
    "This application uses the trained K-Means model "
    "to predict the cluster of a song based on its audio features."
)

st.subheader("Enter Song Audio Features")


# --------------------------------
# 5. User input
# --------------------------------

inputs = {}

for feature in features:

    minimum = float(df[feature].min())
    maximum = float(df[feature].max())
    default = float(df[feature].median())

    # Avoid identical slider boundaries
    if minimum == maximum:
        maximum = minimum + 1.0

    if feature == "duration_ms":
        step = 1000.0
    elif feature == "tempo":
        step = 0.1
    elif feature == "loudness":
        step = 0.1
    else:
        step = 0.01

    inputs[feature] = st.slider(
        feature.capitalize(),
        min_value=minimum,
        max_value=maximum,
        value=min(max(default, minimum), maximum),
        step=step
    )


# --------------------------------
# 6. Predict cluster
# --------------------------------

if st.button("Predict Song Cluster"):

    input_df = pd.DataFrame([inputs], columns=features)

    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_df)

    # Predict cluster using saved K-Means model
    cluster = int(kmeans.predict(input_scaled)[0])

    st.subheader("Prediction Result")

    st.success(f"Predicted Cluster: {cluster}")

    # Cluster interpretations from our notebook
    cluster_names = {
        0: "Chill and Acoustic",
        1: "Party and Energetic",
        2: "Speech-heavy, Rap and Spoken-word"
    }

    st.write(f"**Song Category:** {cluster_names[cluster]}")


# --------------------------------
# 7. PCA Visualization
# --------------------------------

st.subheader("K-Means Cluster Visualization")

# Scale original dataset
X_scaled = scaler.transform(df[features])

# Apply PCA using three components, as in our notebook
pca = PCA(n_components=3)

X_pca = pca.fit_transform(X_scaled)

# Get cluster labels from the saved K-Means model
predicted_labels = kmeans.predict(X_scaled)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot existing songs
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=predicted_labels,
    cmap="viridis",
    alpha=0.15,
    s=10
)

ax.set_title("Amazon Music Songs - K-Means Clusters (PCA)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

plt.colorbar(scatter, ax=ax, label="Cluster")

st.pyplot(fig)


# --------------------------------
# 8. Explained variance
# --------------------------------

st.write(
    "**PCA Explained Variance (3 Components):**",
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

st.write("Amazon Music clusturing project- BalakumaranM")
