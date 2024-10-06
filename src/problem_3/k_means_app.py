from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

from .custom_kmeans import CustomKMeans
from .streamlit_base import StreamlitBase


class KMeansClusteringApp(StreamlitBase):
    """Streamlit application for analyzing k-means clustering."""

    def __init__(self, model: KeyedVectors) -> None:
        """
        Initialize the KMeans Clustering app with a word2vec model.

        Args:
            model (gensim.models.KeyedVectors): The word2vec model to use for clustering.
        """
        super().__init__(model)

    def run(self) -> None:
        """
        Run the KMeans clustering app on a word2vec model.

        This function guides the user through the following steps:

        1. Get input from the user.
        2. Wait for the user to confirm input before processing.
        3. Preprocess and filter the words.
        4. Perform KMeans clustering.
        5. Reduce the dimensionality of the word vectors for visualization.
        6. Create clusters dictionary and display results.
        7. Visualize the results.
        8. Show the reduced word embedding matrix.
        """
        st.title("KMeans Clustering")

        # Step 1: Get input from the user
        words, k, method, n_dimensions, distance_method = self.get_user_input()

        # Step 2: Wait for the user to confirm input before processing
        if st.button("Run Clustering"):
            # Step 3: Preprocess and filter the words
            filtered_words, word_vectors = self.preprocess_words(words, distance_method)

            # Step 4: Perform KMeans clustering
            if word_vectors is None:
                st.error("Please enter all valid words.")
                return

            labels = self.perform_kmeans(word_vectors, k, distance_method)

            # Step 5: Dimensionality reduction for visualization
            reduced_vectors = self.reduce_dimensions(word_vectors, method, n_dimensions)

            # Step 6: Create clusters dictionary and display results
            self.display_clusters(filtered_words, labels, k)

            # Step 7: Visualize the results
            plot_df = self.create_plot_dataframe(
                filtered_words, labels, reduced_vectors, n_dimensions
            )
            fig = self.visualize_words_cluster(n_dimensions, plot_df)
            st.plotly_chart(fig)

            # Step 8: Show the reduced word embedding matrix
            self.display_reduced_matrix(filtered_words, reduced_vectors, n_dimensions)

    def get_user_input(self) -> tuple[List[str], int, str, int, str]:
        """
        Get input from the user and return the input words, number of clusters (k),
        dimensionality reduction method, number of dimensions for visualization, and
        distance computing method.

        Returns:
            tuple[List[str], int, str, int, str]
        """
        input_words = st.text_input("Enter words (comma-separated):", "")
        words = input_words.split(",")
        k = st.slider("Number of clusters (k):", min_value=2, max_value=10, value=2)
        method = st.selectbox(
            "Select dimensionality reduction method:", ["PCA", "t-SNE"]
        )
        n_dimensions = st.selectbox("Number of dimensions for visualization:", [2, 3])
        distance_method = st.selectbox(
            "Select distance computing method:", ["euclidean", "cosine"]
        )
        return words, k, method, n_dimensions, distance_method

    def preprocess_words(
        self, words: List[str], distance_method: str
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Preprocess the list of words by filtering out words that are not in the model,
        and computing distances for the remaining words.

        Args:
            words (List[str]): The list of words to preprocess.
            distance_method (str): The distance metric to use when computing distances.
        Returns:
            Tuple[List[str], Optional[np.ndarray]]: A tuple containing the filtered list
            of words and the computed distances. If any words are missing from the model,
            the second element of the tuple is None.
        """
        filtered_words = [word.strip() for word in words if word.strip() in self.model]
        word_vectors = self.check_word_in_model(filtered_words)
        if word_vectors is not None and word_vectors.size > 0:
            word_vectors = self.compute_distances(word_vectors, distance_method)
            return filtered_words, word_vectors
        return filtered_words, None

    def perform_kmeans(
        self, word_vectors: np.ndarray, k: int, distance_method: str
    ) -> np.ndarray:
        """
        Perform KMeans clustering on the word vectors.

        Args:
            word_vectors (np.ndarray): The word vectors to cluster.
            k (int): The number of clusters to form.
            distance_method (str): The distance metric to use when computing distances.

        Returns:
            np.ndarray: The cluster labels for each word vector.
        """
        kmeans = CustomKMeans(n_clusters=k, random_state=0, metric=distance_method)
        kmeans.fit(word_vectors)
        return kmeans.labels_

    def display_clusters(self, words: List[str], labels: np.ndarray, k: int) -> None:
        """
        Display the words in their respective clusters.

        Args:
            words (List[str]): The list of words.
            labels (np.ndarray): The cluster labels for each word.
            k (int): The number of clusters.
        """
        clusters: Dict[int, List[str]] = {i: [] for i in range(k)}

        for word, label in zip(words, labels):
            clusters[label].append(word)

        st.subheader("Clustered Words")
        for cluster, words_in_cluster in clusters.items():
            st.write(f"Cluster {cluster}: {words_in_cluster}")

    def create_plot_dataframe(
        self,
        words: List[str],
        labels: np.ndarray,
        reduced_vectors: np.ndarray,
        n_dimensions: int,
    ) -> pd.DataFrame:
        """
        Create a DataFrame for visualization of the words in 2D or 3D space.

        Args:
            words (List[str]): The list of words.
            labels (np.ndarray): The cluster labels for each word.
            reduced_vectors (np.ndarray): The reduced word vectors.
            n_dimensions (int): The number of dimensions for visualization.

        Returns:
            pd.DataFrame: The DataFrame for visualization.
        """
        plot_data = {"Word": words, "Cluster": labels}
        for i in range(n_dimensions):
            plot_data[f"Dim{i+1}"] = reduced_vectors[:, i]
        return pd.DataFrame(plot_data)

    def display_reduced_matrix(
        self, words: List[str], reduced_vectors: np.ndarray, n_dimensions: int
    ) -> None:
        """
        Display the reduced word embeddings in a DataFrame.

        Args:
            words (List[str]): The list of words.
            reduced_vectors (np.ndarray): The reduced word vectors.
            n_dimensions (int): The number of dimensions for visualization.
        """
        reduced_df = pd.DataFrame(
            reduced_vectors,
            index=words,
            columns=[f"Dim{i+1}" for i in range(n_dimensions)],
        )
        st.subheader("Matrix of Word Embeddings (Reduced Dimensions)")
        st.dataframe(reduced_df)

    def visualize_words_cluster(self, n_dimensions: int, plot_df: pd.DataFrame):
        """
        Visualize words in 2D or 3D space with cluster labels.

        Args:
            n_dimensions (int): The number of dimensions for visualization.
            plot_df (pd.DataFrame): The DataFrame containing words and their reduced vectors.

        Returns:
            plotly.graph_objects.Figure: The figure for visualization.
        """
        fig = plt.figure(figsize=(10, 7))
        if n_dimensions == 2:
            fig = px.scatter(
                plot_df,
                x="Dim1",
                y="Dim2",
                color="Cluster",
                text="Word",
                hover_data=["Word", "Cluster"],
                color_continuous_scale="Viridis",
            )
            fig.update_traces(
                textposition="top left",
                marker={"size": 12, "line": {"width": 2, "color": "black"}},
            )

        elif n_dimensions == 3:
            fig = px.scatter_3d(
                plot_df,
                x="Dim1",
                y="Dim2",
                z="Dim3",
                color="Cluster",
                text="Word",
                hover_data=["Word", "Cluster"],
                color_continuous_scale="Viridis",
            )
            fig.update_traces(
                textposition="top left",
                marker={"size": 12, "line": {"width": 2, "color": "black"}},
            )
            # Adjust the layout to increase the figure size

            fig.update_layout(
                width=1000,  # Increase width of the plot
                height=800,  # Increase height of the plot
                scene={
                    "xaxis_title": "Dim1",
                    "yaxis_title": "Dim2",
                    "zaxis_title": "Dim3",
                },
            )
        return fig

    def compute_distances(self, word_vectors: np.ndarray, method: str) -> np.ndarray:
        """
        Compute distances between word vectors based on the selected method.

        Args:
            word_vectors (np.ndarray): The word vectors to compute distances for.
            method (str): The distance metric to use ('euclidean' or 'cosine').

        Returns:
            np.ndarray: The computed distances between word vectors.
        """
        if method == "cosine":
            return 1 - cosine_similarity(word_vectors)
        else:
            return word_vectors
