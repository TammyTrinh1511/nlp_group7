from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class StreamlitBase(ABC):
    """
    Abstract base class for Streamlit applications using word2vec models.

    This class provides a foundation for building Streamlit applications that
    utilize word embeddings from a word2vec model, offering methods for
    dimensionality reduction, visualization, and word vector validation.

    Attributes:
        model: The word2vec model used for embeddings.
    """

    def __init__(self, model) -> None:
        """
        Initialize the StreamlitBase instance with a word2vec model.

        Args:
            model: The word2vec model to be used in the application.
        """
        self.model = model

    @abstractmethod
    def run(self) -> None:
        """
        Abstract method to run the Streamlit application.

        This method should be implemented in subclasses to define the
        application's functionality.
        """
        pass

    def reduce_dimensions(
        self, word_vectors: np.ndarray, method: str, n_components: int
    ) -> np.ndarray:
        """
        Reduce the dimensionality of word vectors using specified method.

        Args:
            word_vectors (np.ndarray): The input word vectors to be reduced.
            method (str): The method to use for dimensionality reduction ('PCA' or 't-SNE').
            n_components (int): The number of components to keep after reduction.

        Returns:
            np.ndarray: The reduced word vectors.
        """
        if method == "PCA":
            pca = PCA(n_components=n_components)
            return pca.fit_transform(word_vectors)
        elif method == "t-SNE":
            perplexity = min(30, word_vectors.shape[0] - 1)
            tsne = TSNE(
                n_components=n_components, perplexity=perplexity, random_state=0
            )
            return tsne.fit_transform(word_vectors)

    def visualize_words(
        self, words: List[str], reduced_vectors: np.ndarray, labels: np.ndarray
    ):
        """
        Visualize words in 2D or 3D space based on reduced vectors and cluster labels.

        Args:
            words (List[str]): The list of words to visualize.
            reduced_vectors (np.ndarray): The reduced word vectors for visualization.
            labels (np.ndarray): The cluster labels corresponding to each word.

        Returns:
            px.Figure: A Plotly figure object containing the scatter plot.
        """
        plot_data = {
            "Word": words,
            "Cluster": labels,
        }

        for i in range(reduced_vectors.shape[1]):
            plot_data[f"Dim{i + 1}"] = reduced_vectors[:, i]

        plot_df = pd.DataFrame(plot_data)

        color_sequence = ["blue", "red", "green"]

        if reduced_vectors.shape[1] == 2:
            fig = px.scatter(
                plot_df,
                x="Dim1",
                y="Dim2",
                color="Cluster",
                text="Word",
                hover_data=["Word", "Cluster"],
                title="2D Visualization of Words",
                color_discrete_sequence=color_sequence,
            )
            fig.update_traces(
                textposition="top left",
                marker={"size": 12, "line": {"width": 2, "color": "black"}},
            )
        elif reduced_vectors.shape[1] == 3:
            fig = px.scatter_3d(
                plot_df,
                x="Dim1",
                y="Dim2",
                z="Dim3",
                color="Cluster",
                text="Word",
                hover_data=["Word", "Cluster"],
                title="3D Visualization of Words",
                color_discrete_sequence=color_sequence,
            )
            fig.update_traces(
                textposition="top left",
                marker={"size": 12, "line": {"width": 2, "color": "black"}},
            )

        return fig

    def check_word_in_model(self, filtered_words: List[str]) -> Optional[np.ndarray]:
        """
        Check if all words in the filtered list are in the word2vec model.

        If a word is not in the model, it is added to the missing_words list and an error message
        is displayed to the user. If there are missing words, return None. Otherwise, return
        the list of word vectors.

        Args:
            filtered_words (List[str]): The list of words to check.

        Returns:
            Optional[np.ndarray]: Embedding of words, otherwise None.
        """
        word_vectors = []
        missing_words = []

        for word in filtered_words:
            try:
                word_vectors.append(self.model[word])
            except KeyError:
                missing_words.append(word)

        if missing_words:
            st.error(
                f"The following words don't have embeddings: {', '.join(missing_words)}"
            )
            return None
        else:
            word_vectors = np.array(word_vectors)
            return word_vectors
