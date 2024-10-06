import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors

from .custom_word2vec import CustomWord2Vec
from .streamlit_base import StreamlitBase


class Word2VecAnalysisApp(StreamlitBase):
    """Streamlit application for analyzing word embeddings using Word2Vec."""

    def __init__(self, model: KeyedVectors) -> None:
        """
        Initialize the Word2VecAnalysisApp with a word2vec model.

        Args:
            model: The word2vec model to be used for analysis.
        """
        super().__init__(model)

    def run(self) -> None:
        """Run the Streamlit app for Word2Vec analysis."""
        st.title("Word2Vec Analysis")
        feature = st.selectbox(
            "Select Feature:", ["Word Analogy", "Most Similar", "Doesn't Match"]
        )
        method = st.radio(
            "Select dimensionality reduction method for visualization:",
            ("PCA", "t-SNE"),
        )
        dimension = st.radio("Select visualization dimension:", ("2D", "3D"))

        # Word Analogy
        if feature == "Word Analogy":
            self.word_analogy(method, dimension)

        # Most Similar
        elif feature == "Most Similar":
            self.most_similar(method, dimension)

        # Doesn't Match
        elif feature == "Doesn't Match":
            self.doesnt_match(method, dimension)

    def word_analogy(self, method: str, dimension: str) -> None:
        """
        Perform word analogy based on user input and visualize the results.

        Args:
            method (str): The dimensionality reduction method to use for visualization
            ('PCA' or 't-SNE').
            dimension (str): The dimensionality of the visualization ('2D' or '3D').
        """
        st.header("Word Analogy (e.g., woman + royal - man = queen)")
        positive_words = st.text_input("Enter positive words (comma-separated):", "")
        negative_words = st.text_input("Enter negative words (comma-separated):", "")
        top_n = st.slider("Top N results for analogy", 1, 30, 10)
        distance_method = st.selectbox(
            "Select distance computing method:", ["euclidean", "cosine"]
        )

        if st.button("Find Word Analogy"):
            if positive_words and negative_words:  # Check if input is provided
                positive_list = [word.strip() for word in positive_words.split(",")]
                negative_list = [word.strip() for word in negative_words.split(",")]

                custom_model = CustomWord2Vec(self.model, distance_method)
                analogy_result = custom_model.most_similar_analogy(
                    positive_list=positive_list, negative_list=negative_list, topn=top_n
                )
                analogy_df = pd.DataFrame(
                    analogy_result, columns=["Word", "Similarity"]
                )
                st.dataframe(analogy_df)

                words_to_visualize = (
                    positive_list + negative_list + [w for w, _ in analogy_result]
                )
                word_vectors = np.array(
                    [
                        self.model[word]
                        for word in words_to_visualize
                        if word in self.model
                    ]
                )
                reduced_vectors = self.reduce_dimensions(
                    word_vectors, method, 2 if dimension == "2D" else 3
                )
                labels = np.array(
                    [0] * len(positive_list)
                    + [1] * len(negative_list)
                    + [2] * len(analogy_result)
                )
                fig = self.visualize_words(words_to_visualize, reduced_vectors, labels)
                st.plotly_chart(fig)
            else:
                st.warning("Please enter both positive and negative words.")

    def most_similar(self, method: str, dimension: str) -> None:
        """
        Find and visualize the most similar words based on user input.

        Args:
            method (str): The dimensionality reduction method to use for visualization
            ('PCA' or 't-SNE').
            dimension (str): The dimensionality of the visualization ('2D' or '3D').
        """
        st.header("Most Similar Words")
        input_words = st.text_input(
            "Enter words to find similar words (comma-separated):", ""
        )
        top_n = st.slider("Top N similar words", 1, 30, 10)
        distance_method = st.selectbox(
            "Select distance computing method:", ["euclidean", "cosine"]
        )

        if st.button("Find Similar Words"):
            if input_words:  # Check if input is provided
                positive_list = [word.strip() for word in input_words.split(",")]
                custom_model = CustomWord2Vec(self.model, distance_method)
                similar_words = custom_model.most_similar_analogy(
                    positive_list=positive_list, topn=top_n
                )
                similar_df = pd.DataFrame(similar_words, columns=["Word", "Similarity"])
                st.dataframe(similar_df)

                words_to_visualize = positive_list + [w for w, _ in similar_words]
                word_vectors = np.array(
                    [
                        self.model[word]
                        for word in words_to_visualize
                        if word in self.model
                    ]
                )
                reduced_vectors = self.reduce_dimensions(
                    word_vectors, method, 2 if dimension == "2D" else 3
                )
                labels = np.array([0] * len(positive_list) + [1] * len(similar_words))
                fig = self.visualize_words(words_to_visualize, reduced_vectors, labels)
                st.plotly_chart(fig)
            else:
                st.warning("Please enter words to find similar words.")

    def doesnt_match(self, method: str, dimension: str) -> None:
        """
        Identify and visualize the word that doesn't match in a given list.

        Args:
            method (str): The dimensionality reduction method to use for visualization
            ('PCA' or 't-SNE').
            dimension (str): The dimensionality of the visualization ('2D' or '3D').
        """
        st.header("Which word doesn't match?")
        words_to_check = st.text_input("Enter words to check (comma-separated):", "")
        distance_method = st.selectbox(
            "Select distance computing method:", ["euclidean", "cosine"]
        )

        if st.button("Find Doesn't Match"):
            if words_to_check:  # Check if input is provided
                word_list = [word.strip() for word in words_to_check.split(",")]
                custom_model = CustomWord2Vec(self.model, distance_method)
                odd_word = custom_model.doesnt_match(word_list)
                st.write(f"The word that doesn't match is: **{odd_word}**")

                word_vectors = np.array(
                    [self.model[word] for word in word_list if word in self.model]
                )
                reduced_vectors = self.reduce_dimensions(
                    word_vectors, method, 2 if dimension == "2D" else 3
                )
                labels = np.array([0] * len(word_list))
                fig = self.visualize_words(word_list, reduced_vectors, labels)
                st.plotly_chart(fig)
            else:
                st.warning("Please enter words to check.")
