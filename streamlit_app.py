import streamlit as st
from gensim.models import KeyedVectors

from src.problem3.word2vec_app import Word2VecAnalysisApp
from src.problem_3.k_means_app import KMeansClusteringApp


@st.cache_resource
def load_google_news_model() -> KeyedVectors:
    """
    Load the Google News Word2Vec model.

    Returns:
        KeyedVectors: The loaded Word2Vec model.
    """
    return KeyedVectors.load_word2vec_format(
        "DATA/GoogleNews-vectors-negative300.bin", binary=True
    )


def main() -> None:
    """
    Main function for running the Streamlit application.

    This function serves as the entry point for the application, allowing users to choose between
    KMeans Clustering and Word2Vec Analysis. It initializes the selected application mode
    and loads the necessary Word2Vec model for use within the application.

    The application includes a sidebar for navigation between modes and displays the respective
    application interface based on the user's selection.
    """
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the application mode", ["KMeans Clustering", "Word2Vec Analysis"]
    )
    # Load the Word2Vec model
    model = load_google_news_model()

    # Choose the application mode
    if app_mode == "KMeans Clustering":
        k_means_app = KMeansClusteringApp(model)
        k_means_app.run()
    elif app_mode == "Word2Vec Analysis":
        word2vec_analysis_app = Word2VecAnalysisApp(model)
        word2vec_analysis_app.run()


if __name__ == "__main__":
    main()
