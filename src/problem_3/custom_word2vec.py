from typing import Dict, List, Optional, Tuple

import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm


class CustomWord2Vec:
    """
    A class for performing word vector operations using a word2vec model.

    This class provides methods for calculating word similarities, performing word analogies,
    and identifying the odd word out in a list of words.

    Attributes:
        model: The word2vec model used to retrieve word vectors.
        method: The method used for similarity computation ('cosine' or 'euclidean').
    """

    def __init__(self, model: KeyedVectors, method: str) -> None:
        """
        Initialize the CustomWord2Vec with a word2vec model and a similarity method.

        Args:
            model: The word2vec model to be used for vector operations.
            method: The method for computing similarity ('cosine' or 'euclidean').
        """
        self.model = model
        self.method = method

    def _compute_similarity(self, target_vec: (np.ndarray)) -> Dict[str, np.ndarray]:
        """
        Compute the similarity between a target vector and all word vectors in the model.

        Args:
            target_vec (np.ndarray): The target word vector to compare.

        Returns:
            dict: A dictionary with words as keys and their similarity scores as values.
        """
        similarities = {}
        for word in self.model.key_to_index.keys():
            word_vec = self.model[word]  # Load word vector lazily
            if self.method == "cosine":
                sim = np.dot(target_vec, word_vec) / (norm(word_vec) * norm(target_vec))
            else:
                sim = -norm(
                    target_vec - word_vec
                )  # Use negative distance for similarity
            similarities[word] = sim
        return similarities

    def most_similar_analogy(
        self,
        positive_list: List[str],
        negative_list: Optional[List[str]] = None,
        topn=10,
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar words based on a word analogy (e.g., king - man + woman).

        Args:
            positive_list (List[str]): List of words that contribute positively to the analogy.
            negative_list (Optional[List[str]]): List of words that contribute negatively
            to the analogy (optional).
            topn (int): The number of top similar words to return.

        Returns:
           list (List[Tuple[str, float]]): A sorted list of tuples containing words and their
           similarity scores, limited to top N.
        """
        positive_vecs = np.array([self.model[word] for word in positive_list])
        negative_vecs = (
            np.array([self.model[word] for word in negative_list])
            if negative_list
            else np.zeros_like(positive_vecs)
        )

        # Calculate the target vector for the analogy
        target_vec = np.sum(positive_vecs, axis=0) - np.sum(negative_vecs, axis=0)

        # Compute similarities using lazy loading
        similarities = self._compute_similarity(target_vec)

        # Sort and return top N words
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:topn]

    def doesnt_match(self, word_list: List[str]) -> str:
        """
        Identify the word that does not match the others in a given list.

        Args:
            word_list (List[str]): A list of words to check for the odd one out.

        Returns:
            str: The word that is most dissimilar to the others.
        """
        word_vecs = np.array([self.model[word] for word in word_list])
        mean_vec = np.mean(word_vecs, axis=0)

        # Compute distances using lazy loading
        distances = self._compute_similarity(mean_vec)

        # Return the word with the smallest similarity (most different)
        word_distances = {word: distances[word] for word in word_list}
        return sorted(word_distances.items(), key=lambda x: x[1])[0][0]
