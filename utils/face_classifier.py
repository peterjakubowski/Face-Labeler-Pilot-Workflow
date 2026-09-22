from typing import Any

import numpy as np
import streamlit as st
from streamlit.connections import BaseConnection

from config import COMPARE_FACES_TOLERANCE, TOP_K


class FaceClassifierKNN(BaseConnection[dict]):
    """
    K nearest neighbors face classifier as a Streamlit connection class.
    """

    def _connect(self, **kwargs: Any) -> dict:
        """
        A connection instance is a dictionary with two keys: embeddings and names.
        When the connection is uninitialized, both values are None.
        When initialized, values are numpy arrays.
        :param kwargs:
        :return: Embeddings and dames dictionary
        """

        return {"embeddings": None,
                "names": None
                }

    def is_initialized(self) -> bool:
        """
        Check if the list of embeddings has been created.
        :return: True or False
        """

        return self._instance.get('embeddings') is not None

    def info(self) -> str:
        """
        Returns a string of summary text about the face classifier with counts of embeddings and unique names.
        :return: String of summary statistics
        """

        if not self.is_initialized():
            return "Face classifier is uninitialized."

        number_of_embeddings = self._instance.get('embeddings').shape[0]
        number_of_unique_names = np.unique(self._instance.get('names')).shape[0]

        return (f"Face classifier contains **{number_of_embeddings}** total face embeddings "
                f"and **{number_of_unique_names}** unique names.")

    def is_in(self, embedding: np.ndarray) -> bool:
        """
        Check if an embedding (or similar) is already in the list of embeddings.
        :param embedding: The new embedding to check.
        :return: True or False
        """

        current_embeddings: None | np.ndarray = self._instance.get('embeddings', None)

        if current_embeddings is None or current_embeddings.shape[0] < 1:
            return False

        new_embedding = np.asarray(embedding, dtype=np.float32)
        distances = np.linalg.norm(current_embeddings - new_embedding, axis=1)

        return distances[np.argmin(distances)] <= 1e-3

    def unique_names(self) -> list[str]:
        """
        Returns a sorted list of unique names from the list of names.
        :return: List of unique names
        """

        if (names := self._instance.get('names')) is None:
            return []

        return sorted(np.unique(names))

    def load_reference_data(self, reference_data: list[dict]):
        """
        Loads the reference data (labeled faces/embeddings) into the face classifier.
        :param reference_data: List of dictionaries with names and embeddings.
        :return:
        """

        embeddings = []
        names = []

        for ref in reference_data:
            ref_embedding = ref.get('embedding', None)
            ref_name = ref.get('name', None)
            if ref_embedding is not None and ref_name is not None:
                embeddings.append(np.asarray(ref_embedding, dtype=np.float32))
                names.append(str(ref_name))

        if embeddings and names:
            self._instance['embeddings'] = np.array(embeddings, dtype=np.float32)
            self._instance['names'] = np.array(names, dtype=str)
        else:
            self._instance['embeddings'] = np.empty((0, 128))
            self._instance['names'] = np.empty(0)

    def add_new_face(self, person_shown: str, embedding: np.ndarray):
        """
        Adds a new face or identity to the face classifier.
        :param person_shown: The name of the person shown
        :param embedding: The embedding of the face
        :return:
        """

        new_embedding = np.asarray(embedding, dtype=np.float32)

        if self._instance.get('embeddings') is None:
            self._instance['embeddings'] = np.array([new_embedding], dtype=np.float32)
            self._instance['names'] = np.array([person_shown], dtype=str)
            return

        self._instance['embeddings'] = np.vstack([self._instance.get('embeddings'), new_embedding])
        self._instance['names'] = np.append(self._instance.get('names'), person_shown)

    def predict(self, embedding: np.ndarray, k: int = TOP_K, threshold: float = COMPARE_FACES_TOLERANCE) -> tuple[str, float]:
        """
        Predicts the name of an unidentified face using the face classifier.
        Compares an unknown embedding against a list of labeled embeddings to find
        the nearest neighbors, casts weighted votes based on distance, and returns
        the predicted name along with a confidence percentage.
        :param embedding: The face embedding to identify.
        :param k: The number of nearest neighbors to include in the vote.
        :param threshold: The maximum distance a neighbor can be to be considered.
        :return: Tuple with predicted name and confidence percentage
        """

        # make sure the face embedding we're trying to name is a numpy array
        unidentified_face_embedding = np.asarray(embedding, dtype=np.float32)
        # retrieve array of faces that have been named
        known_face_embeddings: None | np.ndarray = self._instance.get('embeddings', None)
        # If we don't know any faces yet, we can't predict a name
        if known_face_embeddings is None or known_face_embeddings.shape[0] < 1:
            return "Unknown face", 0.0

        # calculate the Euclidean distance between all known face embeddings and face embedding we're trying to name
        distances = np.linalg.norm(known_face_embeddings - unidentified_face_embedding, axis=1)
        # find the indices for the k nearest neighbors
        top_k_indices = np.argsort(distances)[:k]
        # get distances and names for the k nearest neighbors
        neighbor_distances = distances[top_k_indices]
        neighbor_names = self._instance.get('names')[top_k_indices]
        # keep valid neighbors, remove any neighbors that are above the threshold distance
        valid_mask = neighbor_distances <= threshold
        valid_distances = neighbor_distances[valid_mask]
        valid_names = neighbor_names[valid_mask]
        # if we don't have any neighbors below the threshold, we can't predict a name
        if valid_distances.shape[0] < 1:
            return "Unknown face", 0.0
        # calculate the inverse distances weights for all valid distances, add a small value to avoid zero division
        weights = 1.0 / (valid_distances + 1e-5)
        # keep a tally of scores by name
        scores: dict[str, np.float32] = {}
        # iterate over each name and tally its weighted score
        for name in np.unique(valid_names):
            scores[name] = np.sum(weights[valid_names == name])
        # find the name with the max score
        winner = max(scores, key=scores.get)
        # sum up all the weights in the neighborhood
        total_neighborhood_weights = np.sum(weights)
        # calculate the confidence percentage for the winner's score
        confidence_percentage = (scores[winner] / total_neighborhood_weights) * 100

        return str(winner), round(float(confidence_percentage), 2)


face_conn = st.connection("face_classifier", type=FaceClassifierKNN)
