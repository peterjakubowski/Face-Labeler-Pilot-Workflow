from typing import Any

import numpy as np
import streamlit as st
from streamlit.connections import BaseConnection

from config import COMPARE_FACES_TOLERANCE, TOP_K


class FaceClassifierKNN(BaseConnection[dict]):

    def _connect(self, **kwargs: Any) -> dict:

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

        if self._instance.get('embeddings') is None:
            return ""

        number_of_embeddings = self._instance.get('embeddings', np.empty(0)).shape[0]
        number_of_unique_names = np.unique(self._instance.get('names', np.empty(0))).shape[0]

        return (f"Face classifier contains **{number_of_embeddings}** total face embeddings "
                f"and **{number_of_unique_names}** unique names")

    def is_in(self, embedding: np.ndarray) -> bool:
        """
        Check if an embedding (or similar) is already in the list of embeddings.
        :param embedding: The new embedding to check.
        :return: True or False
        """

        if (current_embeddings := self._instance.get('embeddings')) is None:
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

        self._instance['embeddings'] = np.array(embeddings, dtype=np.float32)
        self._instance['names'] = np.array(names, dtype=str)

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

        unidentified_face_embedding = np.asarray(embedding, dtype=np.float32)

        known_face_embeddings = self._instance.get('embeddings', None)

        if known_face_embeddings is None:
            return "Unknown face", 0.0

        distances = np.linalg.norm(known_face_embeddings - unidentified_face_embedding, axis=1)

        top_k_indices = np.argsort(distances)[:k]

        neighbor_distances = distances[top_k_indices]
        neighbor_names = self._instance.get('names')[top_k_indices]

        valid_mask = neighbor_distances <= threshold
        valid_distances = neighbor_distances[valid_mask]
        valid_names = neighbor_names[valid_mask]

        if valid_distances.shape[0] < 1:
            return "Unknown face", 0.0

        weights = 1.0 / (valid_distances + 1e-5)

        scores = {}

        for name in np.unique(valid_names):
            scores[name] = np.sum(weights[valid_names == name])

        winner = max(scores, key=scores.get)

        total_neighborhood_weights = np.sum(weights)

        confidence_percentage = (scores[winner] / total_neighborhood_weights) * 100

        return str(winner), round(float(confidence_percentage), 2)


face_conn = st.connection("face_classifier", type=FaceClassifierKNN)
