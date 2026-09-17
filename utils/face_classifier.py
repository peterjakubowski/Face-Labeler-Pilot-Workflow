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

        return self._instance.get('embeddings') is not None

    def load_reference_data(self, reference_data: list[dict]):

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

        new_embedding = np.asarray(embedding, dtype=np.float32)

        if self._instance.get('embeddings', None):
            self._instance['embeddings'] = np.array([new_embedding], dtype=np.float32)
            self._instance['names'] = np.array([person_shown], dtype=str)
            return

        self._instance['embeddings'] = np.vstack([self._instance.get('embeddings'), new_embedding])
        self._instance['names'] = np.append(self._instance.get('names'), person_shown)

    def predict(self, embedding: np.ndarray, k: int = TOP_K, threshold: float = COMPARE_FACES_TOLERANCE) -> tuple[str, float]:

        unidentified_face_embedding = np.asarray(embedding, dtype=np.float32)

        known_face_embeddings = np.asarray(self._instance.get('embeddings', np.empty(0)), dtype=np.float32)

        if known_face_embeddings.shape[0] < 1:
            return "Unknown face", 0.0

        distances = np.linalg.norm(known_face_embeddings - unidentified_face_embedding, axis=1)

        top_k_indices = np.argpartition(distances, kth=k)[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

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
