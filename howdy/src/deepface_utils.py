# Shared DeepFace utility functions used by compare.py, test.py, and other modules

import numpy as np


def resolve_video_certainty(config, model_name, distance_metric):
    """Resolve certainty setting using DeepFace-native thresholds.

    Returns the distance threshold to use for face matching. If config has
    certainty = "auto", uses DeepFace's built-in threshold for the given
    model + metric combination. Otherwise parses the numeric value and
    validates it.
    """
    certainty_raw = config.get("video", "certainty", fallback="auto").strip()

    if certainty_raw.lower() == "auto":
        from deepface.modules.verification import find_threshold

        return find_threshold(model_name, distance_metric)

    certainty_value = float(certainty_raw)
    if certainty_value <= 0:
        raise ValueError("certainty must be greater than 0")

    if distance_metric in ("cosine", "euclidean_l2") and certainty_value >= 1:
        raise ValueError(
            "certainty must be lower than 1 for cosine/euclidean_l2 metrics"
        )

    return certainty_value


def compute_distances(face_encoding, encodings_np, distance_metric):
    """Compute distances between a face embedding and all stored encodings.

    Args:
        face_encoding: 1D numpy array of the detected face embedding.
        encodings_np: 2D numpy array of stored embeddings (N x dim).
        distance_metric: One of "cosine", "euclidean", or "euclidean_l2".

    Returns:
        1D numpy array of distances, one per stored encoding.
    """
    if distance_metric == "cosine":
        # Cosine distance = 1 - cosine_similarity
        face_norm = np.linalg.norm(face_encoding)
        enc_norms = np.linalg.norm(encodings_np, axis=1)
        cosine_similarities = np.dot(encodings_np, face_encoding) / (
            enc_norms * face_norm + 1e-10
        )
        return 1 - cosine_similarities
    elif distance_metric == "euclidean_l2":
        # L2-normalize then compute euclidean distance
        face_norm_vec = face_encoding / (np.linalg.norm(face_encoding) + 1e-10)
        enc_norm_vecs = encodings_np / (
            np.linalg.norm(encodings_np, axis=1, keepdims=True) + 1e-10
        )
        return np.linalg.norm(enc_norm_vecs - face_norm_vec, axis=1)
    else:
        # Default: raw euclidean distance
        return np.linalg.norm(encodings_np - face_encoding, axis=1)


def encoding_to_model_index(flat_index, models):
    """Map a flat encodings array index to (model_index, model_label).

    When multiple models each have multiple data entries, the flat index
    into the concatenated encodings array needs to be mapped back to the
    correct model.

    Args:
        flat_index: Index into the flattened encodings array.
        models: List of model dicts, each with a "data" list and "label" string.

    Returns:
        Tuple of (model_index, label_string).
    """
    cumulative = 0
    for i, model in enumerate(models):
        cumulative += len(model["data"])
        if flat_index < cumulative:
            return i, model["label"]
    # Fallback — should not happen with valid data
    return len(models) - 1, models[-1]["label"]
