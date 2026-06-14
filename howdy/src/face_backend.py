from contextlib import contextmanager
import os
import sys
import warnings

import numpy as np


INSIGHTFACE_COSINE_SIMILARITY_THRESHOLD = 0.48
INSIGHTFACE_COSINE_DISTANCE_THRESHOLD = 1 - INSIGHTFACE_COSINE_SIMILARITY_THRESHOLD


class FaceBackendError(RuntimeError):
    pass


@contextmanager
def suppress_backend_noise():
    """Silence noisy native model initialization output."""
    sys.stdout.flush()
    sys.stderr.flush()

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        os.close(devnull_fd)


class FaceBackend:
    """InsightFace SCRFD + Buffalo recognition backend."""

    name = "insightface-buffalo"

    def __init__(self, config):
        try:
            with suppress_backend_noise():
                from insightface.app import FaceAnalysis
        except ImportError as err:
            raise FaceBackendError(
                "Cannot import insightface; install insightface and onnxruntime"
            ) from err

        model_pack = config.get("core", "insightface_model_pack", fallback="buffalo_s")
        if model_pack not in ("buffalo_s", "buffalo_sc"):
            raise FaceBackendError(
                "Unsupported InsightFace model pack: " + model_pack
            )

        model_root = config.get("core", "insightface_model_root", fallback="~/.insightface")
        detector_size = config.getint("core", "detector_size", fallback=320)
        detector_threshold = config.getfloat("core", "detector_score_threshold", fallback=0.5)

        try:
            with suppress_backend_noise():
                self.app = FaceAnalysis(
                    name=model_pack,
                    root=model_root,
                    allowed_modules=["detection", "recognition"],
                    providers=["CPUExecutionProvider"],
                )
                self.app.prepare(
                    ctx_id=-1,
                    det_thresh=detector_threshold,
                    det_size=(detector_size, detector_size),
                )
        except Exception as err:
            raise FaceBackendError(str(err)) from err

    def warmup(self):
        pass

    def represent(self, frame, enforce_detection=True):
        with suppress_backend_noise():
            faces = self.app.get(frame)
        if not faces:
            if enforce_detection:
                raise FaceBackendError("No face detected")
            return []

        results = []
        for face in faces:
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append(
                {
                    "embedding": np.asarray(embedding, dtype=np.float32).flatten().tolist(),
                    "facial_area": {
                        "x": max(0, x1),
                        "y": max(0, y1),
                        "w": max(0, x2 - x1),
                        "h": max(0, y2 - y1),
                    },
                }
            )

        if not results and enforce_detection:
            raise FaceBackendError("No face embedding extracted")

        return results


def resolve_video_certainty(config, distance_metric):
    certainty_raw = config.get("video", "certainty", fallback="auto").strip()

    if certainty_raw.lower() == "auto":
        if distance_metric == "cosine":
            return INSIGHTFACE_COSINE_DISTANCE_THRESHOLD
        if distance_metric == "euclidean_l2":
            return 1.0
        raise ValueError("InsightFace auto certainty supports cosine or euclidean_l2")

    certainty_value = float(certainty_raw)
    if certainty_value <= 0:
        raise ValueError("certainty must be greater than 0")

    if distance_metric == "cosine" and certainty_value >= 1:
        raise ValueError("certainty must be lower than 1 for cosine distance")

    return certainty_value


def compute_distances(face_encoding, encodings_np, distance_metric):
    if distance_metric == "cosine":
        face_norm = np.linalg.norm(face_encoding)
        enc_norms = np.linalg.norm(encodings_np, axis=1)
        cosine_similarities = np.dot(encodings_np, face_encoding) / (
            enc_norms * face_norm + 1e-10
        )
        return 1 - cosine_similarities
    if distance_metric == "euclidean_l2":
        face_norm_vec = face_encoding / (np.linalg.norm(face_encoding) + 1e-10)
        enc_norm_vecs = encodings_np / (
            np.linalg.norm(encodings_np, axis=1, keepdims=True) + 1e-10
        )
        return np.linalg.norm(enc_norm_vecs - face_norm_vec, axis=1)
    raise ValueError("InsightFace supports cosine and euclidean_l2 distance only")


def encoding_to_model_index(flat_index, models):
    cumulative = 0
    for i, model in enumerate(models):
        cumulative += len(model["data"])
        if flat_index < cumulative:
            return i, model["label"]
    return len(models) - 1, models[-1]["label"]
