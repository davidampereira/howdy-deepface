import os

import cv2
import numpy as np

import paths


OPENCV_COSINE_SIMILARITY_THRESHOLD = 0.363
OPENCV_COSINE_DISTANCE_THRESHOLD = 1 - OPENCV_COSINE_SIMILARITY_THRESHOLD
OPENCV_NORM_L2_THRESHOLD = 1.128


class FaceBackendError(RuntimeError):
    pass


def _configured_path(config, option, default_path):
    path = config.get("core", option, fallback=str(default_path)).strip()
    return path or str(default_path)


def _opencv_model_path(filename):
    return paths.data_dir / "models" / "opencv" / filename


class FaceBackend:
    """OpenCV YuNet detector + SFace recognizer backend."""

    name = "opencv-yunet-sface"

    def __init__(self, config):
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise FaceBackendError(
                "OpenCV was built without FaceDetectorYN/FaceRecognizerSF support"
            )

        detector_model_path = _configured_path(
            config,
            "detector_model_path",
            _opencv_model_path("face_detection_yunet_2023mar.onnx"),
        )
        recognizer_model_path = _configured_path(
            config,
            "recognizer_model_path",
            _opencv_model_path("face_recognition_sface_2021dec.onnx"),
        )

        for model_path in (detector_model_path, recognizer_model_path):
            if not os.path.exists(model_path):
                raise FaceBackendError("Required face model is missing: " + model_path)

        score_threshold = config.getfloat("core", "detector_score_threshold", fallback=0.6)
        nms_threshold = config.getfloat("core", "detector_nms_threshold", fallback=0.3)
        top_k = config.getint("core", "detector_top_k", fallback=5000)

        self.detector = cv2.FaceDetectorYN.create(
            model=detector_model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=recognizer_model_path,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def warmup(self):
        pass

    def represent(self, frame, enforce_detection=True):
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)

        if faces is None or len(faces) == 0:
            if enforce_detection:
                raise FaceBackendError("No face detected")
            return []

        results = []
        for face in faces:
            face_box = face[:-1]
            aligned = self.recognizer.alignCrop(frame, face_box)
            embedding = self.recognizer.feature(aligned).flatten().astype(np.float32)
            x, y, w, h = face[:4].astype(int)
            results.append(
                {
                    "embedding": embedding.tolist(),
                    "facial_area": {
                        "x": max(0, x),
                        "y": max(0, y),
                        "w": max(0, w),
                        "h": max(0, h),
                    },
                }
            )

        return results


def resolve_video_certainty(config, distance_metric):
    certainty_raw = config.get("video", "certainty", fallback="auto").strip()

    if certainty_raw.lower() == "auto":
        if distance_metric == "euclidean_l2":
            return OPENCV_NORM_L2_THRESHOLD
        if distance_metric == "cosine":
            return OPENCV_COSINE_DISTANCE_THRESHOLD
        raise ValueError("OpenCV SFace auto certainty supports cosine or euclidean_l2")

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
    raise ValueError("OpenCV SFace supports cosine and euclidean_l2 distance only")


def encoding_to_model_index(flat_index, models):
    cumulative = 0
    for i, model in enumerate(models):
        cumulative += len(model["data"])
        if flat_index < cumulative:
            return i, model["label"]
    return len(models) - 1, models[-1]["label"]
