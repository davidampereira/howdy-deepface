#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PREFIX="${PREFIX:-/usr/local}"
SUDO="${SUDO:-sudo}"
BUILD_DIR="${BUILD_DIR:-build}"
PAM_CONF_DIR="${PAM_CONF_DIR:-/etc/pam.d}"
PAM_DIR="${PAM_DIR:-}"

info() {
	printf '==> %s\n' "$*"
}

fail() {
	printf 'error: %s\n' "$*" >&2
	exit 1
}

require_command() {
	command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

backend_name() {
	if grep -q "opencv-yunet-sface" howdy/src/face_backend.py; then
		printf 'opencv'
	elif grep -q "insightface-buffalo" howdy/src/face_backend.py; then
		printf 'buffalo'
	else
		fail "cannot determine backend from howdy/src/face_backend.py"
	fi
}

detect_pam_dir() {
	local module_path

	if [ -n "$PAM_DIR" ]; then
		printf '%s\n' "$PAM_DIR"
		return
	fi

	if [ -d "$PAM_CONF_DIR" ]; then
		module_path="$(
			awk '
				$0 !~ /^[[:space:]]*#/ {
					for (i = 1; i <= NF; i++) {
						if ($i ~ /^\/.*\/pam_howdy\.so$/) {
							print $i
							exit
						}
					}
				}
			' "$PAM_CONF_DIR"/* 2>/dev/null || true
		)"

		if [ -n "$module_path" ]; then
			dirname "$module_path"
			return
		fi
	fi

	printf '%s\n' "$PREFIX/lib/$(gcc -dumpmachine)/security"
}

create_venv() {
	info "Creating Python environment at $VENV_DIR"
	require_command "$UV_BIN"
	"$UV_BIN" venv --allow-existing --system-site-packages --python "$PYTHON_BIN" "$VENV_DIR"
	"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --upgrade pip setuptools wheel
}

check_opencv_python() {
	info "Checking OpenCV Python support"
	"$VENV_DIR/bin/python" - <<'PY'
import cv2
missing = [
    name for name in ("FaceDetectorYN", "FaceRecognizerSF")
    if not hasattr(cv2, name)
]
if missing:
    raise SystemExit(
        "OpenCV is missing required APIs: "
        + ", ".join(missing)
        + ". Install python3-opencv and libopencv-dev from your distro."
    )
print(cv2.__version__)
PY
}

install_buffalo_python() {
	info "Installing InsightFace backend Python dependencies"
	"$UV_BIN" pip install --python "$VENV_DIR/bin/python" insightface onnxruntime
}

prepare_buffalo_models() {
	local pack
	local root
	local model_dir

	pack="$(awk -F= '/^[[:space:]]*insightface_model_pack[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2}' howdy/src/config.ini)"
	root="$(awk -F= '/^[[:space:]]*insightface_model_root[[:space:]]*=/{sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2); print $2}' howdy/src/config.ini)"
	pack="${pack:-buffalo_s}"
	root="${root:-~/.insightface}"
	model_dir="${root/#\~/$HOME}/models/$pack"

	if [ -d "$model_dir" ] && find "$model_dir" -maxdepth 1 -name '*.onnx' | grep -q .; then
		info "Using existing InsightFace model pack at $model_dir"
		return
	fi

	info "Downloading InsightFace model pack $pack to $model_dir"
	INSIGHTFACE_MODEL_PACK="$pack" INSIGHTFACE_MODEL_ROOT="$root" "$VENV_DIR/bin/python" - <<'PY'
import os

from insightface.utils.storage import download

pack = os.environ["INSIGHTFACE_MODEL_PACK"]
root = os.environ["INSIGHTFACE_MODEL_ROOT"]
download("models", pack, force=True, root=root)
PY
}

download() {
	local url="$1"
	local output="$2"

	if [ -s "$output" ]; then
		info "Using existing $output"
		return
	fi

	info "Downloading $url"
	curl -L "$url" -o "$output"
}

install_opencv_models() {
	local model_dir="$PREFIX/share/howdy/models/opencv"
	local yunet="/tmp/face_detection_yunet_2023mar.onnx"
	local sface="/tmp/face_recognition_sface_2021dec.onnx"

	download \
		"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
		"$yunet"
	download \
		"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
		"$sface"

	info "Installing OpenCV model files to $model_dir"
	$SUDO install -d "$model_dir"
	$SUDO install -m 0644 "$yunet" "$model_dir/"
	$SUDO install -m 0644 "$sface" "$model_dir/"
}

check_buffalo_models() {
	local pack
	local root
	local model_dir

	pack="$(awk -F= '/^[[:space:]]*insightface_model_pack[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2}' howdy/src/config.ini)"
	root="$(awk -F= '/^[[:space:]]*insightface_model_root[[:space:]]*=/{sub(/^[[:space:]]*/, "", $2); sub(/[[:space:]]*$/, "", $2); print $2}' howdy/src/config.ini)"
	pack="${pack:-buffalo_s}"
	root="${root:-~/.insightface}"
	model_dir="${root/#\~/$HOME}/models/$pack"

	if [ ! -d "$model_dir" ] || ! find "$model_dir" -maxdepth 1 -name '*.onnx' | grep -q .; then
		fail "InsightFace model pack missing at $model_dir. Download $pack from the InsightFace model zoo and extract it there, or set insightface_model_root in config.ini."
	fi
}

configure_build() {
	local python_path="$VENV_DIR/bin/python"
	local pam_dir

	pam_dir="$(detect_pam_dir)"
	info "Installing PAM module to $pam_dir"

	if [ -d "$BUILD_DIR" ]; then
		info "Reconfiguring Meson build"
		meson setup --wipe "$BUILD_DIR" -Dpython_path="$python_path" -Dprefix="$PREFIX" -Dpam_dir="$pam_dir"
	else
		info "Configuring Meson build"
		meson setup "$BUILD_DIR" -Dpython_path="$python_path" -Dprefix="$PREFIX" -Dpam_dir="$pam_dir"
	fi
}

compile_and_install() {
	info "Compiling"
	meson compile -C "$BUILD_DIR"

	info "Installing to $PREFIX"
	$SUDO meson install -C "$BUILD_DIR"
}

smoke_test() {
	local py_path

	py_path="$PREFIX/lib/$(gcc -dumpmachine)/howdy"
	if [ ! -d "$py_path" ]; then
		py_path="$(find "$PREFIX/lib" -path '*/howdy/face_backend.py' -printf '%h\n' -quit 2>/dev/null || true)"
	fi

	[ -n "$py_path" ] || fail "could not find installed Howdy Python sources under $PREFIX/lib"

	info "Checking installed backend"
	PYTHONPATH="$py_path" "$VENV_DIR/bin/python" - <<'PY'
import configparser
from face_backend import FaceBackend
import paths_factory

config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())
backend = FaceBackend(config)
print(backend.name)
PY
}

main() {
	local backend
	backend="$(backend_name)"

	create_venv

	case "$backend" in
		opencv)
			check_opencv_python
			install_opencv_models
			;;
		buffalo)
			install_buffalo_python
			prepare_buffalo_models
			check_buffalo_models
			;;
	esac

	configure_build
	compile_and_install
	smoke_test

	info "Install complete for $backend backend"
}

main "$@"
