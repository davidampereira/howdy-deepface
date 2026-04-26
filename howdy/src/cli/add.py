# Save the face of the user in encoded form

# Import required modules
import time
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import configparser
import builtins
import numpy as np
import paths_factory

from recorders.video_capture import VideoCapture
from i18n import _

# Try to import insightface and give a nice error if we can't
# Add should be the first point where import issues show up
try:
    from insightface.app import FaceAnalysis
except ImportError as err:
    print(err)

    print(_("\nCan't import the insightface module, check the output of"))
    print("pip3 show insightface")
    sys.exit(1)

import cv2

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

insightface_model_pack = config.get("core", "model_pack", fallback="buffalo_sc")
det_size = config.getint("core", "det_size", fallback=320)

# Initialize InsightFace (suppress internal print statements)
_stdout = sys.stdout
sys.stdout = open(os.devnull, "w")
try:
    face_app = FaceAnalysis(
        name=insightface_model_pack,
        allowed_modules=["detection", "recognition"],
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=(det_size, det_size))
finally:
    sys.stdout.close()
    sys.stdout = _stdout

user = builtins.howdy_user
# The permanent file to store the encoded model in
enc_file = paths_factory.user_model_path(user)
# Known encodings
encodings = []

# Make the ./models folder if it doesn't already exist
if not os.path.exists(paths_factory.user_models_dir_path()):
    print(_("No face model folder found, creating one"))
    os.makedirs(paths_factory.user_models_dir_path())

# To try read a premade encodings file if it exists
try:
    encodings = json.load(open(enc_file))
except FileNotFoundError:
    encodings = []

# Print a warning if too many encodings are being added
if len(encodings) > 3:
    print(
        _(
            "NOTICE: Each additional model slows down the face recognition engine slightly"
        )
    )
    print(_("Press Ctrl+C to cancel\n"))

# Make clear what we are doing if not human
if not builtins.howdy_args.plain:
    print(_("Adding face model for the user ") + user)

# Set the default label
label = "Initial model"

# some id's can be skipped, but the last id is always the maximum
next_id = encodings[-1]["id"] + 1 if encodings else 0

# Get the label from the cli arguments if provided
if builtins.howdy_args.arguments:
    label = builtins.howdy_args.arguments[0]

# Or set the default label
else:
    label = _("Model #") + str(next_id)

# Keep de default name if we can't ask questions
if builtins.howdy_args.y:
    print(_('Using default label "%s" because of -y flag') % (label,))
else:
    # Ask the user for a custom label
    label_in = input(_("Enter a label for this new model [{}]: ").format(label))

    # Set the custom label (if any) and limit it to 24 characters
    if label_in != "":
        label = label_in[:24]

# Remove illegal characters
if "," in label:
    print(_('NOTICE: Removing illegal character "," from model name'))
    label = label.replace(",", "")

# Prepare the metadata for insertion
insert_model = {"time": int(time.time()), "label": label, "id": next_id, "data": []}

# Set up video_capture
video_capture = VideoCapture(config)

print(_("\nPlease look straight into the camera"))

# Give the user time to read
time.sleep(2)

# Will contain found face detections
detected_faces = []
# Count the number of read frames
frames = 0
# Count the number of illuminated read frames
valid_frames = 0
# Count the number of illuminated frames that
# were rejected for being too dark
dark_tries = 0
# Track the running darkness total
dark_running_total = 0

dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Loop through frames till we hit a timeout
while frames < 60:
    frames += 1
    # Grab a single frame of video
    frame, gsframe = video_capture.read_frame()
    gsframe = clahe.apply(gsframe)

    # Create a histogram of the image with 8 values
    hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
    # All values combined for percentage calculation
    hist_total = np.sum(hist)

    # Calculate frame darkness
    darkness = hist[0] / hist_total * 100

    # If the image is fully black due to a bad camera read,
    # skip to the next frame
    if (hist_total == 0) or (darkness == 100):
        continue

    # Include this frame in calculating our average session brightness
    dark_running_total += darkness
    valid_frames += 1

    # If the image exceeds darkness threshold due to subject distance,
    # skip to the next frame
    if darkness > dark_threshold:
        dark_tries += 1
        continue

    # Ensure frame is BGR for InsightFace
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Get all faces from that frame using InsightFace
    try:
        detected_faces = face_app.get(frame)
    except Exception as e:
        print(_("InsightFace error: ") + str(e), file=sys.stderr)
        detected_faces = []

    # If we've found at least one, we can continue
    if detected_faces:
        break

video_capture.release()

# If we've found no faces, try to determine why
if not detected_faces:
    if valid_frames == 0:
        print(_("Camera saw only black frames - is IR emitter working?"))
    elif valid_frames == dark_tries:
        print(_("All frames were too dark, please check dark_threshold in config"))
        print(
            _("Average darkness: {avg}, Threshold: {threshold}").format(
                avg=str(dark_running_total / valid_frames),
                threshold=str(dark_threshold),
            )
        )
    else:
        print(_("No face detected, aborting"))
    sys.exit(1)

# If more than 1 faces are detected we can't know which one belongs to the user
elif len(detected_faces) > 1:
    print(_("Multiple faces detected, aborting"))
    sys.exit(1)

# Get the normalized embedding from InsightFace result
face_encoding = detected_faces[0].normed_embedding.tolist()

# Validate embedding before saving
if not face_encoding or not isinstance(face_encoding, list) or len(face_encoding) == 0:
    print(_("Failed to extract a valid face embedding, aborting"))
    sys.exit(1)

insert_model["data"].append(face_encoding)

# Insert full object into the list
encodings.append(insert_model)

# Save the new encodings to disk
with open(enc_file, "w") as datafile:
    json.dump(encodings, datafile)

# Give let the user know how it went
print(
    _("""\nScan complete
Added a new model to """)
    + user
)
