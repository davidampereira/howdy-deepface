# Compare incoming video with known faces
# Running in a local python instance to get around PATH issues

# Import time so we can start timing asap
import time

# Start timing
timings = {
	"st": time.time()
}

# Import required modules
import sys
import os
import json
import configparser
import cv2
from deepface import DeepFace
from datetime import timezone, datetime
import atexit
import subprocess
import snapshot
import numpy as np
import _thread as thread
import paths_factory
from recorders.video_capture import VideoCapture
from i18n import _

def exit(code=None):
	"""Exit while closing howdy-gtk properly"""
	global gtk_proc

	# Exit the auth ui process if there is one
	if "gtk_proc" in globals():
		gtk_proc.terminate()

	# Exit compare
	if code is not None:
		sys.exit(code)


def init_detector(lock):
	"""Pre-warm DeepFace models by loading them into memory"""
	global deepface_model_name, deepface_detector_name

	try:
		DeepFace.build_model(deepface_model_name)
	except Exception as e:
		print(_("Error loading DeepFace model: ") + str(e))
		lock.release()
		exit(1)

	# Note the time it took to initialize the model
	timings["ll"] = time.time() - timings["ll"]
	lock.release()


def make_snapshot(type):
	"""Generate snapshot after detection"""
	snapshot.generate(snapframes, [
		type + _(" LOGIN"),
		_("Date: ") + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
		_("Scan time: ") + str(round(time.time() - timings["fr"], 2)) + "s",
		_("Frames: ") + str(frames) + " (" + str(round(frames / (time.time() - timings["fr"]), 2)) + "FPS)",
		_("Hostname: ") + os.uname().nodename,
		_("Best certainty value: ") + str(round(lowest_certainty, 3))
	])


def send_to_ui(type, message):
	"""Send message to the auth ui"""
	global gtk_proc

	# Only execute of the process started
	if "gtk_proc" in globals():
		# Format message so the ui can parse it
		message = type + "=" + message + " \n"

		# Try to send the message to the auth ui, but it's okay if that fails
		try:
			if gtk_proc.poll() is None: # Make sure the gtk_proc is still running before write into the pipe
				gtk_proc.stdin.write(bytearray(message.encode("utf-8")))
				gtk_proc.stdin.flush()
		except IOError:
			pass


# Make sure we were given an username to test against
if len(sys.argv) < 2:
	exit(12)

# The username of the user being authenticated
user = sys.argv[1]
# The model file contents
models = []
# Encoded face models
encodings = []
# Amount of ignored 100% black frames
black_tries = 0
# Amount of ignored dark frames
dark_tries = 0
# Total amount of frames captured
frames = 0
# Captured frames for snapshot capture
snapframes = []
# Tracks the lowest certainty value in the loop
lowest_certainty = 10
# DeepFace model and detector names
deepface_model_name = None
deepface_detector_name = None

# Try to load the face model from the models folder
try:
	models = json.load(open(paths_factory.user_model_path(user)))

	for model in models:
		encodings += model["data"]
except FileNotFoundError:
	exit(10)

# Check if the file contains a model
if len(models) < 1:
	exit(10)

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

# Get all config values needed
deepface_model_name = config.get("core", "recognition_model", fallback="ArcFace")
deepface_detector_name = config.get("core", "detector_backend", fallback="retinaface")
deepface_distance_metric = config.get("core", "distance_metric", fallback="cosine")
timeout = config.getint("video", "timeout", fallback=4)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=50.0)
end_report = config.getboolean("debug", "end_report", fallback=False)
save_failed = config.getboolean("snapshots", "save_failed", fallback=False)
save_successful = config.getboolean("snapshots", "save_successful", fallback=False)
gtk_stdout = config.getboolean("debug", "gtk_stdout", fallback=False)
rotate = config.getint("video", "rotate", fallback=0)

# Get certainty threshold — "auto" means use DeepFace's built-in threshold
certainty_raw = config.get("video", "certainty", fallback="auto")
if certainty_raw.strip().lower() == "auto":
	# Use DeepFace's built-in threshold for the chosen model + metric
	from deepface.modules.verification import find_threshold
	video_certainty = find_threshold(deepface_model_name, deepface_distance_metric)
else:
	video_certainty = float(certainty_raw)

# Send the gtk output to the terminal if enabled in the config
gtk_pipe = sys.stdout if gtk_stdout else subprocess.DEVNULL

# Start the auth ui, register it to be always be closed on exit
try:
	gtk_proc = subprocess.Popen(["howdy-gtk", "--start-auth-ui"], stdin=subprocess.PIPE, stdout=gtk_pipe, stderr=gtk_pipe)
	atexit.register(exit)
except FileNotFoundError:
	pass

# Write to the stdin to redraw ui
send_to_ui("M", _("Starting up..."))

# Save the time needed to start the script
timings["in"] = time.time() - timings["st"]

# Import face recognition, takes some time
timings["ll"] = time.time()

# Start threading and wait for init to finish
lock = thread.allocate_lock()
lock.acquire()
thread.start_new_thread(init_detector, (lock, ))

# Start video capture on the IR camera
timings["ic"] = time.time()

video_capture = VideoCapture(config)

# Read exposure from config to use in the main loop
exposure = config.getint("video", "exposure", fallback=-1)

# Note the time it took to open the camera
timings["ic"] = time.time() - timings["ic"]

# wait for thread to finish
lock.acquire()
lock.release()
del lock

# Fetch the max frame height
max_height = config.getfloat("video", "max_height", fallback=320.0)

# Get the height of the image (which would be the width if screen is portrait oriented)
height = video_capture.internal.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
if rotate == 2:
	height = video_capture.internal.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
# Calculate the amount the image has to shrink
scaling_factor = (max_height / height) or 1

# Fetch config settings out of the loop
timeout = config.getint("video", "timeout", fallback=4)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)
end_report = config.getboolean("debug", "end_report", fallback=False)

# Initiate histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Let the ui know that we're ready
send_to_ui("M", _("Identifying you..."))

# Precompute numpy arrays of stored encodings for fast comparison
encodings_np = np.array(encodings)

# Start the read loop
frames = 0
valid_frames = 0
timings["fr"] = time.time()
dark_running_total = 0

while True:
	# Increment the frame count every loop
	frames += 1

	# Form a string to let the user know we're real busy
	ui_subtext = "Scanned " + str(valid_frames - dark_tries) + " frames"
	if (dark_tries > 1):
		ui_subtext += " (skipped " + str(dark_tries) + " dark frames)"
	# Show it in the ui as subtext
	send_to_ui("S", ui_subtext)

	# Stop if we've exceeded the time limit
	if time.time() - timings["fr"] > timeout:
		# Create a timeout snapshot if enabled
		if save_failed:
			make_snapshot(_("FAILED"))

		if dark_tries == valid_frames:
			print(_("All frames were too dark, please check dark_threshold in config"))
			print(_("Average darkness: {avg}, Threshold: {threshold}").format(avg=str(dark_running_total / max(1, valid_frames)), threshold=str(dark_threshold)))
			exit(13)
		else:
			exit(11)

	# Grab a single frame of video
	frame, gsframe = video_capture.read_frame()
	gsframe = clahe.apply(gsframe)

	# If snapshots have been turned on
	if save_failed or save_successful:
		# Start capturing frames for the snapshot
		if len(snapframes) < 3:
			snapframes.append(frame)

	# Create a histogram of the image with 8 values
	hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
	# All values combined for percentage calculation
	hist_total = np.sum(hist)

	# Calculate frame darkness
	darkness = (hist[0] / hist_total * 100)

	# If the image is fully black due to a bad camera read,
	# skip to the next frame
	if (hist_total == 0) or (darkness == 100):
		black_tries += 1
		continue

	dark_running_total += darkness
	valid_frames += 1

	# If the image exceeds darkness threshold due to subject distance,
	# skip to the next frame
	if (darkness > dark_threshold):
		dark_tries += 1
		continue

	# If the height is too high
	if scaling_factor != 1:
		# Apply that factor to the frame
		frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
		gsframe = cv2.resize(gsframe, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

	# If camera is configured to rotate = 1, check portrait in addition to landscape
	if rotate == 1:
		if frames % 3 == 1:
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
		if frames % 3 == 2:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)

	# If camera is configured to rotate = 2, check portrait orientation
	elif rotate == 2:
		if frames % 2 == 0:
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
		else:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)

	# Get face embeddings from the frame using DeepFace
	try:
		results = DeepFace.represent(
			img_path=frame,
			model_name=deepface_model_name,
			detector_backend=deepface_detector_name,
			enforce_detection=False,
			align=True,
		)
	except Exception:
		# Skip frame on any DeepFace error
		continue

	# Loop through each detected face
	for result in results:
		face_encoding = np.array(result["embedding"])

		# Compute distance between this face and all stored encodings
		if deepface_distance_metric == "cosine":
			# Cosine distance = 1 - cosine_similarity
			face_norm = np.linalg.norm(face_encoding)
			enc_norms = np.linalg.norm(encodings_np, axis=1)
			cosine_similarities = np.dot(encodings_np, face_encoding) / (enc_norms * face_norm + 1e-10)
			distances = 1 - cosine_similarities
		elif deepface_distance_metric == "euclidean":
			distances = np.linalg.norm(encodings_np - face_encoding, axis=1)
		elif deepface_distance_metric == "euclidean_l2":
			# L2-normalize then compute euclidean distance
			face_norm_vec = face_encoding / (np.linalg.norm(face_encoding) + 1e-10)
			enc_norm_vecs = encodings_np / (np.linalg.norm(encodings_np, axis=1, keepdims=True) + 1e-10)
			distances = np.linalg.norm(enc_norm_vecs - face_norm_vec, axis=1)
		else:
			distances = np.linalg.norm(encodings_np - face_encoding, axis=1)

		# Get best match
		match_index = np.argmin(distances)
		match = distances[match_index]

		# Update certainty if we have a new low
		if lowest_certainty > match:
			lowest_certainty = match

		# Check if a match that's confident enough
		if 0 < match < video_certainty:
			timings["tt"] = time.time() - timings["st"]
			timings["fl"] = time.time() - timings["fr"]

			# If set to true in the config, print debug text
			if end_report:
				def print_timing(label, k):
					"""Helper function to print a timing from the list"""
					print("  %s: %dms" % (label, round(timings[k] * 1000)))

				# Print a nice timing report
				print(_("Time spent"))
				print_timing(_("Starting up"), "in")
				print(_("  Open cam + load libs: %dms") % (round(max(timings["ll"], timings["ic"]) * 1000, )))
				print_timing(_("  Opening the camera"), "ic")
				print_timing(_("  Importing recognition libs"), "ll")
				print_timing(_("Searching for known face"), "fl")
				print_timing(_("Total time"), "tt")

				print(_("\nResolution"))
				width = video_capture.fw or 1
				print(_("  Native: %dx%d") % (height, width))
				# Save the new size for diagnostics
				scale_height, scale_width = frame.shape[:2]
				print(_("  Used: %dx%d") % (scale_height, scale_width))

				# Show the total number of frames and calculate the FPS by dividing it by the total scan time
				print(_("\nFrames searched: %d (%.2f fps)") % (frames, frames / timings["fl"]))
				print(_("Black frames ignored: %d ") % (black_tries, ))
				print(_("Dark frames ignored: %d ") % (dark_tries, ))
				print(_("Certainty of winning frame: %.3f") % (match, ))

				print(_("Winning model: %d (\"%s\")") % (match_index, models[match_index]["label"]))

			# Make snapshot if enabled
			if save_successful:
				make_snapshot(_("SUCCESSFUL"))

			# Run rubberstamps if enabled
			if config.getboolean("rubberstamps", "enabled", fallback=False):
				import rubberstamps

				send_to_ui("S", "")

				if "gtk_proc" not in vars():
					gtk_proc = None

				rubberstamps.execute(config, gtk_proc, {
					"video_capture": video_capture,
					"clahe": clahe
				})

			# End peacefully
			exit(0)

	if exposure != -1:
		# For a strange reason on some cameras (e.g. Lenoxo X1E) setting manual exposure works only after a couple frames
		# are captured and even after a delay it does not always work. Setting exposure at every frame is reliable though.
		video_capture.internal.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # 1 = Manual
		video_capture.internal.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
