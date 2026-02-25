import time
import cv2

from i18n import _

# Import the root rubberstamp class
from rubberstamps import RubberStamp

# Import MediaPipe for facial landmark detection
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


class nod(RubberStamp):
    def declare_config(self):
        """Set the default values for the optional arguments"""
        self.options["min_distance"] = 6
        self.options["min_directions"] = 2

    def run(self):
        """Track a users nose to see if they nod yes or no using MediaPipe"""
        self.set_ui_text(_("Nod to confirm"), self.UI_TEXT)
        self.set_ui_text(_("Shake your head to abort"), self.UI_SUBTEXT)

        # Stores relative distance between the 2 eyes in the last frame
        # Used to calculate the distance of the nose traveled in relation to face size in the frame
        last_reldist = -1
        # Last point the nose was at
        last_nosepoint = {"x": -1, "y": -1}
        # Contains booleans recording successful nods and their directions
        recorded_nods = {"x": [], "y": []}

        starttime = time.time()

        # Use MediaPipe Face Mesh for landmark detection
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:
            # Keep running the loop while we have not hit timeout yet
            while time.time() < starttime + self.options["timeout"]:
                # Read a frame from the camera
                color_frame, gsframe = self.video_capture.read_frame()

                # Apply CLAHE to the grayscale frame (kept for consistency)
                gsframe = self.clahe.apply(gsframe)

                # MediaPipe requires RGB input — use color frame for best results
                frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)

                # Only continue if exactly 1 face is visible in the frame
                if (
                    not results.multi_face_landmarks
                    or len(results.multi_face_landmarks) != 1
                ):
                    continue

                # Get facial landmarks
                landmarks = results.multi_face_landmarks[0]
                h, w = color_frame.shape[:2]

                # MediaPipe landmark indices:
                #   Left eye outer corner:  33
                #   Right eye outer corner: 263
                #   Nose tip:               1
                left_eye = landmarks.landmark[33]
                right_eye = landmarks.landmark[263]
                nose_tip = landmarks.landmark[1]

                # Convert normalized coordinates to pixel coordinates
                left_eye_x = int(left_eye.x * w)
                right_eye_x = int(right_eye.x * w)
                nose_x = int(nose_tip.x * w)
                nose_y = int(nose_tip.y * h)

                # Calculate the relative distance between the 2 eyes
                reldist = abs(left_eye_x - right_eye_x)

                if last_reldist < 0:
                    last_reldist = reldist
                    last_nosepoint["x"] = nose_x
                    last_nosepoint["y"] = nose_y
                    continue

                # Average this out with the distance found in the last frame to smooth it out
                avg_reldist = (last_reldist + reldist) / 2

                # Calculate horizontal movement (shaking head) and vertical movement (nodding)
                for axis in ["x", "y"]:
                    # Get the location of the nose on the active axis
                    nosepoint = nose_x if axis == "x" else nose_y

                    mindist = self.options["min_distance"]
                    # Get the relative movement by taking the distance traveled and dividing it by eye distance
                    movement = (
                        (nosepoint - last_nosepoint[axis]) * 100 / max(avg_reldist, 1)
                    )

                    # If the movement is over the minimal distance threshold
                    if movement < -mindist or movement > mindist:
                        # If this is the first recorded nod, add it to the array
                        if len(recorded_nods[axis]) == 0:
                            recorded_nods[axis].append(movement < 0)

                        # Otherwise, only add this nod if the previous nod with in the other direction
                        elif recorded_nods[axis][-1] != (movement < 0):
                            recorded_nods[axis].append(movement < 0)

                    # Check if we have nodded enough on this axis
                    if len(recorded_nods[axis]) >= self.options["min_directions"]:
                        # If nodded yes, show confirmation in ui
                        if axis == "y":
                            self.set_ui_text(
                                _("Confirmed authentication"), self.UI_TEXT
                            )
                        # If shaken no, show abort message
                        else:
                            self.set_ui_text(_("Aborted authentication"), self.UI_TEXT)

                        # Remove subtext
                        self.set_ui_text("", self.UI_SUBTEXT)

                        # Return true for nodding yes and false for shaking no
                        time.sleep(0.8)
                        return axis == "y"

                    # Save the relative distance and the nosepoint for next loop
                    last_reldist = reldist
                    last_nosepoint[axis] = nosepoint

        # We've fallen out of the loop, so timeout has been hit
        return not self.options["failsafe"]
