import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

import mediapipe as mp

import os
import urllib.request
import time
import math


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "AI Driver Drowsiness Detector"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "face_landmarker.task"
)

# MediaPipe Face Landmarker model
MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/"
    "face_landmarker/"
    "face_landmarker/"
    "float16/"
    "1/"
    "face_landmarker.task"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


# ============================================================
# MEDIAPIPE FACE LANDMARK INDEXES
# ============================================================

# These indexes correspond to the MediaPipe Face Mesh topology.

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

# Additional eye landmarks
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Mouth landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Face/head reference points
NOSE_TIP = 1
CHIN = 152

LEFT_FACE = 234
RIGHT_FACE = 454

FOREHEAD = 10


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_model():

    if os.path.exists(MODEL_PATH):

        return True

    try:

        print(
            "Downloading MediaPipe Face Landmarker..."
        )

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print(
            "Model downloaded:",
            MODEL_PATH
        )

        return True

    except Exception as e:

        print(
            "MODEL DOWNLOAD ERROR:",
            e
        )

        return False


# ============================================================
# DISTANCE
# ============================================================

def distance(p1, p2):

    if p1 is None or p2 is None:

        return 0.0

    return math.sqrt(
        (
            p1[0] - p2[0]
        ) ** 2
        +
        (
            p1[1] - p2[1]
        ) ** 2
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

class DriverDrowsinessDetector:

    def __init__(self, root):

        self.root = root

        self.root.title(
            APP_NAME
        )

        self.root.geometry(
            "1450x900"
        )

        self.root.minsize(
            1100,
            700
        )

        # ====================================================
        # VIDEO
        # ====================================================

        self.cap = None

        self.camera_mode = False

        self.video_mode = False

        self.paused = False

        self.video_path = None

        self.video_fps = 30

        self.total_frames = 0

        self.frame_number = 0

        self.timestamp_ms = 0

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        self.landmarker = None

        self.create_landmarker()

        # ====================================================
        # DROWSINESS PARAMETERS
        # ====================================================

        self.eye_threshold = 0.21

        self.yawn_threshold = 0.55

        self.closed_frames = 0

        self.yawn_frames = 0

        self.blink_count = 0

        self.yawn_count = 0

        self.last_eye_state = "OPEN"

        self.was_eye_closed = False

        self.alarm_active = False

        self.alarm_start_time = None

        # ====================================================
        # DROWSINESS SCORE
        # ====================================================

        self.drowsiness_score = 0

        self.current_status = "ALERT"

        # ====================================================
        # DATA
        # ====================================================

        self.analysis_data = []

        # ====================================================
        # DISPLAY SETTINGS
        # ====================================================

        self.show_landmarks = True

        self.show_metrics = True

        self.show_alert = True

        # ====================================================
        # GUI
        # ====================================================

        self.create_gui()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close
        )


    # ========================================================
    # MEDIAPIPE INITIALIZATION
    # ========================================================

    def create_landmarker(self):

        if not download_model():

            messagebox.showerror(
                "MediaPipe Model Error",
                (
                    "Could not download the MediaPipe "
                    "Face Landmarker model.\n\n"
                    "Place the model manually at:\n"
                    f"{MODEL_PATH}"
                )
            )

            return

        try:

            options = FaceLandmarkerOptions(

                base_options=BaseOptions(
                    model_asset_path=MODEL_PATH
                ),

                running_mode=RunningMode.VIDEO,

                num_faces=1,

                min_face_detection_confidence=0.5,

                min_face_presence_confidence=0.5,

                min_tracking_confidence=0.5,

                output_face_blendshapes=False,

                output_facial_transformation_matrixes=False
            )

            self.landmarker = (
                FaceLandmarker
                .create_from_options(
                    options
                )
            )

            print(
                "MediaPipe Face Landmarker READY"
            )

        except Exception as e:

            print(
                "MediaPipe initialization failed:",
                e
            )

            messagebox.showerror(
                "MediaPipe Error",
                str(e)
            )


    # ========================================================
    # GUI
    # ========================================================

    def create_gui(self):

        style = ttk.Style()

        try:

            style.theme_use(
                "clam"
            )

        except Exception:

            pass

        # ====================================================
        # MAIN
        # ====================================================

        main = ttk.Frame(
            self.root
        )

        main.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ====================================================
        # SIDEBAR
        # ====================================================

        sidebar = ttk.Frame(
            main,
            width=280
        )

        sidebar.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=10,
            pady=10
        )

        sidebar.pack_propagate(
            False
        )

        ttk.Label(
            sidebar,
            text="🚗 Driver AI",
            font=(
                "Segoe UI",
                23,
                "bold"
            )
        ).pack(
            pady=(10, 5)
        )

        ttk.Label(
            sidebar,
            text="Drowsiness • Eye • Yawn\nDetection",
            justify=tk.CENTER
        ).pack(
            pady=(0, 20)
        )

        # ====================================================
        # SOURCE BUTTONS
        # ====================================================

        ttk.Button(
            sidebar,
            text="🎥 Open Live Camera",
            command=self.open_camera
        ).pack(
            fill=tk.X,
            pady=4
        )

        ttk.Button(
            sidebar,
            text="📹 Load Recorded Video",
            command=self.load_video
        ).pack(
            fill=tk.X,
            pady=4
        )

        self.play_btn = ttk.Button(
            sidebar,
            text="▶ Play",
            command=self.toggle_pause
        )

        self.play_btn.pack(
            fill=tk.X,
            pady=4
        )

        ttk.Button(
            sidebar,
            text="⏹ Stop",
            command=self.stop
        ).pack(
            fill=tk.X,
            pady=4
        )

        ttk.Button(
            sidebar,
            text="🔄 Reset",
            command=self.reset
        ).pack(
            fill=tk.X,
            pady=4
        )

        ttk.Button(
            sidebar,
            text="📊 Export CSV",
            command=self.export_csv
        ).pack(
            fill=tk.X,
            pady=4
        )

        ttk.Separator(
            sidebar
        ).pack(
            fill=tk.X,
            pady=15
        )

        # ====================================================
        # EYE SENSITIVITY
        # ====================================================

        ttk.Label(
            sidebar,
            text="Eye Closure Sensitivity",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=tk.W
        )

        self.eye_slider = ttk.Scale(
            sidebar,
            from_=0.15,
            to=0.30,
            value=self.eye_threshold,
            command=self.change_eye_threshold
        )

        self.eye_slider.pack(
            fill=tk.X,
            pady=8
        )

        ttk.Label(
            sidebar,
            text="Lower = more sensitive"
        ).pack(
            anchor=tk.W
        )

        # ====================================================
        # YAWN SENSITIVITY
        # ====================================================

        ttk.Label(
            sidebar,
            text="Yawn Sensitivity",
            font=(
                "Segoe UI",
                10,
                "bold"
            )
        ).pack(
            anchor=tk.W,
            pady=(12, 0)
        )

        self.yawn_slider = ttk.Scale(
            sidebar,
            from_=0.35,
            to=0.80,
            value=self.yawn_threshold,
            command=self.change_yawn_threshold
        )

        self.yawn_slider.pack(
            fill=tk.X,
            pady=8
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        self.landmarks_var = tk.BooleanVar(
            value=True
        )

        self.metrics_var = tk.BooleanVar(
            value=True
        )

        self.alert_var = tk.BooleanVar(
            value=True
        )

        ttk.Checkbutton(
            sidebar,
            text="Show Face Landmarks",
            variable=self.landmarks_var
        ).pack(
            anchor=tk.W
        )

        ttk.Checkbutton(
            sidebar,
            text="Show Metrics",
            variable=self.metrics_var
        ).pack(
            anchor=tk.W
        )

        ttk.Checkbutton(
            sidebar,
            text="Show Alerts",
            variable=self.alert_var
        ).pack(
            anchor=tk.W
        )

        ttk.Separator(
            sidebar
        ).pack(
            fill=tk.X,
            pady=15
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status = ttk.Label(
            sidebar,
            text="Status: Ready",
            wraplength=250
        )

        self.status.pack(
            pady=5
        )

        ttk.Button(
            sidebar,
            text="❌ Quit",
            command=self.close
        ).pack(
            side=tk.BOTTOM,
            fill=tk.X,
            pady=10
        )

        # ====================================================
        # RIGHT SIDE
        # ====================================================

        right = ttk.Frame(
            main
        )

        right.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10
        )

        ttk.Label(
            right,
            text="AI Driver Drowsiness Monitoring",
            font=(
                "Segoe UI",
                18,
                "bold"
            )
        ).pack(
            pady=5
        )

        # ====================================================
        # VIDEO
        # ====================================================

        self.video_label = tk.Label(
            right,
            bg="#111111",
            fg="white",
            text=(
                "🚗\n\n"
                "Open the camera\n"
                "or load a driving video"
            ),
            font=(
                "Segoe UI",
                18
            )
        )

        self.video_label.pack(
            fill=tk.BOTH,
            expand=True
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        self.progress = ttk.Progressbar(
            right,
            orient=tk.HORIZONTAL,
            mode="determinate"
        )

        self.progress.pack(
            fill=tk.X,
            pady=5
        )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = ttk.Frame(
            right
        )

        metrics.pack(
            fill=tk.X
        )

        self.status_value = self.make_metric(
            metrics,
            "😴 Status",
            "ALERT"
        )

        self.score_value = self.make_metric(
            metrics,
            "🧠 Score",
            "0"
        )

        self.blink_value = self.make_metric(
            metrics,
            "👁 Blinks",
            "0"
        )

        self.yawn_value = self.make_metric(
            metrics,
            "🥱 Yawns",
            "0"
        )

        self.eye_value = self.make_metric(
            metrics,
            "👁 EAR",
            "0.00"
        )

        self.fps_value = self.make_metric(
            metrics,
            "⚡ FPS",
            "0"
        )

        # ====================================================
        # INFORMATION
        # ====================================================

        self.info_label = ttk.Label(
            right,
            text=(
                "Waiting for driver..."
            ),
            font=(
                "Segoe UI",
                11
            )
        )

        self.info_label.pack(
            pady=8
        )


    # ========================================================
    # METRIC CARD
    # ========================================================

    def make_metric(
        self,
        parent,
        title,
        value
    ):

        box = ttk.Frame(
            parent
        )

        box.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=3
        )

        ttk.Label(
            box,
            text=title
        ).pack()

        label = ttk.Label(
            box,
            text=value,
            font=(
                "Segoe UI",
                15,
                "bold"
            )
        )

        label.pack()

        return label


    # ========================================================
    # CAMERA
    # ========================================================

    def open_camera(self):

        self.stop()

        self.reset()

        self.cap = cv2.VideoCapture(
            0
        )

        if not self.cap.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Camera could not be opened."
            )

            return

        self.camera_mode = True

        self.video_mode = False

        self.paused = False

        self.frame_number = 0

        self.timestamp_ms = 0

        self.status.config(
            text="Status: Live camera"
        )

        self.play_btn.config(
            text="⏸ Pause"
        )

        self.process_frame()


    # ========================================================
    # LOAD VIDEO
    # ========================================================

    def load_video(self):

        path = filedialog.askopenfilename(

            title="Select Driving Video",

            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv"
                )
            ]
        )

        if not path:

            return

        self.stop()

        self.reset()

        self.video_path = path

        self.cap = cv2.VideoCapture(
            path
        )

        if not self.cap.isOpened():

            messagebox.showerror(
                "Video Error",
                "Could not open video."
            )

            return

        self.video_fps = (
            self.cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if self.video_fps <= 0:

            self.video_fps = 30

        self.total_frames = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        self.camera_mode = False

        self.video_mode = True

        self.paused = False

        self.frame_number = 0

        self.timestamp_ms = 0

        self.status.config(
            text="Status: Video loaded"
        )

        self.play_btn.config(
            text="⏸ Pause"
        )

        self.process_frame()


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self):

        if (
            self.cap is None
            or self.paused
            or (
                not self.camera_mode
                and not self.video_mode
            )
        ):

            return

        start = time.perf_counter()

        ret, frame = self.cap.read()

        if not ret:

            if self.video_mode:

                self.video_finished()

            return

        self.frame_number += 1

        # ====================================================
        # TIMESTAMP
        # ====================================================

        if self.camera_mode:

            # Use real elapsed processing time
            # converted to milliseconds.
            self.timestamp_ms += 33

        else:

            self.timestamp_ms = int(
                (
                    self.frame_number
                    /
                    self.video_fps
                )
                * 1000
            )

        # ====================================================
        # ANALYZE
        # ====================================================

        frame = self.analyze_frame(
            frame
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        self.display_frame(
            frame
        )

        # ====================================================
        # FPS
        # ====================================================

        processing_time = (
            time.perf_counter()
            - start
        )

        current_fps = (
            1.0 /
            processing_time
            if processing_time > 0
            else 0
        )

        self.update_dashboard(
            current_fps
        )

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            self.video_mode
            and self.total_frames > 0
        ):

            value = (
                self.frame_number
                /
                self.total_frames
                *
                100
            )

            self.progress["value"] = value

        # ====================================================
        # NEXT FRAME
        # ====================================================

        delay = max(
            1,
            int(
                1000 /
                max(
                    self.video_fps,
                    30
                )
            )
        )

        self.root.after(
            delay,
            self.process_frame
        )


    # ========================================================
    # ANALYZE FRAME
    # ========================================================

    def analyze_frame(
        self,
        frame
    ):

        if self.landmarker is None:

            return frame

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        try:

            result = (
                self.landmarker
                .detect_for_video(
                    mp_image,
                    self.timestamp_ms
                )
            )

        except Exception as e:

            print(
                "Detection error:",
                e
            )

            return frame

        # ====================================================
        # NO FACE
        # ====================================================

        if not result.face_landmarks:

            self.current_status = (
                "NO FACE"
            )

            cv2.putText(
                frame,
                "NO DRIVER FACE DETECTED",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            return frame

        # ====================================================
        # FIRST FACE
        # ====================================================

        landmarks = (
            result.face_landmarks[0]
        )

        # ====================================================
        # POINT HELPER
        # ====================================================

        def get_point(index):

            if (
                index < 0
                or
                index >= len(landmarks)
            ):

                return None

            lm = landmarks[index]

            return (
                int(lm.x * w),
                int(lm.y * h)
            )

        # ====================================================
        # EYE LANDMARKS
        # ====================================================

        left_top = get_point(
            LEFT_EYE_TOP
        )

        left_bottom = get_point(
            LEFT_EYE_BOTTOM
        )

        left_left = get_point(
            LEFT_EYE_LEFT
        )

        left_right = get_point(
            LEFT_EYE_RIGHT
        )

        right_top = get_point(
            RIGHT_EYE_TOP
        )

        right_bottom = get_point(
            RIGHT_EYE_BOTTOM
        )

        right_left = get_point(
            RIGHT_EYE_LEFT
        )

        right_right = get_point(
            RIGHT_EYE_RIGHT
        )

        # ====================================================
        # MOUTH
        # ====================================================

        mouth_left = get_point(
            MOUTH_LEFT
        )

        mouth_right = get_point(
            MOUTH_RIGHT
        )

        mouth_top = get_point(
            MOUTH_TOP
        )

        mouth_bottom = get_point(
            MOUTH_BOTTOM
        )

        # ====================================================
        # EAR
        # ====================================================

        left_ear = self.calculate_ear(
            left_top,
            left_bottom,
            left_left,
            left_right
        )

        right_ear = self.calculate_ear(
            right_top,
            right_bottom,
            right_left,
            right_right
        )

        ear = (
            left_ear
            +
            right_ear
        ) / 2.0

        # ====================================================
        # MOUTH OPENING
        # ====================================================

        mouth_width = distance(
            mouth_left,
            mouth_right
        )

        mouth_height = distance(
            mouth_top,
            mouth_bottom
        )

        if mouth_width > 0:

            mouth_ratio = (
                mouth_height
                /
                mouth_width
            )

        else:

            mouth_ratio = 0

        # ====================================================
        # EYE STATE
        # ====================================================

        eyes_closed = (
            ear <
            self.eye_threshold
        )

        # ====================================================
        # BLINK / EYE CLOSURE
        # ====================================================

        if eyes_closed:

            self.closed_frames += 1

        else:

            if (
                self.closed_frames >= 2
                and
                self.closed_frames < 12
            ):

                self.blink_count += 1

            self.closed_frames = 0

        # ====================================================
        # YAWN
        # ====================================================

        yawning = (
            mouth_ratio >
            self.yawn_threshold
        )

        if yawning:

            self.yawn_frames += 1

        else:

            if self.yawn_frames >= 8:

                self.yawn_count += 1

            self.yawn_frames = 0

        # ====================================================
        # DROWSINESS ANALYSIS
        # ====================================================

        status, score = (
            self.calculate_drowsiness(
                ear,
                eyes_closed,
                mouth_ratio,
                yawning
            )
        )

        self.current_status = status

        self.drowsiness_score = score

        # ====================================================
        # DRAW LANDMARKS
        # ====================================================

        if self.landmarks_var.get():

            self.draw_face_landmarks(
                frame,
                landmarks,
                w,
                h
            )

            self.draw_eye_points(
                frame,
                left_top,
                left_bottom,
                left_left,
                left_right
            )

            self.draw_eye_points(
                frame,
                right_top,
                right_bottom,
                right_left,
                right_right
            )

        # ====================================================
        # DRAW METRICS
        # ====================================================

        if self.metrics_var.get():

            self.draw_metrics(
                frame,
                ear,
                mouth_ratio,
                score
            )

        # ====================================================
        # ALERT
        # ====================================================

        if self.alert_var.get():

            self.draw_alert(
                frame,
                status
            )

        # ====================================================
        # SAVE ANALYSIS
        # ====================================================

        current_time = (
            self.frame_number
            /
            self.video_fps
        )

        self.analysis_data.append(
            {
                "frame":
                    self.frame_number,

                "time":
                    current_time,

                "EAR":
                    ear,

                "left_EAR":
                    left_ear,

                "right_EAR":
                    right_ear,

                "mouth_ratio":
                    mouth_ratio,

                "eyes_closed":
                    int(
                        eyes_closed
                    ),

                "yawning":
                    int(
                        yawning
                    ),

                "closed_frames":
                    self.closed_frames,

                "blink_count":
                    self.blink_count,

                "yawn_count":
                    self.yawn_count,

                "drowsiness_score":
                    score,

                "status":
                    status
            }
        )

        return frame


    # ========================================================
    # EAR CALCULATION
    # ========================================================

    def calculate_ear(
        self,
        top,
        bottom,
        left,
        right
    ):

        if (
            top is None
            or
            bottom is None
            or
            left is None
            or
            right is None
        ):

            return 0.0

        vertical = distance(
            top,
            bottom
        )

        horizontal = distance(
            left,
            right
        )

        if horizontal == 0:

            return 0.0

        return (
            vertical
            /
            horizontal
        )


    # ========================================================
    # DROWSINESS LOGIC
    # ========================================================

    def calculate_drowsiness(
        self,
        ear,
        eyes_closed,
        mouth_ratio,
        yawning
    ):

        score = 0

        # ====================================================
        # EYE CLOSURE
        # ====================================================

        if eyes_closed:

            if self.closed_frames >= 5:

                score += 25

            if self.closed_frames >= 10:

                score += 20

            if self.closed_frames >= 20:

                score += 25

            if self.closed_frames >= 30:

                score += 20

        # ====================================================
        # YAWNING
        # ====================================================

        if yawning:

            score += 15

            if self.yawn_frames >= 15:

                score += 10

        # ====================================================
        # LONG EYE CLOSURE
        # ====================================================

        if (
            self.closed_frames
            >= 20
        ):

            status = (
                "DROWSY"
            )

        elif (
            self.closed_frames
            >= 10
        ):

            status = (
                "WARNING"
            )

        elif (
            yawning
            and
            self.yawn_frames >= 15
        ):

            status = (
                "WARNING"
            )

        elif score >= 35:

            status = (
                "WARNING"
            )

        else:

            status = (
                "ALERT"
            )

        return status, min(
            score,
            100
        )


    # ========================================================
    # DRAW FACE LANDMARKS
    # ========================================================

    def draw_face_landmarks(
        self,
        frame,
        landmarks,
        width,
        height
    ):

        important = (

            LEFT_EYE
            +
            RIGHT_EYE
            +
            [
                MOUTH_LEFT,
                MOUTH_RIGHT,
                MOUTH_TOP,
                MOUTH_BOTTOM
            ]
        )

        for index in important:

            if index >= len(landmarks):

                continue

            lm = landmarks[index]

            x = int(
                lm.x * width
            )

            y = int(
                lm.y * height
            )

            cv2.circle(
                frame,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )


    # ========================================================
    # DRAW EYE POINTS
    # ========================================================

    def draw_eye_points(
        self,
        frame,
        top,
        bottom,
        left,
        right
    ):

        points = [
            top,
            bottom,
            left,
            right
        ]

        for point in points:

            if point is None:

                continue

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 255),
                -1
            )

        if (
            left is not None
            and
            right is not None
        ):

            cv2.line(
                frame,
                left,
                right,
                (255, 255, 0),
                2
            )

        if (
            top is not None
            and
            bottom is not None
        ):

            cv2.line(
                frame,
                top,
                bottom,
                (255, 0, 255),
                2
            )


    # ========================================================
    # DRAW METRICS
    # ========================================================

    def draw_metrics(
        self,
        frame,
        ear,
        mouth_ratio,
        score
    ):

        h, w = frame.shape[:2]

        cv2.rectangle(
            frame,
            (
                15,
                15
            ),
            (
                300,
                155
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            f"EAR: {ear:.3f}",
            (
                30,
                45
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Mouth: {mouth_ratio:.3f}",
            (
                30,
                75
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Score: {score}",
            (
                30,
                105
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Closed: {self.closed_frames}",
            (
                30,
                135
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


    # ========================================================
    # DRAW ALERT
    # ========================================================

    def draw_alert(
        self,
        frame,
        status
    ):

        h, w = frame.shape[:2]

        if status == "ALERT":

            text = (
                "DRIVER ALERT"
            )

            color = (
                0,
                255,
                0
            )

        elif status == "WARNING":

            text = (
                "DROWSINESS WARNING"
            )

            color = (
                0,
                165,
                255
            )

        elif status == "DROWSY":

            text = (
                "!!! DROWSY DRIVER !!!"
            )

            color = (
                0,
                0,
                255
            )

        else:

            text = (
                "FACE NOT DETECTED"
            )

            color = (
                0,
                0,
                255
            )

        cv2.rectangle(
            frame,
            (
                15,
                h - 75
            ),
            (
                min(
                    w - 15,
                    420
                ),
                h - 20
            ),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            text,
            (
                30,
                h - 40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            3
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    def update_dashboard(
        self,
        fps
    ):

        self.status_value.config(
            text=self.current_status
        )

        self.score_value.config(
            text=str(
                self.drowsiness_score
            )
        )

        self.blink_value.config(
            text=str(
                self.blink_count
            )
        )

        self.yawn_value.config(
            text=str(
                self.yawn_count
            )
        )

        self.fps_value.config(
            text=f"{fps:.1f}"
        )

        # EAR from latest data
        if self.analysis_data:

            latest = (
                self.analysis_data[-1]
            )

            self.eye_value.config(
                text=f"{latest['EAR']:.3f}"
            )

        self.info_label.config(
            text=(
                f"Status: {self.current_status}   |   "
                f"Blinks: {self.blink_count}   |   "
                f"Yawns: {self.yawn_count}   |   "
                f"Drowsiness Score: "
                f"{self.drowsiness_score}"
            )
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    def display_frame(
        self,
        frame
    ):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(
            rgb
        )

        width = (
            self.video_label.winfo_width()
        )

        height = (
            self.video_label.winfo_height()
        )

        if width < 100:

            width = 1000

        if height < 100:

            height = 600

        image.thumbnail(
            (
                width,
                height
            ),
            Image.Resampling.LANCZOS
        )

        photo = ImageTk.PhotoImage(
            image
        )

        self.video_label.configure(
            image=photo,
            text=""
        )

        self.video_label.image = photo


    # ========================================================
    # PAUSE
    # ========================================================

    def toggle_pause(self):

        if (
            not self.video_mode
            and
            not self.camera_mode
        ):

            return

        self.paused = (
            not self.paused
        )

        if self.paused:

            self.play_btn.config(
                text="▶ Play"
            )

            self.status.config(
                text="Status: Paused"
            )

        else:

            self.play_btn.config(
                text="⏸ Pause"
            )

            self.status.config(
                text="Status: Processing"
            )

            self.process_frame()


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.video_mode = False

        self.camera_mode = False

        self.paused = False

        if self.cap:

            self.cap.release()

            self.cap = None

        self.play_btn.config(
            text="▶ Play"
        )

        self.status.config(
            text="Status: Stopped"
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.frame_number = 0

        self.timestamp_ms = 0

        self.closed_frames = 0

        self.yawn_frames = 0

        self.blink_count = 0

        self.yawn_count = 0

        self.drowsiness_score = 0

        self.current_status = (
            "ALERT"
        )

        self.analysis_data = []

        self.progress["value"] = 0

        self.update_dashboard(
            0
        )


    # ========================================================
    # THRESHOLD SETTINGS
    # ========================================================

    def change_eye_threshold(
        self,
        value
    ):

        self.eye_threshold = float(
            value
        )


    def change_yawn_threshold(
        self,
        value
    ):

        self.yawn_threshold = float(
            value
        )


    # ========================================================
    # VIDEO FINISHED
    # ========================================================

    def video_finished(self):

        self.video_mode = False

        if self.cap:

            self.cap.release()

            self.cap = None

        self.progress["value"] = 100

        self.play_btn.config(
            text="▶ Play"
        )

        self.status.config(
            text="Status: Analysis complete"
        )

        messagebox.showinfo(
            "Analysis Complete",
            (
                f"Blink Count: "
                f"{self.blink_count}\n\n"

                f"Yawns: "
                f"{self.yawn_count}\n\n"

                f"Final Drowsiness Score: "
                f"{self.drowsiness_score}\n\n"

                f"Final Status: "
                f"{self.current_status}"
            )
        )


    # ========================================================
    # EXPORT CSV
    # ========================================================

    def export_csv(self):

        if not self.analysis_data:

            messagebox.showwarning(
                "No Analysis",
                "Analyze a video first."
            )

            return

        path = filedialog.asksaveasfilename(

            defaultextension=".csv",

            filetypes=[
                (
                    "CSV",
                    "*.csv"
                )
            ]
        )

        if not path:

            return

        df = pd.DataFrame(
            self.analysis_data
        )

        df.to_csv(
            path,
            index=False
        )

        messagebox.showinfo(
            "Export Complete",
            f"Saved:\n{path}"
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.stop()

        try:

            if self.landmarker:

                self.landmarker.close()

        except Exception:

            pass

        self.root.destroy()


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = DriverDrowsinessDetector(
        root
    )

    root.mainloop()