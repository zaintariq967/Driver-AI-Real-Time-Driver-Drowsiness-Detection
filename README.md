# Driver AI – Real-Time Driver Drowsiness Detection

A desktop application that monitors a driver's face in real time (live webcam or recorded video) and detects **drowsiness, eye closure, blinking and yawning** using computer vision. When fatigue is detected, the app raises an on-screen alert to help prevent accidents.

> Built with Python, OpenCV and facial landmark detection.

---

## Features

- **Live camera & recorded video** – analyse a webcam feed or load a video file
- **Eye Aspect Ratio (EAR)** – detects eye closure and counts blinks
- **Mouth Aspect Ratio (MAR)** – detects yawning
- **Drowsiness score** – combines eye closure, blinks and yawns into one score
- **Status alerts** – shows `DRIVER ALERT` / drowsy warnings on the video
- **Adjustable sensitivity** – sliders for eye closure and yawn thresholds
- **Face landmark overlay** – toggle landmarks, metrics and alerts on/off
- **Live stats bar** – Status, Score, Blinks, Yawns, EAR and FPS
- **Playback controls** – Pause, Stop, Reset
- **Export CSV report** – save the session results for later analysis
- **Analysis summary** – popup with blink count, yawns, final score and status

---

## Tech Stack

| Area | Tools |
|------|-------|
| Language | Python 3 |
| Computer Vision | OpenCV |
| Face Landmarks | MediaPipe Face Mesh *(update if you used dlib)* |
| GUI | Tkinter *(update if you used another toolkit)* |
| Data / Reports | NumPy, CSV |

---

## How It Works

1. Capture frames from the webcam or a video file.
2. Detect the face and extract facial landmarks.
3. Compute **EAR** from the eye landmarks and **MAR** from the mouth landmarks.
4. If EAR stays below the threshold for several frames → eyes closed / drowsy.
5. If MAR goes above the threshold → yawn counted.
6. Update the drowsiness score and show `ALERT` or a drowsiness warning.

```
EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)
```

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/zaintariq967/Driver-AI-Real-Time-Driver-Drowsiness-Detection
cd Driver-AI-Real-Time-Driver-Drowsiness-Detection

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py               # replace with your main file name
```

---

## Usage

1. Click **Open Live Camera** or **Load Recorded Video**.
2. Adjust **Eye Closure** and **Yawn** sensitivity if needed.
3. Watch the live EAR, mouth ratio, score and status.
4. Click **Export CSV** to save the report.

---

## Project Structure

```
driver-drowsiness-detection/
├── app.py              # App entry point (GUI)
├── requirements.txt     # Dependencies
├── README.md
└── models/
      └──face_landmarker.task

```

*(Adjust to match your actual files.)*

---

## Future Improvements

- Audio alarm when drowsiness is detected
- Head-pose / distraction detection
- Better performance in low light
- Embedded deployment (Raspberry Pi / in-car)

---

## Author

**Zain Tariq**
GitHub: [@zaintariq](https://github.com/zaintariq967)

---

## 📄 License

This project is licensed under the MIT License.
