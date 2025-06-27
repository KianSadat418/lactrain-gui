import json
import numpy as np
from datetime import datetime, timedelta

# === Parameters ===
duration_seconds = 10
fps = 10
num_frames = duration_seconds * fps
peg_count = 6

frames = []

start_time = datetime.now()

for i in range(num_frames):
    t = i / fps  # time in seconds

    # === Gaze origin: orbiting in circle ===
    origin = np.array([
        100 * np.cos(0.2 * t),
        100 * np.sin(0.2 * t),
        30 + 20 * np.sin(0.5 * t)
    ])

    # === Gaze target: moving forward in Z ===
    target = np.array([
        150 + 20 * np.sin(0.3 * t),
        20 * np.cos(0.3 * t),
        200 + 10 * np.sin(0.5 * t)
    ])

    gaze_line = [origin.tolist(), target.tolist()]
    gaze_distance = np.linalg.norm(target - origin)
    roi = 25 + 10 * np.sin(0.5 * t)
    intercept = int((np.sin(0.3 * t) > 0))  # toggle intercept

    # === Pegs: oscillating slightly ===
    peg_data = []
    for j in range(peg_count):
        x = 50 * j + 10 * np.sin(0.4 * t + j)
        y = 5 * np.sin(0.3 * t + j)
        z = j * 10 + 5 * np.cos(0.2 * t + j)
        peg_data.append([x, y, z])

    timestamp = (start_time + timedelta(seconds=t)).isoformat()

    frame = {
        "timestamp": timestamp,
        "gaze_data": {
            "gaze_line": gaze_line,
            "roi": roi,
            "intercept": intercept,
            "gaze_distance": gaze_distance
        },
        "peg_data": peg_data,
        "transform_id": 0
    }

    frames.append(frame)

# === Save to JSON ===
with open("test_recording_10s.json", "w") as f:
    json.dump(frames, f, indent=2)

print(f"Generated {num_frames} frames in test_recording_10s.json")