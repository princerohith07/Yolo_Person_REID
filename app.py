import gradio as gr
import cv2
import numpy as np
import json
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model for person detection
yolo_model = YOLO("yolov8n.pt")  

# Initialize DeepSort tracker
tracker = DeepSort(max_age=50)

# File to store tracking results
tracking_data_file = "tracking_data.json"

# Dictionary to store flagged IDs and their colors
flagged_ids = {}

# Function to process video and store tracking results
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracking_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons using YOLO
        results = yolo_model(frame)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Only consider persons (class 0) with confidence > 0.4
                if cls == 0 and conf > 0.4:
                    detections.append(([x1, y1, x2, y2], conf, None))

        # Update tracks using DeepSort
        tracks = tracker.update_tracks(detections, frame=frame)

        frame_data = []
        for track in tracks:
            if track.is_confirmed() and track.time_since_update == 0:
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                track_id = track.track_id
                frame_data.append({
                    "frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2]
                })

        tracking_data.append(frame_data)

    cap.release()

    # Save tracking data to JSON file
    with open(tracking_data_file, "w") as f:
        json.dump(tracking_data, f)

    return "Tracking data saved! Now reprocess the video to apply flags."

# Function to overlay bounding boxes with flagging
def reprocess_video(video_path, suspicious_id, clean_id):
    global flagged_ids

    # Ensure tracking data file exists
    if not os.path.exists(tracking_data_file):
        return "Error: Tracking data not found. Please process the video first."

    # Update flagged IDs
    if suspicious_id:
        flagged_ids[str(suspicious_id)] = (0, 0, 255)  # Red for suspicious
    if clean_id:
        flagged_ids[str(clean_id)] = (0, 255, 0)  # Green for clean

    # Load tracking data
    with open(tracking_data_file, "r") as f:
        tracking_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index < len(tracking_data) and tracking_data[frame_index]:
            for obj in tracking_data[frame_index]:
                x1, y1, x2, y2 = obj["bbox"]
                track_id = obj["id"]

                color = flagged_ids.get(str(track_id), (0, 255, 0))  # Default green if not flagged

                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    return output_path

# Gradio UI
def gradio_interface(video, process_first, suspicious_id, clean_id):
    if process_first:
        return process_video(video)
    return reprocess_video(video, suspicious_id, clean_id)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Checkbox(label="Process Video First"),
        gr.Number(label="Suspicious ID (Mark as Red)", value=None),
        gr.Number(label="Clean ID (Mark as Green)", value=None)

    ],
    outputs=gr.Video(label="Processed Video"),
    title="Person Tracking and Flagging with YOLO and DeepSort",
    description="Upload a video to track persons. First, process the video, then reprocess it to mark specific IDs without re-running detection."
)

# Launch the app
iface.launch()
