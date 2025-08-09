# main_accident_detector.py (or whatever your main file is named)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
import time
from collections import deque

# --- Import the new alarm manager ---
import alarm_manager 

print("--- Step 12: Statistical Window-Based Accident Confirmation for CCTV Monitoring ---", flush=True)
# Current time is Monday, July 7, 2025 at 12:06:16 PM IST.
print(f"Current System Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}", flush=True)

# --- Configuration ---
# YOLO Model
YOLO_MODEL_NAME = 'yolov8n.pt'
YOLO_MODEL_PATH = YOLO_MODEL_NAME

# Keras Accident Classifier Model
KERAS_MODEL_PATH = r'C:\Users\reddy\OneDrive\Desktop\project_gemini\fine_tuned_accident_detector_model.h5' # <--- IMPORTANT: Verify this path and filename
KERAS_IMG_HEIGHT, KERAS_IMG_WIDTH = 224, 224 # Must match your Keras model's input size

# --- Video Source Configuration ---
# For real-time CCTV camera:
VIDEO_SOURCE = r"A:/ACCIDENT_DETECTION/videos/0IPAgFHyRVI.mp4" # Use 0 for default webcam, 1 for external USB cam, etc.
# For IP cameras (RTSP stream):
# VIDEO_SOURCE = "rtsp://username:password@ip_address:port/path_to_stream" 
# Example (replace with your actual camera URL):
# VIDEO_SOURCE = "rtsp://admin:admin@192.168.1.100:554/stream1" 
# For a video file (for testing):
# VIDEO_SOURCE = r"A:/ACCIDENT_DETECTION/videos/0IPAgFHyRVI.mp4" 

OUTPUT_CLIPS_DIR = 'accident_clips'

# --- Accident Detection Logic Parameters (Per-Frame Potential) ---
RELEVANT_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person']
OVERLAP_THRESHOLD = 0.2 # Minimum IOU for two boxes to be considered "overlapping"
# Keras Classifier Confidence Threshold
KERAS_ACCIDENT_THRESHOLD = 0.55 # Probability from Keras model >= this will indicate a potential accident

# --- Statistical Confirmation Window Parameters ---
CONFIRMATION_WINDOW_SIZE_FRAMES = 50 # Number of frames in the sliding window to analyze
MIN_ACCIDENT_FRAMES_IN_WINDOW = 20 # Minimum number of potential accident frames within the window to confirm an accident (e.g., 130-140 for 200 frames)

# --- Clip Saving Parameters ---
FRAMES_BEFORE_ACCIDENT_CLIP = 30 # Frames to include BEFORE the confirmation point in the saved clip (e.g., 1-2 seconds)
FRAMES_AFTER_ACCIDENT_CLIP = 90 # Frames to include AFTER the confirmation point in the saved clip (e.g., 3-5 seconds)

# --- Cooldown Period for Debouncing Multiple Detections of Same Accident ---
ACCIDENT_COOLDOWN_FRAMES = 150 # Adjust as needed (e.g., 3-5 seconds of footage at 30 FPS)

# --- Specific Email for Alerts (Your chosen recipient for testing/monitoring) ---
# UPDATED EMAIL ADDRESS HERE:
ALERT_RECIPIENT_EMAIL = '22951a6614@iare.ac.in' 

# --- Load Models ---
print(f"\nLoading YOLOv8 model: {YOLO_MODEL_PATH}", flush=True)
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLOv8 model loaded successfully!", flush=True)
except Exception as e:
    print(f"Error loading YOLO model: {e}", flush=True)
    print("Please ensure 'ultralytics' is installed and model name is correct.", flush=True)
    exit()

print(f"\nLoading Keras Accident Classifier model from: {KERAS_MODEL_PATH}", flush=True)
try:
    keras_model = load_model(KERAS_MODEL_PATH)
    print("Keras model loaded successfully!", flush=True)
except Exception as e:
    print(f"Error loading Keras model: {e}", flush=True)
    print("Please ensure the Keras model path is correct and it's a valid .h5 file.", flush=True)
    exit()

# --- Create Output Directory ---
os.makedirs(OUTPUT_CLIPS_DIR, exist_ok=True)
print(f"Output directory for clips: {OUTPUT_CLIPS_DIR}", flush=True)

# --- Open Video Capture ---
print(f"\nOpening video source: {VIDEO_SOURCE}", flush=True)
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}", flush=True)
    print("If using a webcam, ensure it's not in use by another application.", flush=True)
    print("If using a video file, ensure the path is correct and the file exists.", flush=True)
    exit()

# Get video properties for VideoWriter
FPS = cap.get(cv2.CAP_PROP_FPS)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FOURCC = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files

print(f"\nStarting Hybrid Accident Detection. Press 'q' to quit.", flush=True)

frame_count = 0
start_time = time.time()

# --- Buffers for Statistical Window and Video Writing ---
accident_potential_status_window = deque(maxlen=CONFIRMATION_WINDOW_SIZE_FRAMES)
frame_buffer = deque(maxlen=CONFIRMATION_WINDOW_SIZE_FRAMES + FRAMES_BEFORE_ACCIDENT_CLIP + FRAMES_AFTER_ACCIDENT_CLIP + 10)

# --- System State Variables ---
current_system_state = "NORMAL"
cooldown_frame_counter = 0
out = None
clip_frames_written_after_confirmation = 0

# Function to calculate Intersection over Union (IOU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame. Stopping detection.", flush=True)
        break

    frame_count += 1
    display_frame = frame.copy()

    # --- 1. YOLO Object Detection ---
    yolo_results = yolo_model.predict(display_frame, verbose=False)
    current_frame_objects = []

    for r in yolo_results:
        boxes = r.boxes
        names = r.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names[class_id]

            if class_name in RELEVANT_CLASSES:
                current_frame_objects.append((x1, y1, x2, y2, class_name, confidence))

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # --- 2. Keras Model Prediction (Whole Frame Classification) ---
    keras_frame = cv2.resize(frame, (KERAS_IMG_WIDTH, KERAS_IMG_HEIGHT))
    keras_frame = cv2.cvtColor(keras_frame, cv2.COLOR_BGR2RGB)
    keras_frame = keras_frame.astype('float32') / 255.0
    keras_frame_input = np.expand_dims(keras_frame, axis=0)

    keras_prediction_prob = keras_model.predict(keras_frame_input, verbose=0)[0][0]

    # --- 3. Per-Frame Potential Accident Status ('AND' logic) ---
    yolo_based_accident_flag = False
    if len(current_frame_objects) >= 2:
        for i in range(len(current_frame_objects)):
            obj1_box = current_frame_objects[i][:4]
            obj1_class = current_frame_objects[i][4]

            for j in range(i + 1, len(current_frame_objects)):
                obj2_box = current_frame_objects[j][:4]
                obj2_class = current_frame_objects[j][4]

                if obj1_class != obj2_class or (obj1_class == obj2_class and calculate_iou(obj1_box, obj2_box) < 0.9):
                    iou = calculate_iou(obj1_box, obj2_box)
                    if iou >= OVERLAP_THRESHOLD:
                        yolo_based_accident_flag = True
                        break
            if yolo_based_accident_flag:
                break
    
    keras_based_accident_flag = (keras_prediction_prob >= KERAS_ACCIDENT_THRESHOLD)

    is_current_frame_potential_accident = yolo_based_accident_flag and keras_based_accident_flag

    # --- Update Buffers ---
    accident_potential_status_window.append(is_current_frame_potential_accident)
    frame_buffer.append(display_frame.copy())

    # --- State Machine for Accident Confirmation and Clipping ---

    if current_system_state == "NORMAL":
        num_true_in_window = sum(accident_potential_status_window)

        if len(accident_potential_status_window) == CONFIRMATION_WINDOW_SIZE_FRAMES and \
           num_true_in_window >= MIN_ACCIDENT_FRAMES_IN_WINDOW:
            
            current_system_state = "ACCIDENT_CONFIRMED_AND_CLIPPING"
            
            print(f"\n***** ACCIDENT CONFIRMED! at Frame {frame_count}. ({num_true_in_window}/{CONFIRMATION_WINDOW_SIZE_FRAMES} potential frames). Initiating clip recording... *****", flush=True)
            
            # --- Output Accident Location to Terminal ---
            print(f"ACCIDENT LOCATION: {alarm_manager.CCTV_LOCATION_NAME} (Lat: {alarm_manager.CCTV_LATITUDE}, Lon: {alarm_manager.CCTV_LONGITUDE})", flush=True)
            # --- End Output ---

            clip_filename = os.path.join(OUTPUT_CLIPS_DIR, f"accident_clip_start_frame_{frame_count - CONFIRMATION_WINDOW_SIZE_FRAMES}.mp4")
            out = cv2.VideoWriter(clip_filename, FOURCC, FPS, (WIDTH, HEIGHT))
            
            start_buffer_idx_for_clip = max(0, len(frame_buffer) - (CONFIRMATION_WINDOW_SIZE_FRAMES + FRAMES_BEFORE_ACCIDENT_CLIP))
            
            for i in range(start_buffer_idx_for_clip, len(frame_buffer)):
                out.write(frame_buffer[i])
            
            clip_frames_written_after_confirmation = len(frame_buffer) - start_buffer_idx_for_clip
            
            # --- Call the alarm functions (SMS is simulated, Email is real) ---
            try:
                alarm_manager.send_sms_alert( # This will now print a simulated message
                    alarm_manager.CCTV_LOCATION_NAME, 
                    alarm_manager.CCTV_LATITUDE, 
                    alarm_manager.CCTV_LONGITUDE, 
                    clip_filename
                )
                alarm_manager.send_email_alert( # This will send a real email
                    ALERT_RECIPIENT_EMAIL, # This is your updated email
                    alarm_manager.CCTV_LOCATION_NAME, 
                    alarm_manager.CCTV_LATITUDE, 
                    alarm_manager.CCTV_LONGITUDE, 
                    clip_filename
                )
                print("Alerts triggered (SMS Simulated, Email Real) via Alarm Manager!", flush=True)
            except Exception as alert_e:
                print(f"Error triggering alerts through Alarm Manager: {alert_e}", flush=True)
            # --- End of alarm function calls ---

    elif current_system_state == "ACCIDENT_CONFIRMED_AND_CLIPPING":
        if out is not None:
            out.write(display_frame)
            clip_frames_written_after_confirmation += 1

            total_clip_desired_frames = (CONFIRMATION_WINDOW_SIZE_FRAMES + FRAMES_BEFORE_ACCIDENT_CLIP + FRAMES_AFTER_ACCIDENT_CLIP + 1)
            
            if clip_frames_written_after_confirmation >= total_clip_desired_frames:
                print(f"Clip finished for accident confirmed at Frame {frame_count}. Total clip frames: {clip_frames_written_after_confirmation}. Clip saved.", flush=True)
                out.release()
                out = None
                current_system_state = "COOLDOWN"
                cooldown_frame_counter = ACCIDENT_COOLDOWN_FRAMES
                print(f"Entering cooldown period for {ACCIDENT_COOLDOWN_FRAMES} frames...", flush=True)

    elif current_system_state == "COOLDOWN":
        cooldown_frame_counter -= 1
        if cooldown_frame_counter <= 0:
            current_system_state = "NORMAL"
            print(f"Cooldown period ended at Frame {frame_count}. System ready for new detections.", flush=True)
    
    # --- Display Status on Screen ---
    status_text = "No Accident"
    status_color = (0, 255, 0)
    
    if current_system_state == "ACCIDENT_CONFIRMED_AND_CLIPPING":
        status_text = "ACCIDENT CONFIRMED! (Recording Clip)"
        status_color = (0, 0, 255)
    elif current_system_state == "COOLDOWN":
        status_text = f"Cooldown ({cooldown_frame_counter}/{ACCIDENT_COOLDOWN_FRAMES})"
        status_color = (255, 165, 0)
    else:
        if len(accident_potential_status_window) == CONFIRMATION_WINDOW_SIZE_FRAMES:
             num_true_in_window = sum(accident_potential_status_window)
             status_text = f"Monitoring ({num_true_in_window}/{CONFIRMATION_WINDOW_SIZE_FRAMES} potential)"
             status_color = (255, 255, 0)
        else:
             status_text = f"Buffering ({len(accident_potential_status_window)}/{CONFIRMATION_WINDOW_SIZE_FRAMES})"
             status_color = (100, 100, 100)


    keras_prob_text = f"Keras Prob: {keras_prediction_prob:.2f}"
    cv2.putText(display_frame, keras_prob_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(display_frame, status_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    cv2.imshow('Hybrid Accident Detector (YOLO + Keras)', display_frame)

    # --- Handle Quit Command ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"\nAverage FPS: {fps:.2f}", flush=True)

if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()
print("Hybrid accident detection stopped.", flush=True)