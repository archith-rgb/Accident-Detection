import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print("GPU(s) available for real-time inference.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU found. Running inference on CPU.")


# --- Configuration ---
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Must match your model's input size
MODEL_PATH = 'fine_tuned_accident_detector_model.h5' # Path to your saved model
CONFIDENCE_THRESHOLD = 0.54 # Adjust as needed: probability >= this will be 'Accident'

# --- Video Source Configuration ---
# OPTION 1: Use a webcam (usually 0 for default webcam)
# VIDEO_SOURCE = 0

# OPTION 2: Use a video file
# Make sure the video file is in the same directory as this script, or provide a full path.
VIDEO_SOURCE = r"A:/ACCIDENT_DETECTION/videos/0IPAgFHyRVI.mp4" # <--- IMPORTANT: Change this to your video file name or path

# --- Verify Model Path ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it was saved correctly.")
    exit()

print(f"\nLoading model from: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the model path is correct and it's a valid Keras .h5 file.")
    exit()

print(f"\nOpening video source: {VIDEO_SOURCE}")
# Open the video capture
# If using a webcam, VIDEO_SOURCE = 0
# If using a video file, VIDEO_SOURCE = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open video source {VIDEO_SOURCE}")
    print("If using a webcam, ensure it's not in use by another application.")
    print("If using a video file, ensure the path is correct and the file exists.")
    exit()

print("\nStarting real-time detection. Press 'q' to quit.")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read() # Read a frame from the video
    if not ret:
        print("End of video stream or error reading frame.")
        break

    frame_count += 1

    # --- Pre-process the frame for the model ---
    # 1. Resize the frame to the model's input size
    img_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # 2. Convert BGR (OpenCV default) to RGB (Keras/TF model expectation)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # 3. Convert to float32 and normalize pixel values to [0, 1]
    img_normalized = img_rgb.astype('float32') / 255.0

    # 4. Add a batch dimension: (height, width, channels) -> (1, height, width, channels)
    img_input = np.expand_dims(img_normalized, axis=0)

    # --- Make Prediction ---
    prediction = model.predict(img_input)
    # The model outputs a single probability (e.g., [[0.95]]). We need the scalar value.
    accident_probability = prediction[0][0]

    # --- Interpret Prediction and Display Results ---
    label_text = "No Accident"
    color = (0, 255, 0) # Green for no accident

    if accident_probability >= CONFIDENCE_THRESHOLD:
        label_text = "ACCIDENT DETECTED!"
        color = (0, 0, 255) # Red for accident

    # Format text to show probability (helpful for debugging)
    display_text = f"{label_text} (Prob: {accident_probability:.2f})"

    # Overlay text on the original frame
    cv2.putText(frame, display_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Accident Detector', frame)

    # --- Handle Quit Command ---
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
end_time = time.time()
fps = frame_count / (end_time - start_time)
print(f"\nAverage FPS: {fps:.2f}")

cap.release() # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV display windows
print("Real-time detection stopped.")