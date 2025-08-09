# evaluate_keras_model.py (with Metrics Table and Graph)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Added accuracy_score
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # New import for tables

print("--- Starting Keras Model Evaluation for Confusion Matrix ---", flush=True)

# --- Configuration for Evaluation ---
KERAS_MODEL_PATH = r'C:\Users\reddy\OneDrive\Desktop\project_gemini\fine_tuned_accident_detector_model.h5'
KERAS_IMG_HEIGHT, KERAS_IMG_WIDTH = 224, 224 # Must match your Keras model's input size

# IMPORTANT: Update this path to the ROOT of your dataset
# (e.g., the folder containing 'training', 'validation', 'testing')
DATASET_ROOT_DIR = r'C:\Users\reddy\OneDrive\Desktop\project_gemini\dataset_prepared' 

# We will specifically use the 'testing' folder for evaluation
TEST_SET_DIR = os.path.join(DATASET_ROOT_DIR, 'test')
ACCIDENT_DIR = os.path.join(TEST_SET_DIR, 'accident')
NORMAL_DIR = os.path.join(TEST_SET_DIR, 'non accident') # Your 'normal' images are here

# The threshold you use in your main detection script to classify an image as 'accident'
KERAS_ACCIDENT_THRESHOLD = 0.55 

# --- Load Keras Model ---
print(f"\nLoading Keras Accident Classifier model from: {KERAS_MODEL_PATH}", flush=True)
try:
    keras_model = load_model(KERAS_MODEL_PATH)
    print("Keras model loaded successfully for evaluation!", flush=True)
except Exception as e:
    print(f"Error loading Keras model: {e}", flush=True)
    print("Please ensure the Keras model path is correct and it's a valid .h5 file.", flush=True)
    exit()

true_labels = []    # Stores the actual labels (0 for Normal, 1 for Accident)
predicted_labels = [] # Stores the model's predictions (0 for Normal, 1 for Accident)

print(f"\nEvaluating Keras model on 'testing' dataset at: {TEST_SET_DIR}", flush=True)

# --- Function to process images and make predictions ---
def process_and_predict(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            # print(f"Warning: Could not read image {image_path}. Skipping.", flush=True) # Uncomment for verbose warning
            return None
        
        img_resized = cv2.resize(img, (KERAS_IMG_WIDTH, KERAS_IMG_HEIGHT))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype('float32') / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction_prob = keras_model.predict(img_input, verbose=0)[0][0]
        # Classify based on your defined threshold
        predicted_class = 1 if prediction_prob >= KERAS_ACCIDENT_THRESHOLD else 0 # 1 for Accident, 0 for Normal
        return predicted_class

    except Exception as e:
        print(f"Error processing {image_path}: {e}", flush=True)
        return None

# --- Process 'accident' images (True Label = 1) ---
if os.path.exists(ACCIDENT_DIR) and os.path.isdir(ACCIDENT_DIR):
    print(f"Processing 'accident' images from: {ACCIDENT_DIR}", flush=True)
    for img_name in os.listdir(ACCIDENT_DIR):
        img_path = os.path.join(ACCIDENT_DIR, img_name)
        if os.path.isfile(img_path):
            prediction = process_and_predict(img_path)
            if prediction is not None:
                true_labels.append(1) # Ground truth is Accident
                predicted_labels.append(prediction)
else:
    print(f"Warning: 'accident' directory not found or not a directory at {ACCIDENT_DIR}. Please check path and existence.", flush=True)

# --- Process 'normal' images (True Label = 0) ---
if os.path.exists(NORMAL_DIR) and os.path.isdir(NORMAL_DIR):
    print(f"Processing 'normal' images from: {NORMAL_DIR}", flush=True)
    for img_name in os.listdir(NORMAL_DIR):
        img_path = os.path.join(NORMAL_DIR, img_name)
        if os.path.isfile(img_path):
            prediction = process_and_predict(img_path)
            if prediction is not None:
                true_labels.append(0) # Ground truth is Normal (No Accident)
                predicted_labels.append(prediction)
else:
    print(f"Warning: 'normal' directory not found or not a directory at {NORMAL_DIR}. Please check path and existence.", flush=True)

# --- Generate Confusion Matrix, Classification Report, Table & Graph ---
if len(true_labels) > 0:
    print("\n--- Evaluation Results ---", flush=True)
    
    # Calculate Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix (Raw):")
    print(cm)
    
    # Classification Report
    target_names = ['Normal', 'Accident'] 
    report = classification_report(true_labels, predicted_labels, target_names=target_names, output_dict=True)
    
    # Overall Accuracy
    overall_accuracy = accuracy_score(true_labels, predicted_labels)

    print("\nClassification Report (Detailed):")
    print(classification_report(true_labels, predicted_labels, target_names=target_names))

    # --- Performance Metrics Table ---
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Support'],
        'Normal': [report['Normal']['precision'], report['Normal']['recall'], report['Normal']['f1-score'], report['Normal']['support']],
        'Accident': [report['Accident']['precision'], report['Accident']['recall'], report['Accident']['f1-score'], report['Accident']['support']],
        'Overall': ['-', '-', '-', '-'] # Placeholder for metrics not per-class
    }
    
    # Add overall accuracy
    metrics_data['Overall'][0] = f"{overall_accuracy:.4f}" # Place accuracy in the first row of overall
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Metric', inplace=True)
    
    print("\n--- Performance Metrics Summary Table ---")
    print(metrics_df.to_string()) # Use to_string() for better console formatting

    # --- Plotting the Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix - Accident Detection')
    plt.show()

    # --- Plotting Comparison Graph of Metrics ---
    metrics_for_graph = {
        'Precision': [report['Normal']['precision'], report['Accident']['precision']],
        'Recall': [report['Normal']['recall'], report['Accident']['recall']],
        'F1-Score': [report['Normal']['f1-score'], report['Accident']['f1-score']]
    }

    metrics_df_graph = pd.DataFrame(metrics_for_graph, index=target_names)
    
    metrics_df_graph.plot(kind='bar', figsize=(10, 6))
    plt.title('Comparison of Key Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=0) # Keep labels horizontal
    plt.grid(axis='y', linestyle='--')
    plt.ylim(0, 1) # Metrics are between 0 and 1
    plt.legend(title='Metric Type')
    plt.tight_layout()
    plt.show()

    # You can also save the plots:
    # plt.savefig('confusion_matrix.png')
    # plt.savefig('performance_metrics_bar_chart.png')
    # print("\nConfusion matrix and performance metrics plots saved.", flush=True)

else:
    print("No images found or processed in the test dataset directories. Cannot generate any evaluation metrics.", flush=True)

print("\n--- Keras Model Evaluation Complete ---", flush=True)