import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print("GPU(s) available for evaluation.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU found. Running evaluation on CPU.")

# --- Configuration (must match your training configuration) ---
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32 # Can be adjusted for evaluation
BASE_DATA_DIR = r'C:/Users/reddy/OneDrive/Desktop/project_gemini/dataset_prepared' # Assuming 'dataset' folder is in the same directory as this script

TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')
MODEL_PATH = 'fine_tuned_accident_detector_model.h5' # Path to your saved model

# --- Verify Model and Dataset Directories Exist ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure it was saved correctly.")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"Error: Test dataset directory not found at {TEST_DIR}. Please check BASE_DATA_DIR path.")
    exit()

print(f"\nLoading model from: {MODEL_PATH}")
# Load the best saved model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure your Keras version matches the version used to save the model.")
    exit()

# --- Setup Test Data Generator (no augmentation for evaluation) ---
test_datagen = ImageDataGenerator(rescale=1./255) # Just normalization

print("\nLoading test data for evaluation...")
test_generator = test_datagen.flow_from_directory(TEST_DIR,
                                                  target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='binary',
                                                  shuffle=False) # Important: do NOT shuffle test data

# --- Evaluate the Model ---
print("\n--- Step 4: Evaluate the Keras Model ---")
print(f"Evaluating model on {test_generator.samples} test samples...")

# Evaluate the model on the test data
evaluation_results = model.evaluate(test_generator, verbose=1)

# Print evaluation metrics
print(f"\nTest Loss: {evaluation_results[0]:.4f}")
print(f"Test Accuracy: {evaluation_results[1]:.4f}")

print("\n--- Evaluation complete. ---")