import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2 # Will be used later for real-time video processing
import matplotlib.pyplot as plt # Useful for visualizing results or data

print(f"TensorFlow Version: {tf.__version__}")
# Check if GPU is available (highly recommended for performance)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print("GPU(s) available. Training and inference will be faster.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU found. Running on CPU may be slow for deep learning tasks.")

# --- Configuration for Image Processing ---
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Standard input size for ResNet50
BATCH_SIZE = 32 # Number of images to process in one go

# IMPORTANT: This should point to the folder containing your 'train', 'validation', 'test' directories.
# If your 'dataset' folder is directly in the same location as this script, 'dataset' is correct.
BASE_DATA_DIR = r'C:/Users/reddy/OneDrive/Desktop/project_gemini/dataset_prepared'

TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DATA_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

# --- Verify Dataset Directories Exist ---
print(f"\nChecking dataset directories:")
if not os.path.exists(TRAIN_DIR):
    print(f"Error: Training directory not found at {TRAIN_DIR}")
    print("Please ensure your 'dataset' folder is correctly placed and contains a 'train' subfolder with 'accident' and 'non_accident' subfolders inside.")
    exit()
if not os.path.exists(VALIDATION_DIR):
    print(f"Error: Validation directory not found at {VALIDATION_DIR}")
    print("Please ensure your 'dataset' folder is correctly placed and contains a 'validation' subfolder.")
    exit()
if not os.path.exists(TEST_DIR):
    print(f"Error: Test directory not found at {TEST_DIR}")
    print("Please ensure your 'dataset' folder is correctly placed and contains a 'test' subfolder.")
    exit()
print("All required dataset directories found. Proceeding with data loading.")

# --- Data Augmentation and Preprocessing Setup ---
# For training data: Apply various augmentations (rotations, shifts, flips, etc.) and normalize pixel values.
# Augmentation helps the model generalize better by seeing diverse variations of images.
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values from [0, 255] to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest' # Strategy for filling in new pixels created by transformations
)

# For validation and test data: Only normalize pixel values. No augmentation is applied during evaluation.
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Keep test data in fixed order for consistent final evaluation
)

# --- Load Images from Directories using flow_from_directory ---
print("\nLoading training data...")
# flow_from_directory automatically infers labels from subfolder names.
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH), # Resizes all images to 224x224
    batch_size=BATCH_SIZE,
    class_mode='binary', # Essential for 2-class classification (accident/non_accident)
    shuffle=True # Shuffle training data for better model learning
)

print("Loading validation data...")
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False # Keep validation data in fixed order for consistent evaluation
)



# --- Crucial: Understand Class Indices ---
# This dictionary maps the class names (your folder names) to the numerical labels (0 or 1)
# used by the generators. For example, {'accident': 0, 'non_accident': 1} or vice-versa.
print(f"\nClass indices mapping: {train_generator.class_indices}")

print("\n--- Step 1 complete. Data generators are set up. ---")
print("Please review the output, especially the 'Found X images...' lines for each generator,")
print("and critically, note down the 'Class indices mapping' as you will need it later.")
print("Once verified, you can proceed to Step 2: Feature Extraction with ResNet50.")

# (Continue from your existing accident_detector.py script after data generators are set up)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model # Added load_model for fine-tuning
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("\n--- Step 3 (Revised): Build and Train Keras CNN Model ---")

# --- PART 1: Build and Train the Classification Head (on frozen base) ---
print("\n--- Part 1: Training the classification head with frozen ResNet50 ---")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in base_model.layers:
    layer.trainable = False # Ensure all base model layers are frozen initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile with a slightly higher learning rate for the new head
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("Keras CNN Model (Head Only) built and compiled.")
model.summary()

# Callbacks for initial head training
# We'll save this model's weights separately if we want to restart fine-tuning
initial_model_checkpoint_path = 'initial_head_only_model.h5'
initial_checkpoint = ModelCheckpoint(initial_model_checkpoint_path,
                                    monitor='val_accuracy',
                                    save_best_only=True,
                                    mode='max',
                                    verbose=0) # Set verbose to 0 to keep initial training output clean

early_stopping_initial = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)


print("\nInitial training of the classification head (frozen base)...")
history_initial = model.fit(
    train_generator,
    epochs=10, # Fewer epochs for initial head training
    validation_data=validation_generator,
    callbacks=[early_stopping_initial, initial_checkpoint],
    verbose=1
)

# Load the best weights from the head-only training before fine-tuning
model.load_weights(initial_model_checkpoint_path)
print(f"Loaded best weights from initial head training: {initial_model_checkpoint_path}")


# --- PART 2: Fine-tuning the entire model (unfreezing some base layers) ---
print("\n--- Part 2: Fine-tuning ResNet50 base and classification head ---")

# Decide how many layers to unfreeze. ResNet50 has many layers.
# A common strategy is to unfreeze the last few convolutional blocks.
# Let's start by unfreezing the last 20 layers of ResNet50.
# You can experiment with this number (e.g., 30, 50, etc.).
for layer in base_model.layers[-20:]: # Unfreeze the last 20 layers
    layer.trainable = True
# You can also selectively unfreeze blocks like this (ResNet50 block5_conv1 is typically late):
# for layer in base_model.layers:
#     if layer.name == 'conv5_block1_0_conv': # or similar specific layer name
#         layer.trainable = True
#     else:
#         layer.trainable = False # Make sure earlier layers stay frozen, though they already are

# Re-compile the model with a very low learning rate for fine-tuning
# A much smaller learning rate is crucial to avoid destroying pre-trained weights.
model.compile(optimizer=Adam(learning_rate=0.00001), # VERY SMALL learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nModel re-compiled for fine-tuning (ResNet50 layers unfrozen).")
model.summary() # Review trainable parameters now

# Callbacks for fine-tuning phase
fine_tune_early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True) # More patience for fine-tuning
final_model_checkpoint_path = 'fine_tuned_accident_detector_model.h5' # New name for the fine-tuned model
fine_tune_checkpoint = ModelCheckpoint(final_model_checkpoint_path,
                                       monitor='val_accuracy',
                                       save_best_only=True,
                                       mode='max',
                                       verbose=1)

print("\nStarting fine-tuning...")
history_fine_tune = model.fit(
    train_generator,
    epochs=50, # Set a sufficiently high number of epochs, EarlyStopping will stop it.
    validation_data=validation_generator,
    callbacks=[fine_tune_early_stopping, fine_tune_checkpoint],
    verbose=1
)

print("\nKeras CNN model fine-tuning complete.")
print(f"Final best model saved to {final_model_checkpoint_path}")

print("\n--- Step 3 (Revised) complete. Keras CNN model trained and saved. ---")
print("You can now proceed to Step 4: Evaluate the Keras Model.")