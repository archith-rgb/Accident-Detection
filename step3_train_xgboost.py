'''# scripts/step3_train_xgboost.py
import numpy as np
import xgboost as xgb
from xgboost.callback import EarlyStopping as xgb_EarlyStopping
import os

# --- Paths for loading features and saving models ---
FEATURES_DIR = '../features' # Path to your features folder
MODELS_DIR = '../models'     # Path to your models folder

# --- NEW: Load the extracted features and labels ---
try:
    train_features = np.load(os.path.join(FEATURES_DIR, 'train_features.npy'))
    train_labels = np.load(os.path.join(FEATURES_DIR, 'train_labels.npy'))
    validation_features = np.load(os.path.join(FEATURES_DIR, 'validation_features.npy'))
    validation_labels = np.load(os.path.join(FEATURES_DIR, 'validation_labels.npy'))
    # No need to load test features/labels here, as they are used in evaluation (Step 4)
    print(f"Features and labels loaded successfully from {FEATURES_DIR}/")
except FileNotFoundError as e:
    print(f"Error loading features: {e}. Please ensure step2_feature_extraction.py was run successfully and saved the files.")
    exit()

print("\n--- Step 3: Train XGBoost Classifier ---")

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

early_stopping_callback = xgb_EarlyStopping(
    patience=50,
    monitor='validation_0-logloss',
    data_name='validation_0'
)

print("Training XGBoost model on extracted features...")
xgb_model.fit(
    train_features, train_labels,
    eval_set=[(validation_features, validation_labels)],
    callbacks=[early_stopping_callback],
    verbose=True
)

print("XGBoost training complete.")

# --- Save the trained XGBoost model ---
xgb_model_path = os.path.join(MODELS_DIR, 'xgboost_accident_detector.json')
try:
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost model saved successfully to {xgb_model_path}")
except Exception as e:
    print(f"Error saving XGBoost model: {e}")
    print("Please check write permissions or disk space in the models directory.")

print("\n--- Step 3 complete. XGBoost model trained and saved. ---")
print("You can now run step4_evaluate_model.py independently.")'''

print(f"XGBoost version currently active in script: {xgb.__version__}")