import os
import json
import cv2
import numpy as np
import tensorflow as tf
from preprocessing import CLASSES, NUM_CLASSES, read_manifest

# Global variable to cache the loaded model instance
_MODEL = None

def get_model(model_path=None):
    """
    1. Loads and caches the saved Keras model so repeated calls don't reload it.
    """
    global _MODEL
    if _MODEL is None:
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "saved_model", "best_model_stage1.keras")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        print(f"Loading and caching model from: {model_path}...")
        _MODEL = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
    return _MODEL

def preprocess_single_image(image_path):
    """
    2. Reads and preprocesses a single image file the same way as training:
       - OpenCV read
       - BGR to RGB conversion
       - Resize to 224x224
       - Normalize pixel values to [0.0, 1.0]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image file at: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def predict_image(image_path, model_path=None):
    """
    3 & 4. Runs prediction and returns exact dictionary format:
    {
        "predicted_class": <top class name string>,
        "confidence": <float 0-1>,
        "top_3": [
            {"class": <name>, "confidence": <float>}, ...
        ],
        "image_id": <the input filename>
    }
    """
    # 1. Load / get cached model
    model = get_model(model_path)
    
    # 2. Preprocess single image
    img = preprocess_single_image(image_path)
    img_batch = np.expand_dims(img, axis=0)  # Shape: (1, 224, 224, 3)
    
    # 3. Run prediction
    probs = model.predict(img_batch, verbose=0)[0]
    
    # 4. Format output dictionary
    top_idx = int(np.argmax(probs))
    top_class = CLASSES[top_idx]
    top_confidence = float(probs[top_idx])
    
    # Get top 3 sorted by confidence descending
    top_3_indices = np.argsort(probs)[::-1][:3]
    top_3 = [
        {
            "class": CLASSES[idx],
            "confidence": float(probs[idx])
        }
        for idx in top_3_indices
    ]
    
    image_id = os.path.basename(image_path)
    
    return {
        "predicted_class": top_class,
        "confidence": top_confidence,
        "top_3": top_3,
        "image_id": image_id
    }

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load sample image from test set manifest
    labels, case_ids, _ = read_manifest("test_manifest.csv")
    if len(case_ids) > 0:
        sample_label = labels[0]
        sample_case_id = case_ids[0]
        sample_image_path = os.path.join(base_dir, "images", sample_label, f"{sample_case_id}.jpg")
        
        print(f"--- Testing predict_image on sample test image ---")
        print(f"Sample Path : {sample_image_path}")
        print(f"Actual Label: {sample_label}\n")
        
        result = predict_image(sample_image_path)
        
        print("Prediction Result:")
        print(json.dumps(result, indent=2))
        
        print("\n--- Testing Model Caching (Second Call) ---")
        result2 = predict_image(sample_image_path)
        print("Second call executed without reloading model!")
