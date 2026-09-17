import os
import csv
import numpy as np
import cv2
import tensorflow as tf

# Define the 10 skin condition classes and mapping
CLASSES = [
    "Allergic Contact Dermatitis",
    "Eczema",
    "Folliculitis",
    "Herpes Zoster",
    "Impetigo",
    "Insect Bite",
    "Pigmented purpuric eruption",
    "Psoriasis",
    "Tinea",
    "Urticaria"
]
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

def load_and_preprocess_image(label, case_id, images_dir="images"):
    """
    1. Loads an image given its label and case_id using OpenCV,
       resizes it to 224x224, and normalizes pixel values to [0, 1].
    """
    image_path = os.path.join(images_dir, str(label), f"{str(case_id)}.jpg")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def _to_str(val):
    if hasattr(val, 'numpy'):
        val = val.numpy()
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    return str(val)

def _load_image_np(label_bytes, case_id_bytes, images_dir_bytes):
    label = _to_str(label_bytes)
    case_id = _to_str(case_id_bytes)
    images_dir = _to_str(images_dir_bytes)
    return load_and_preprocess_image(label, case_id, images_dir)

def load_image_tf(label, case_id, images_dir="images"):
    """
    TensorFlow wrapper around OpenCV image loading function.
    """
    img = tf.py_function(
        func=_load_image_np,
        inp=[label, case_id, images_dir],
        Tout=tf.float32
    )
    img.set_shape((224, 224, 3))
    return img

def read_manifest(manifest_csv, images_dir="images"):
    """
    Reads manifest CSV and returns lists of labels, case_ids, and one-hot encoded targets.
    """
    labels = []
    case_ids = []
    one_hot_labels = []
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, manifest_csv) if not os.path.isabs(manifest_csv) else manifest_csv

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row['label'].strip()
            case_id = row['case_id'].strip()
            img_path = os.path.join(base_dir, images_dir, label, f"{case_id}.jpg")
            
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                labels.append(label)
                case_ids.append(case_id)
                idx = CLASS_TO_INDEX[label]
                one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
                one_hot[idx] = 1.0
                one_hot_labels.append(one_hot)

    return labels, case_ids, np.array(one_hot_labels, dtype=np.float32)

def get_augmentation_layer():
    """
    3. Data augmentation sequential layer:
       Random horizontal flip, rotation, zoom, brightness.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
    ], name="data_augmentation")

def create_dataset(manifest_csv, images_dir="images", is_training=False, batch_size=32):
    """
    2, 3, 4. Creates tf.data.Dataset pipeline:
       - Loads images and one-hot labels
       - Applies data augmentation ONLY to training set
       - Batches (batch_size=32) and prefetches
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    abs_images_dir = os.path.join(base_dir, images_dir) if not os.path.isabs(images_dir) else images_dir

    labels, case_ids, one_hot_labels = read_manifest(manifest_csv, images_dir=abs_images_dir)
    
    labels_tensor = tf.constant(labels, dtype=tf.string)
    case_ids_tensor = tf.constant(case_ids, dtype=tf.string)
    images_dir_tensor = tf.constant(abs_images_dir, dtype=tf.string)
    targets_tensor = tf.constant(one_hot_labels, dtype=tf.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((labels_tensor, case_ids_tensor, targets_tensor))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(labels), reshuffle_each_iteration=True)
        
    def map_fn(lbl, cid, target):
        img = load_image_tf(lbl, cid, images_dir_tensor)
        return img, target
        
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    
    if is_training:
        augmenter = get_augmentation_layer()
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def get_datasets(batch_size=32, images_dir="images"):
    """
    Returns (train_ds, val_ds, test_ds)
    """
    train_ds = create_dataset("train_manifest.csv", images_dir=images_dir, is_training=True, batch_size=batch_size)
    val_ds = create_dataset("val_manifest.csv", images_dir=images_dir, is_training=False, batch_size=batch_size)
    test_ds = create_dataset("test_manifest.csv", images_dir=images_dir, is_training=False, batch_size=batch_size)
    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(batch_size=32)
    
    print("\n--- Testing Single Batch from Training Dataset ---")
    for batch_images, batch_labels in train_ds.take(1):
        print(f"Batch images shape : {batch_images.shape}")
        print(f"Batch labels shape : {batch_labels.shape}")
        print(f"Pixel value min/max : {tf.reduce_min(batch_images):.4f} / {tf.reduce_max(batch_images):.4f}")
        
        # Label distribution in this batch
        label_indices = np.argmax(batch_labels.numpy(), axis=1)
        counts = np.bincount(label_indices, minlength=NUM_CLASSES)
        print("\nLabel Distribution in Batch:")
        for cls_name, count in zip(CLASSES, counts):
            print(f"  {cls_name:30s}: {count}")
