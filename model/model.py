import os
import numpy as np
import tensorflow as tf
from preprocessing import get_datasets, read_manifest, CLASSES, CLASS_TO_INDEX, NUM_CLASSES

def compute_class_weights(manifest_csv="train_manifest.csv"):
    """
    Computes balanced class weights for the training set:
    weight_j = total_samples / (num_classes * count_j)
    Returns a dictionary mapping class_index -> weight.
    """
    labels, _, _ = read_manifest(manifest_csv)
    label_indices = [CLASS_TO_INDEX[lbl] for lbl in labels]
    total_samples = len(label_indices)
    counts = np.bincount(label_indices, minlength=NUM_CLASSES)
    
    class_weights = {}
    print("Training Set Class Distribution & Computed Weights:")
    print(f"{'Class Name':<32} {'Count':<8} {'Weight':<10}")
    print("-" * 52)
    for idx, (cls_name, count) in enumerate(zip(CLASSES, counts)):
        if count > 0:
            weight = total_samples / (NUM_CLASSES * count)
        else:
            weight = 1.0
        class_weights[idx] = float(weight)
        print(f"{cls_name:<32} {count:<8} {weight:.4f}")
    print("-" * 52)
    return class_weights

def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """
    1. Loads EfficientNetB0 pretrained on ImageNet (include_top=False, input_shape=(224, 224, 3)), frozen.
    2. Adds GlobalAveragePooling2D, Dropout(0.3), and Dense(10, activation='softmax').
    4. Compiles with Adam, categorical_crossentropy, accuracy, and top-2 accuracy.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    # Freeze base model initially
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_SkinClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )
    return model, base_model

if __name__ == '__main__':
    print("Loading datasets from preprocessing.py...")
    train_ds, val_ds, test_ds = get_datasets(batch_size=32)

    print("\nComputing class weights for training set...")
    class_weights = compute_class_weights("train_manifest.csv")

    print("\nBuilding model...")
    model, base_model = build_model()

    print("\n--- Model Summary ---")
    model.summary()
    
    print("\nReady for model training! (model.fit call reserved for training script)")
