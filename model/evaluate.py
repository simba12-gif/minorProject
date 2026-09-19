import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import get_datasets, CLASSES, NUM_CLASSES

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "saved_model", "best_model_stage1.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at path: {model_path}")

    # 1. Load the Saved Model
    print(f"Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # Load Test Dataset from preprocessing.py
    print("\nLoading test dataset...")
    _, _, test_ds = get_datasets(batch_size=32)

    # 2. Extract True Labels and Run Predictions
    print("Running predictions on test_ds...")
    y_true_list = []
    for batch_images, batch_labels in test_ds:
        y_true_list.append(batch_labels.numpy())

    if len(y_true_list) == 0:
        print("Error: Test dataset is empty. Exiting.")
        return

    y_true_onehot = np.vstack(y_true_list)
    y_true_indices = np.argmax(y_true_onehot, axis=1)

    y_pred_probs = model.predict(test_ds)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # 5. Overall Test Accuracy
    test_acc = accuracy_score(y_true_indices, y_pred_indices)
    print("\n" + "=" * 65)
    print(f"OVERALL TEST ACCURACY: {test_acc * 100:.2f}%")
    print("=" * 65)

    # 3. Print Full Classification Report
    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT")
    print("=" * 65)
    report = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
    print(report)
    print("=" * 65)

    # 4. Compute and Plot Confusion Matrix Heatmap
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar=True,
        linewidths=0.5
    )
    plt.title('Confusion Matrix (best_model_stage1.keras)', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    cm_output_path = os.path.join(base_dir, "confusion_matrix.png")
    plt.savefig(cm_output_path, dpi=300)
    plt.close()
    print(f"\nConfusion matrix heatmap saved to: {cm_output_path}")

if __name__ == '__main__':
    main()
