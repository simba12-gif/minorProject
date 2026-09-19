import os
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import get_datasets
from model import build_model, compute_class_weights

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(base_dir, "saved_model")
    os.makedirs(saved_model_dir, exist_ok=True)

    # 1. Load Datasets and Compute Class Weights
    print("Loading datasets...")
    train_ds, val_ds, test_ds = get_datasets(batch_size=32)

    print("\nComputing class weights for training dataset...")
    class_weights = compute_class_weights("train_manifest.csv")

    # 2. Build Model (Stage 1: Frozen Base)
    print("\nBuilding model with frozen EfficientNetB0 base...")
    model, base_model = build_model()

    # Stage 1 Callbacks
    stage1_checkpoint_path = os.path.join(saved_model_dir, "best_model_stage1.keras")
    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=stage1_checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
    ]

    print("\n==================================================")
    print("STAGE 1: Training Classification Head (15 Epochs max)")
    print("==================================================")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks_stage1
    )

    stage1_val_acc = max(history_stage1.history['val_accuracy'])
    print(f"\n---> Stage 1 Final Best Validation Accuracy: {stage1_val_acc * 100:.2f}%\n")

    # 3. Stage 2: Fine-Tuning (Unfreeze last 30 layers)
    print("==================================================")
    print("STAGE 2: Fine-Tuning Base Model (Last 30 Layers)")
    print("==================================================")
    
    base_model.trainable = True
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Recompile with a lower learning rate (1e-5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )

    stage2_checkpoint_path = os.path.join(saved_model_dir, "best_model_finetuned.keras")
    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=stage2_checkpoint_path,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
    ]

    stage1_epochs_run = len(history_stage1.history['accuracy'])
    total_epochs = stage1_epochs_run + 15

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        initial_epoch=stage1_epochs_run,
        class_weight=class_weights,
        callbacks=callbacks_stage2
    )

    stage2_val_acc = max(history_stage2.history['val_accuracy'])
    print(f"\n---> Stage 2 Final Best Validation Accuracy: {stage2_val_acc * 100:.2f}%\n")

    # 4. Combine Histories and Plot Training Curves
    acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
    val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']
    loss = history_stage1.history['loss'] + history_stage2.history['loss']
    val_loss = history_stage1.history['val_loss'] + history_stage2.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='s')
    plt.axvline(x=stage1_epochs_run + 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='s')
    plt.axvline(x=stage1_epochs_run + 0.5, color='gray', linestyle='--', label='Fine-Tuning Start')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(base_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Combined training curves saved to: {plot_path}")

    # Final Summary
    print("\n==================================================")
    print("TRAINING SUMMARY REPORT")
    print("==================================================")
    print(f"Stage 1 Best Validation Accuracy : {stage1_val_acc * 100:.2f}%")
    print(f"Stage 2 Best Validation Accuracy : {stage2_val_acc * 100:.2f}%")
    print("==================================================")

if __name__ == '__main__':
    main()
