import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


def run_training():
    base_dir = "data/chest_xray"
    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )
    test_gen = val_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    # Model
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss="binary_crossentropy",metrics=["accuracy"])

    print("🚀 Training phase 1...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    # Fine-tuning
    base_model.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss="binary_crossentropy",metrics=["accuracy"])

    print("🔧 Fine-tuning...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    # Evaluation
    y_true = test_gen.classes
    y_pred = model.predict(test_gen)
    auc = roc_auc_score(y_true, y_pred.ravel())
    print(f"✅ Test ROC-AUC: {auc:.4f}")

    os.makedirs("runs", exist_ok=True)
    model.save("runs/pneumonia_densenet121.h5")
    print("💾 Model saved to runs/pneumonia_densenet121.h5")

    # Show some pneumonia test images and predictions
    pneumonia_idx = np.where(test_gen.classes == 1)[0]
    sample_idx = np.random.choice(pneumonia_idx, size=min(6, len(pneumonia_idx)), replace=False)
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(sample_idx):
        img, label = test_gen[idx][0][0], test_gen[idx][1][0]
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"True: {int(label)}, Pred: {pred:.2f}")
        plt.axis('off')
    plt.suptitle("Sample Pneumonia Predictions")
    plt.tight_layout()
    plt.show()

    # Print statistics
    acc = np.mean((y_pred.ravel() > 0.5) == y_true)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")
    