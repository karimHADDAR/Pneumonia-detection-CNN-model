import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

def run_evaluation():
    model = tf.keras.models.load_model("runs/pneumonia_densenet121.h5")
    base_dir = "data/chest_xray"
    test_dir = os.path.join(base_dir, "test")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        test_dir, target_size=(224, 224), batch_size=32, class_mode="binary", shuffle=False
    )

    y_true = test_gen.classes
    y_pred_prob = model.predict(test_gen).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

    # Show some pneumonia test images and predictions
    import matplotlib.pyplot as plt
    pneumonia_idx = np.where(test_gen.classes == 1)[0]
    sample_idx = np.random.choice(pneumonia_idx, size=min(6, len(pneumonia_idx)), replace=False)
    plt.figure(figsize=(12, 8))
    for i, img_idx in enumerate(sample_idx):
        batch_idx = img_idx // test_gen.batch_size
        within_batch_idx = img_idx % test_gen.batch_size
        batch = test_gen[batch_idx]
        img = batch[0][within_batch_idx]
        label = batch[1][within_batch_idx]
        pred = model.predict(np.expand_dims(img, axis=0))[0][0]
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"True: {int(label)}, Pred: {pred:.2f}")
        plt.axis('off')
    plt.suptitle("Sample Pneumonia Predictions")
    plt.tight_layout()
    plt.show()
