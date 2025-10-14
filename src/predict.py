import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def run_prediction(img_path):
    model = tf.keras.models.load_model("runs/pneumonia_densenet121.h5")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
    print(f"Prediction: {label} ({prob:.4f})")
