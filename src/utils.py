import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

def generate_gradcam(model, img_path, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and model.
    Automatically finds the last convolutional layer if not specified.
    Works even if the model has a nested base model (e.g., DenseNet121 inside Sequential).
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # 🔍 Access the base convolutional model (e.g., DenseNet121)
    if "densenet121" in [l.name for l in model.layers]:
        base_model = model.get_layer("densenet121")
    else:
        base_model = model  # fallback, in case it's a direct model

    # If no layer_name is given, find the last conv layer dynamically
    if layer_name is None:
        conv_layers = [layer.name for layer in base_model.layers if "conv" in layer.name]
        if not conv_layers:
            raise ValueError("No convolutional layers found in the model.")
        layer_name = conv_layers[-1]

    # ✅ Get the chosen convolutional layer from the base model
    last_conv_layer = base_model.get_layer(layer_name)

    # Create a model that maps input -> activations + output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )

    # Compute the gradient of the top predicted class wrt conv output
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    return heatmap.numpy()
