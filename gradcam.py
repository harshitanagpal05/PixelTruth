import numpy as np
import cv2


def get_layer_recursive(model, layer_name):
    try:
        return model.get_layer(layer_name)
    except ValueError:
        # Search inside nested layers/models
        for layer in getattr(model, "layers", []):
            if hasattr(layer, "layers"):
                try:
                    return get_layer_recursive(layer, layer_name)
                except ValueError:
                    pass
        raise ValueError(f"No layer named {layer_name} found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer):
    # Import TensorFlow lazily to avoid import-time side effects during tests
    import tensorflow as tf

    target_layer = get_layer_recursive(model, last_conv_layer)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            target_layer.output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img