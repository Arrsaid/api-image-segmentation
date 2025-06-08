import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import register_keras_serializable
from flask import Flask, request, jsonify, send_file
from utils import preprocess_image_bytes, load_real_image, load_real_mask, mask_to_color


@register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

@register_keras_serializable()
def mean_dice_metric(y_true, y_pred):
    return 1 - dice_loss(y_true, y_pred)

@register_keras_serializable()
class MeanIoUMetric(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=8, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

app = Flask(__name__)

# Charger le modèle
model = load_model(
    "model/best_unet_model.keras",
    custom_objects={
        "dice_loss": dice_loss,
        "mean_dice_metric": mean_dice_metric,
        "MeanIoUMetric": MeanIoUMetric
    }
)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Bienvenue sur l'API de segmentation d'images !"}), 200


# Endpoint 1 : prédiction à partir d'une image
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Pas d'image envoyée"}), 400

    image_bytes = request.files["image"].read()
    img_input = preprocess_image_bytes(image_bytes)
    prediction = model.predict(img_input)
    predicted_mask = np.argmax(prediction[0], axis=-1)
    

    colored_mask = mask_to_color(predicted_mask)
    img_io = io.BytesIO()
    colored_mask.save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
