from flask import Flask, request, jsonify
import numpy as np
import cv2
from keras.models import load_model
from keras import backend as K

app = Flask(__name__)

smooth = 100


def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


def jac_distance(y_true, y_pred):
    return -iou(y_true, y_pred)


# Load the pre-trained UNet model
model = load_model('unet_brain_mri_seg.h5',
                   custom_objects={'dice_coef_loss': dice_coef_loss,
                                   'iou': iou,
                                   'dice_coef': dice_coef,
                                   'jac_distance': jac_distance})


@app.route('/predict', methods=['POST'])
def predict():
    # Receive image data from the client
    img_data = request.files['image']
    img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), cv2.IMREAD_COLOR)
    im_width = 256
    im_height = 256
    # Preprocess the image
    img = cv2.resize(img, (im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]

    # Make prediction
    pred = model.predict(img)

    # Convert prediction to binary mask
    pred_mask = (pred > 0.5).astype(np.uint8)

    # Prepare response
    response = {
        'prediction': pred_mask.tolist()
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
