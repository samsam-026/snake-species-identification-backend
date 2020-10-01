import io
import json
import math
import time
import os
import cv2
import base64


from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import numpy as np
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from image_processing import preprocess_image
from custom_model import CustomModel

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class_index = json.load(open('class_index.json'))

# Open and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load("models/custom_model.pth", map_location=device)
model = CustomModel(5)
model.load_state_dict(state)
model = model.to(device)

# switch to `eval` mode for testing
model.eval()

class_index = json.load(open('class_index.json'))

# Values for image transformation
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

img_size = 384

image_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean_nums, std_nums)
                                       ])


def get_species_info(predicted_idx):
    return class_index[predicted_idx]


def transform_image(preprocessed_image):
    try:
        proc_img = Image.fromarray(preprocessed_image)
        return image_transforms(proc_img).unsqueeze(0)
    except:
        return {"message": "A Transform error has occurred"}


def get_prediction(image_bytes, add_data=None):
    try:
        tensor = transform_image(image_bytes)
        if type(tensor) != dict:

            outputs = model(tensor)

            _, preds = torch.max(outputs, 1)
            predicted_idx = preds.item()

            percentage = F.softmax(outputs, dim=1)[0] * 100

            prob_arr = percentage.tolist()
            standard_dev = np.std(prob_arr, dtype=np.float64)
            confidence = percentage[predicted_idx].item()

            class_id = str(predicted_idx + 1)

            if confidence > 50 and standard_dev > 20:
                return {"classId": class_id, "confidence": confidence, "standardDev": standard_dev}
            else:
                return {"message": "Snake species cannot be identified"}
        else:
            return tensor
    except:
        return {"message": "A Prediction error has occurred"}


@app.route('/')
@cross_origin()
def hello():
    return 'API Online'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        img_file = request.files['file']
        filename = img_file.mimetype
        if "image" in filename or "octet-stream" in filename:
            img_bytes = cv2.imdecode(np.frombuffer(
                img_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            preprocessed_image = preprocess_image(img_bytes)

            add_data = None

            values = get_prediction(preprocessed_image, add_data)
            if type(values) == dict and "classId" in values:
                species_val = get_species_info(values["classId"])

                return jsonify({'data': {
                    **species_val,
                    "confidence": values["confidence"],
                    "standardDev": values["standardDev"],
                    "time": current_time,
                }}), 200
            else:
                return jsonify({"error": values}), 400
        else:
            return jsonify({"error": {"message": "Wrong file type", "currentType": filename}}), 400
    except:
        return jsonify({"error": {"message": "An error has occurred"}}), 400
