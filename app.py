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
from custom_model_with_location import CustomModelLoc

import firebase_admin
from firebase_admin import credentials, firestore, storage

from google.cloud.firestore_v1 import Increment

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Initialize firebase
cred = credentials.Certificate(
    "firebase.json")
firebase_admin.initialize_app(cred)

store = firestore.client()
snake_species_ref = store.collection(u'snake-species')
authority_data_ref = store.collection(u'authority-data')

class_index = json.load(open('class_index.json'))

# Open and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load("custom_model.pth", map_location=device)
model = CustomModel(5)
model.load_state_dict(state)
model = model.to(device)

state_loc = torch.load("custom_model_with_location.pth", map_location=device)
model_loc = CustomModelLoc(5)
model_loc.load_state_dict(state_loc)
model = model.to(device)

# switch to `eval` mode for testing
model.eval()
model_loc.eval()

# Values for image transformation
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

img_size = 384

image_transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean_nums, std_nums)
                                       ])


def get_species_info(class_id):
    snake_species = snake_species_ref.document(class_id).get()
    return snake_species.to_dict()


def transform_image(preprocessed_image):
    try:
        proc_img = Image.fromarray(preprocessed_image)
        return image_transforms(proc_img).unsqueeze(0)
    except:
        return {"message": "A Transform error has occurred"}


def get_add_data(form):
    values = [float(form['lat']), float(form['lng']),
              int(form['hour']), int(form['minute'])]
    return np.array([values])


def get_prediction(image_bytes, add_data=None):
    try:
        tensor = transform_image(image_bytes)
        if type(tensor) != dict:

            if add_data is None:
                outputs = model(tensor)
            else:
                outputs = model_loc(tensor, add_data.float())

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


def increment_sighting(class_id):
    authority_data_ref.document(class_id).update({
        u'y': Increment(1)
    })


def save_to_firestore_norm(user_id, classification, lat, lng, current_time):
    data = {
        u'time': current_time,
        u'user': user_id,
        u'classId': classification["classId"],
        u'confidence': classification["confidence"],
        u'showLoc': True,
        u'location': {
            u'lat': lat,
            u'lng': lng
        }
    }
    store.collection("sightings").document(current_time).set(data)
    increment_sighting(classification["classId"])


def save_to_firestore(user_id, classification, current_time):
    data = {
        u'time': current_time,
        u'user': user_id,
        u'classId': classification["classId"],
        u'confidence': classification["confidence"],
    }
    store.collection("sightings").document(current_time).set(data)


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
        user_id = request.form['user']
        if "image" in filename or "octet-stream" in filename:
            img_bytes = cv2.imdecode(np.frombuffer(
                img_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            preprocessed_image = preprocess_image(img_bytes)

            add_data = None
            if "lat" in request.form:
                add_data = torch.from_numpy(get_add_data(request.form))
                add_data = add_data.to(device)

            values = get_prediction(preprocessed_image, add_data)
            if type(values) == dict and "classId" in values:
                species_val = get_species_info(values["classId"])
                if "lat" in request.form:
                    save_to_firestore_norm(
                        user_id, values, request.form['lat'], request.form['lng'], current_time)
                else:
                    save_to_firestore(user_id, values, current_time)

                return jsonify({'data': {
                    **species_val,
                    "confidence": values["confidence"],
                    "standardDev": values["standardDev"],
                    "time": current_time
                }}), 200
            else:
                return jsonify({"error": values}), 400
        else:
            return jsonify({"error": {"message": "Wrong file type", "currentType": filename}}), 400
    except:
        return jsonify({"error": {"message": "An error has occurred"}}), 400
