import cv2
import os
import requests
import json

API_KEY = "9655eb46-0161-11ee-82ad-9a31cf85287d"
MODEL_ID = "8e28ee52-35ed-4fbb-8f09-a5ec14885dd9"
IMAGE_FILENAME = "./test/result1.png"

def create_model():
    url = "https://app.nanonets.com/api/v2/OCR/Model/"
    payload = "{\"categories\" : [\"category1\", \"category2\"], \"model_type\": \"ocr\"}"
    headers = {
        'Content-Type': "application/json",
    }
    response = requests.request("POST", url, headers=headers, \
        auth=requests.auth.HTTPBasicAuth(API_KEY, ''), data=payload)
    print(response.text)

def load_model():
    url = "https://app.nanonets.com/api/v2/OCR/Model/" + MODEL_ID
    response = requests.request('GET', url, auth=requests.auth.HTTPBasicAuth(API_KEY, ''))
    print(response.text)

def upload_file():
    # for i in range():
    # dir_path = "./result_images"
    # count = 0
    # for path in os.listdir(dir_path):
    #     if os.path.isfile(os.path.join(dir_path, path)):
    #         count += 1
    # for i in range(count):
    url = f'https://app.nanonets.com/api/v2/OCR/Model/{MODEL_ID}/UploadFile/'
    data = {'file' :open(f"./images_for_training/train20.png", 'rb'), 'data' :('', '[{"filename":"./handwritten.png", "object": [{"name":"category1", "ocr_text":"text inside the bounding box", "bndbox": {"xmin": 1,"ymin": 1,"xmax": 100, "ymax": 100}}]}]')}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(API_KEY, ''), files=data)
    print(json.dumps(response.text, indent=4))

def predict_file():
    url = f'https://app.nanonets.com/api/v2/OCR/Model/{MODEL_ID}/LabelFile/'
    data = {'file': open('./handwritten.png', 'rb')}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(API_KEY, ''), files=data)
    print(response.text)

upload_file()