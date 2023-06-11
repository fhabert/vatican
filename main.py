from PIL import Image
from roboflow import Roboflow
import cv2
import os
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import requests
import matplotlib.pyplot as plt

# pytesseract.pytesseract.tesseract_cmd = 'C:/Users/felix/miniconda3/envs/tf/Lib/site-packages/pytesseract'

def obtain_thresh_image(ar: np.asarray, index: int):
    image = Image.fromarray(ar)
    resized = image.resize((1000,1000))
    opencv_image = np.array(resized)
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Image.fromarray(binary_image).save(f"./images_for_training/train{index}.png")
    pass 

def get_crop_image():
    rf = Roboflow(api_key="V9qEIrKTpyQTYNn5m42p")
    project = rf.workspace().project("vatican")
    model = project.version(1).model

    # infer on a local image
    dir_path = "./hq_database"
    count = 0
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            count += 1
    print(count)
    for i in range(count):
        info = model.predict(f"./hq_database/image{i}.jpg", confidence=40, overlap=30).json()
        center_x = info["predictions"][0]["x"]
        center_y = info["predictions"][0]["y"]
        w, h = info["predictions"][0]["width"], info["predictions"][0]["height"]
        im = Image.open(f"./hq_database/image{i}.jpg")
        top_left = center_x - int(w/2)
        upper_left = center_y - int(h/2)
        down_right = center_x + int(w/2) 
        lower_right = center_y + int(h/2) 
        im_cropped = im.crop((top_left, upper_left,down_right,lower_right))
        im_cropped_pixels = np.asarray(im_cropped)
        obtain_thresh_image(im_cropped_pixels, i)
    pass


def ocr_practise():
    # img_base = Image.open("./database/image0.jpg")
    # img_base.show()
    img = Image.open("./images_for_training/train3.png")
    img_ar = np.array(img)
    contours,_ = cv2.findContours(img_ar, 3, 2)
    cnt_area = []
    for i in range(0,30):
        cnt_area.append(cv2.contourArea(contours[i]))
    low_threshold = 300
    up_threshold = 8000
    print("Without filters:", len(contours))
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > low_threshold \
                         and cv2.contourArea(cnt) < up_threshold]

    filtered_contours = sort_contours(contours, method="left-to-right")[0]
    print("Number of contours:", len(filtered_contours))
    stored_coordinates = []
    cnt = filtered_contours[0]
    prev_x,_,prev_w,_ = cv2.boundingRect(cnt)
    
    text = ""

    for i in range(1,50,1):
        cnt = filtered_contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        x_left = prev_x + prev_w
        x_right = x
        dis = x_right-x_left
        if dis > -100:
            get_text = translate(x,y,w,h,img_ar)
            if not get_text.isdigit():
                text += get_text
            # plt.imshow(let_resized, cmap="gray")
            # plt.show()
        else:
            stored_coordinates.append([x,y,w+prev_w,h])
        image = cv2.rectangle(img_ar,(x,y),(x+w,y+h),(0,0,0),1)
    
    write_to_file(text)
    _ = plt.figure(figsize=(30, 30))
    plt.imshow(image, cmap="gray")
    plt.show()
    pass

def translate(x,y,w,h,img_ar):
    letters = img_ar[y:y+h, x:x+w]
    let_img = Image.fromarray(letters)
    let_resized = let_img.resize((64,64))
    let_resized.show()
    conv = let_resized.convert("RGB")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    pixel_values = processor(images=conv, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text[0] + " "

def write_to_file(text: str):
    file = open("./translation.txt", "a")
    file.write(f"{text}\n")
    file.close()
    pass


def get_from_file():
    with open("./translation.txt", "r") as f:
        phrases = f.readlines()
        for item in phrases:
            print(item[:-2])

# ocr_practise()
# get_from_file()

let_img = Image.open("./test_hand.jpg")
let_img = Image.open("./small_ph.png")
width, height = let_img.size
im_array  = let_img.resize((width*5, height*5))
conv = np.array(im_array.convert("RGB"))
sharpening_filter  = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
im = cv2.filter2D(conv, -1, sharpening_filter)
# let_resized = let_img.resize((1000,1000))
let_img = Image.fromarray(im)


processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
pixel_values = processor(images=let_img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text[0])

# def sobel():
    # sobelx = cv2.Sobel(img_ar,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(img_ar,cv2.CV_64F,0,1,ksize=5)
    # gradient_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    # contours,hierarchy = cv2.findContours(binary_image, 1, 2)
    # # src_gray = cv2.cvtColor(img_ar, cv2.COLOR_BGR2GRAY)
    # threshold = 100
    # canny_output = cv2.Canny(img_ar, threshold, threshold * 2)
    # gradient_direction = np.arctan2(sobely, sobelx)
    # plt.imshow(canny_output)
    # plt.imshow(gradient_magnitude)
    # plt.show()
    # pass


# get_crop_image()


# print(len(contours))
# img = cv2.imread("./latin1.jpg")
# 
# rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
# dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
# contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
#                                                  cv2.CHAIN_APPROX_NONE)
# im2 = img.copy()
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

# window_name = 'image'
# cv2.imshow(window_name, im2)
# cv2.waitKey(0)
 # file = open("recognized.txt", "w+")
# file.write("")
# file.close()
 
# # Looping through the identified contours
# # Then rectangular part is cropped and passed on
# # to pytesseract for extracting text from it
# # Extracted text is then written into the text file
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
     
#     # Drawing a rectangle on copied image
#     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
     
#     # Cropping the text block for giving input to OCR
#     cropped = im2[y:y + h, x:x + w]
     
#     # Open the file in append mode
#     file = open("recognized.txt", "a")
     
#     # Apply OCR on the cropped image
#     text = pytesseract.image_to_string(cropped)
     
#     # Appending the text into file
#     file.write(text)
#     file.write("\n")
     
#     # Close the file
#     file.close