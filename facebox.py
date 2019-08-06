
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time
from numpy import asarray
from face_detector1 import FaceDetector
from numpy import savez_compressed
MODEL_PATH = 'model.pb'
face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

def draw_boxes_on_image(image_array,image, boxes, scores):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image.size
    global faces
    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        faces=image_array[int(ymin):int(ymax),int(xmin):int(xmax)]
        faces=cv2.resize(faces,(160,160))
        
        draw.text((xmin, ymin), text='{:.3f}'.format(s))
     
    return image_copy,faces

# Training data
faces1=[]
labels=[]
for i in os.listdir("./data/train"):
    for j in os.listdir("./data/train/"+i):
        image_array = cv2.imread("./data/train/"+i+"/"+j)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array)
    
        boxes, scores = face_detector(image_array, score_threshold=0.3)
        image,face=draw_boxes_on_image(image_array,Image.fromarray(image_array), boxes, scores)
    
        #Extraction of faces from images
        face=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
        faces1.append(face)
        labels.append(i)
    #model=load_model("facenet_keras.h5")


trainX=asarray(faces1)
trainy=asarray(labels)

# testing set

faces1=[]
labels=[]
for i in os.listdir("./data/val"):
    for j in os.listdir("./data/val/"+i):
        image_array = cv2.imread("./data/val/"+i+"/"+j)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array)
    
        boxes, scores = face_detector(image_array, score_threshold=0.3)
        image,face=draw_boxes_on_image(image_array,Image.fromarray(image_array), boxes, scores)
    
        #Extraction of faces from images
        face=cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
        faces1.append(face)
        labels.append(i)
    #model=load_model("facenet_keras.h5")


testX=asarray(faces1)
testy=asarray(labels)


# Saving the data

savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)