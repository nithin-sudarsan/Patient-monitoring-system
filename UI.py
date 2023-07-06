import keras
import numpy as np
import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
import os
import tensorflow as tf
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import winsound

global fileName
window = tk.Tk()

window.title("PAIN CLASSIFICATION")

window.geometry("500x510")
img = Image.open("b.jpg")
bg = ImageTk.PhotoImage(img)

lbl = tk.Label(window, image=bg)
lbl.place(x=0, y=0)

#title = tk.Label(text="Click below to choose image file for testing....", background = "lightgreen", fg="Brown", font=("bold", 15))
# title.grid()
dirPath = "testpicture"
fileList = os.listdir(dirPath)
for fileName in fileList:
    os.remove(dirPath + "/" + fileName)

class_labels = ['InitialPain', 'NoPain', 'IntensePain']


def analyze():

    # Path to the folder containing the trained model
    model_path = r'VGG40_HSV.hd5'

    # Path to the single image for prediction
    image_path = fileName

    print(image_path)
    class_labels = ['InitialPain', 'NoPain', 'IntensePain']

    # Parameters
    input_shape = (224, 224, 3)
    num_classes = 3

    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the single image
    image = load_img(image_path, target_size=input_shape[:2])
    image_array = img_to_array(image)
    image_array = preprocess_input(image_array)
    image_array = tf.expand_dims(image_array, axis=0)

    # Make the prediction
    predictions = model.predict(image_array)
    predicted_class_index = tf.argmax(predictions, axis=1)[0]

    # Get the class labels
    # class_labels = list(train_generator.class_indices.keys())

    # Print the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    print(f'Predicted Class: {predicted_class_label}')
    if predicted_class_label == 'IntensePain':
        winsound.Beep(500, 1000)

    Pain = tk.Label(text='THE PREDICTED PAIN CLASS IS :' + predicted_class_label, bg="blue",
                    fg="white", font=("", 15))
    Pain.place(x=75, y=400)
    #disease.grid(column=0, row=4, padx=20, pady=20)
    button = tk.Button(text="Exit", command=exit)
    button.place(x=350, y=440)
   # button.grid(column=0, row=9, padx=20, pady=20)


def openphoto():

    fileName = askopenfilename(initialdir='data\\', title='Select image for analysis ',
                               filetypes=[('image files', '.*')])

    dst = "testpicture"
    print(fileName)
    print(os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    load = load.resize((300, 300))
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="300", width="300")
    img.image = render
    img.place(x=100, y=40)
    #img.grid(column=0, row=1, padx=10, pady = 10)

    button2 = tk.Button(text="PREDICT", command=analyze, width=8,
                        height=1, bg="white", fg='black', font=("bold", 15))
    button2.place(x=200, y=350)
    #button2.grid(column=0, row=2, padx=10, pady = 10)


#buttonbox = tk.Button(text="", width=300, height=300)
# buttonbox.place(x=100,y=200)
button1 = tk.Button(text="Choose file", command=openphoto, width=15, height=1)
button1.grid(column=0, row=1, padx=10, pady=10)


window.mainloop()
