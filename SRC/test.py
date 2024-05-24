import os
import pandas as pd
import streamlit as st
import cv2

# st.write('Blah!!!!')

import matplotlib.pyplot as plt
import glob, random

labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
class_choice = random.choice(labels)

# for i in labels:
#     folderPath = os.path.join('/Users/pranavmoses/Desktop/5505/Project-INFO5505-Pranav Moses/project/tumor.v3i.folder','train',i)

def choose_random_image(folder_path):
    # Get a list of all files in the folder
    class_choice = random.choice(labels)
    folder_path = os.path.join(folder_path, class_choice)
    files = os.listdir(folder_path)
    # Filter out non-image files (you might want to adjust this based on the types of images you have)
    image_files = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    # Choose a random image from the list
    if image_files:
        random_image = random.choice(image_files)
        return os.path.join(folder_path, random_image)
    else:
        return None

folder_path = "/Users/pranavmoses/Desktop/5505/Project-INFO5505-Pranav Moses/project/tumor.v3i.folder/test/glioma_tumor"
random_image_path = choose_random_image(folder_path)
img = cv2.imread(random_image_path)
st.image(img)