import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import glob, random
import os
from skimage.filters import sobel

labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
st.write('Blah!!!!')

def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    for image in range(x_train.shape[0]):
        df = pd.DataFrame()
        input_img = x_train[image, :,:,:]
        img = input_img

         # FEATURE 1 - Pixel values
        #Add pixel values to the data frame
        pixel_values = img.reshape(-1) #flattening
        df['Pixel_Value'] = pixel_values   
        #Pixel value itself as a feature
        #df['Image_Name'] = image   #Capture image name as we read multiple images
        # FEATURE 2 - Bunch of Gabor filter responses
        
                #Generate Gabor features
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                # print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
    #     FEATURE 3 Sobel
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
       
        #Add more filters as needed
        
        #Append features from current image to the dataset
        image_dataset = pd.concat([image_dataset, df])
        
    return image_dataset


def choose_random_image(folder_path):
    class_choice = random.choice(labels)
    folder_path = os.path.join(folder_path, class_choice)
    st.write(class_choice)
    i = random.randint(0, len(os.listdir(folder_path)))
    return os.path.join(folder_path, os.listdir(folder_path)[i])

os.chdir('../')
os.chdir('tumor.v3i.folder/test/')
random_image_path = choose_random_image(os.getcwd())
# print(random_image_path)
img = cv2.imread(random_image_path)
st.image(img)