# import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam
from keras.optimizers import adam_v2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import os


width=256
height=256
depth=3
default_image_size = tuple((height, width))

pred_list=['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

inputShape=(width,height,depth)

def resize_image(image_dir):
    image = cv2.imread(image_dir)
    if image is not None :
        image = cv2.resize(image, default_image_size)   
        cv2.imwrite(image_dir,image)
    return

def load_mod():
    model = Sequential()

    model.add(Conv2D(8, (3, 3),input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(units=15,activation='softmax'))

    model.load_weights(os.getcwd() +'\\imagetrain801batch.h5')
    return model

def solution(name):
    dis = {

            'Pepper__bell___Bacterial_spot':['Bleach treatment' , 'Hot water treatment', 'Transplant treatment with streptomycin', 'Copper sprays', 'Plant activator sprays'],
            'Pepper__bell___healthy':['healthy'],
            'Potato___Early_blight':['Drip irrigation method','copper conatining fungicides sprays'],
            'Potato___healthy':['healthy'],
            'Potato___Late_blight':['Applying copper based fungicide','Foliar spray treatment'],
            'Tomato_Bacterial_spot':['Copper-containing bactericides','Foliar insecticidal treatment','Copper sprays'],
            'Tomato_Early_blight':['Bonide Copper Fungicide sprays','Biofungicide Serenade sprays'],
            'Tomato_healthy':['healthy'],
            'Tomato_Late_blight':['Fungicide sprays','Sanitation method','Drip irrigation method'],
            'Tomato_Leaf_Mold':['Fungicide treatment','Planting resistant cultivars','Copper based fungicide sprays'],
            'Tomato_Septoria_leaf_spot':['Fungicide treatment using copper or potassium bicarbonate','Transplant treatment with chlorothalonil'],
            'Tomato_Spider_mites_Two_spotted_spider_mite':['Applying pesticides like  miticide','Organic sprays'],
            'Tomato__Target_Spot':['Transplant treatment with copper oxychloride or mancozeb','Plant activator sprays'],
            'Tomato__Tomato_mosaic_virus':['Covering plants with a  floating row cover or aluminum foil mulches','Sanitation method'],
            'Tomato_Tomato_YellowLeaf_Curl_Virus':['Transpalnt treatment with bifenthrin or dinotefuran','Insecticidal soap sprays']  
        }
    return dis[name]

def process():
    model=load_mod()
    #start
    resize_image(os.getcwd() +'\\media\\result.jpg')
    #end
    pil_im = Image.open(os.getcwd() +'\\media\\result.jpg', 'r')
    X_test=np.asarray(pil_im,dtype="float" )
    X_test=X_test/255.0
    X_test = X_test.reshape(-1,256, 256,3)

    prediction = list(model.predict(X_test))
    #return pred_list[np.argmax(prediction[0])]
    soln = ''
    soln+=pred_list[np.argmax(prediction[0])]
    soln+='  solution:  '
    for x in solution(pred_list[np.argmax(prediction[0])]):
        soln+=x+'\n'
    return soln