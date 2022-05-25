from tkinter import image_names
import pandas as pd
import numpy as np 
import tensorflow as tf
#=======================
images_data = pd.read_csv("data.csv")
info_data = pd.read_csv("data2.csv")
images_data.head()
images_data = images_data.sort_values(by='num')
images_data.head()
info_data.head()
len(images_data)
len(info_data)
#=======================
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import  array_to_img, img_to_array, load_img
import os
import cv2
datagen = ImageDataGenerator(rotation_range =15, 
                     width_shift_range = 0.2, 
                     height_shift_range = 0.2,  
                     rescale=1./255, 
                     shear_range=0.2, 
                     zoom_range=0.2, 
                     horizontal_flip = True, 
                     fill_mode = 'nearest', 
                     data_format='channels_last', 
                     brightness_range=[0.5, 1.5]) 

imgs = os.listdir(r'C:\Users\un_cs\Documents\capStoneProject\KSA\data\images')

for img in imgs:
    img=cv2.imread(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images"+"\\"+img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'C:\\Users\\user1\\Pictures\\people_1\\preview', save_prefix ='people2', save_format='jpg'):
        i+=1
        if i>10:
            break
#=====================================
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os

datagen = ImageDataGenerator(        
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))
image_directory = r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\"
SIZE = 224
dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):           
        image = io.imread(image_directory + image_name)        
        image = Image.fromarray(image, 'RGB')        
        image = image.resize((SIZE,SIZE)) 
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0
for batch in datagen.flow(x, batch_size=16,
                          save_to_dir= r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images",
                          save_prefix='dr',
                          save_format='jpg'):    
    i += 1    
    if i > 50:        
        break
#==================================================
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(rotation_range =40, 
                     width_shift_range = 0.2, 
                     height_shift_range = 0.2,  
                     rescale=1./255, 
                     shear_range=0.2, 
                     zoom_range=0.2, 
                     horizontal_flip = True, 
                     fill_mode = 'nearest', 
                     data_format='channels_last', 
                     brightness_range=[0.5, 1.5]) 
filenames= os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\")
for f in filenames:
    im = os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\"+f)
    for i in im:
        #print(f+i)
        img = load_img(i)  
        x = img_to_array(img) 
    #  # Reshape the input image 
        x = x.reshape((1, ) + x.shape)  
        j = 0

        #  # generate 50 new augmented images 
        for batch in datagen.flow(x, batch_size = 1, 
                        save_to_dir =r'C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\'+f,  
                        save_prefix='dr',
                        save_format ='jpeg'):
            j += 1
            if j > 50: 
                        break
        # #print(f)
# ============================
datagen = ImageDataGenerator(rotation_range =40, 
                     width_shift_range = 0.2, 
                     height_shift_range = 0.2,  
                     rescale=1./255, 
                     shear_range=0.2, 
                     zoom_range=0.2, 
                     horizontal_flip = True, 
                     fill_mode = 'nearest', 
                     data_format='channels_last', 
                     brightness_range=[0.5, 1.5]) 
# THIS WILL TAKE +30 Minutes 
filenames= os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images")
for f in filenames:
    im = os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\"+f)
    #print(im)
    for i in im :
        img = load_img(i)  
        x = img_to_array(img) 
    #  # Reshape the input image 
        x = x.reshape((1, ) + x.shape) 
        j = 0 
        for batch in datagen.flow(x, batch_size = 1, 
                        save_to_dir =r'C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\'+f,  
                        save_prefix='dr',
                        save_format ='jpeg'):
            j += 1
            if j > 100: 
                        break
for f in filenames:
    im = os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\"+f)
print(len(im)) # 398
#===========
filenames= os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images")
lon = 0
for f in filenames:
    im = os.listdir(r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images\\"+f)
    for i in im :
        lon+=1
print(lon) #119906
#===================================================== Training ===========


from keras.layers.pooling import GlobalAveragePooling1D
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, Convolution2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
images_data.head()
# Encode labels in column 'species'.
images_data['label']= label_encoder.fit_transform(images_data['name'])
images_data.head()
l = list(images_data['label'])
type(l)

#========================================
data_path = r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images"
data_gen = ImageDataGenerator(rescale=1./255,  validation_split=0.3)

train_data = data_gen.flow_from_directory(directory=data_path,target_size=(256, 256), batch_size=32, subset='training', seed=42, class_mode='categorical'  ) 
test_data = data_gen.flow_from_directory(directory=data_path,target_size=(256, 256), batch_size=32, subset='validation', seed=42, class_mode='categorical' ) 
from collections import Counter
counter = Counter(train_data.classes)
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


labels, values = zip(*counter.items())

indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()
type(train_data)
dataset = tf.keras.utils.image_dataset_from_directory(
    r"C:\Users\un_cs\Documents\capStoneProject\KSA\data\images",
    labels='inferred',
    label_mode='categorical',
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
type(dataset)
import matplotlib.pyplot as plt
class_names = dataset.class_names
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
print(class_names)
print(list(dataset.as_numpy_iterator()))
image_size = (180, 180)
batch_size = 128
from tensorflow.keras import layers
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=305)
epochs = 5

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=tf.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_data, epochs=epochs, callbacks=callbacks, validation_data= test_data,
)
