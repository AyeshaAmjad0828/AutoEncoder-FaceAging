import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import cv2
import os

folder_path = 'data/utkcropped'  
image_list = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        image_list.append(image)

#utoencoder model
input_img = Input(shape=(200, 200, channels))  

# Encoder layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Decoder layers
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output image

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Normalizing the data between 0 and 1
X_train = X_train.astype('float32') / 255.
Y_train = Y_train.astype('float32') / 255.

# Training
autoencoder.fit(X_train, Y_train, epochs=50, batch_size=32, shuffle=True)

# Assuming you have a young face image (X_test)
X_test = X_test.astype('float32') / 255.

# Predict the older version of the face
predicted_old_face = autoencoder.predict(X_test)
