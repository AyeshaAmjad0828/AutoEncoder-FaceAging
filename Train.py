import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import cv2
import os
from PIL import Image

Xfolder_path = 'data/cacd/old'  
Yfolder_path = 'data/cacd/young'

image_list = []
image_dimensions = []  

for filename in os.listdir(Xfolder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(Xfolder_path, filename)
        image = Image.open(image_path)
        image_list.append(image)

        # Get image dimensions (width x height)
        width, height = image.size
        image_dimensions.append((width, height))

# Display the image dimensions
print("Image Dimensions:")
for idx, dimensions in enumerate(image_dimensions, start=1):
    print(f"Image {idx}: Width={dimensions[0]}, Height={dimensions[1]}")


num_images = len(os.listdir(Xfolder_path))

# Collect normalized images in a list
Xnormalized_images = []
Ynormalized_images = []

for filename in os.listdir(Xfolder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        Ximage_path = os.path.join(Xfolder_path, filename)
        Ximage = Image.open(Ximage_path)
        
        # Normalize the image
        Ximage_array = np.array(Ximage)
        Xnormalized_image = Ximage_array / 255.0  # Normalizing to range [0, 1]
        Xnormalized_images.append(Xnormalized_image)

for filename in os.listdir(Yfolder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        Yimage_path = os.path.join(Yfolder_path, filename)
        Yimage = Image.open(Yimage_path)
        
        # Normalize the image
        Yimage_array = np.array(Yimage)
        Ynormalized_image = Yimage_array / 255.0  # Normalizing to range [0, 1]
        Ynormalized_images.append(Ynormalized_image)

# Convert the list of normalized images to a NumPy array

X_train = np.array(Xnormalized_images)
Y_train = np.array(Ynormalized_images)


#autoencoder model
input_img = Input(shape=(200, 200, 3))  

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


# Training
autoencoder.fit(X_train, Y_train, epochs=50, batch_size=32, shuffle=True)



test_image_path = 'data/test/elon.jpg'  
test_image = Image.open(test_image_path)

# Display the image (optional)
test_image.show()

# Convert the image to a numpy array
test_image = np.array(test_image)

# Assuming you have a young face image (X_test)
test_image = test_image.astype('float32') / 255.

# Predict the older version of the face
predicted_old_face = autoencoder.predict(test_image)
