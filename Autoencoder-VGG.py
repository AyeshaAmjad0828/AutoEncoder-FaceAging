import os
import numpy as np
import cv2
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import load_model
import matplotlib.pyplot as plt


# Define constants
input_shape = (200, 200, 3)
batch_size = 32
epochs = 50

# Data directories
young_data_dir = 'data/utk/young'
old_data_dir = 'data/utk/old'

# Function to load and preprocess images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, input_shape[:2])  # Resize to target size
            img = img / 255.0  # Normalize pixel values to the range [0, 1]
            images.append(img)
    return np.array(images)

# Load and preprocess data
young_images = load_images_from_directory(young_data_dir)
old_images = load_images_from_directory(old_data_dir)

# Build the VGG16-based autoencoder model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

inputs = Input(shape=input_shape, name='input_image')
encoded = base_model(inputs)
flat = Flatten()(encoded)
encoded_layer = Dense(256, activation='relu', name='encoded')(flat)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded_layer)
decoded = Reshape(input_shape)(decoded)

autoencoder = Model(inputs, decoded, name='autoencoder')

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(
    young_images,  # Use young images for training (you can also use a generator here if needed)
    old_images,  
    batch_size=batch_size,
    epochs=epochs
)

# Save the trained model
autoencoder.save('face_aging_autoencoder.h5')

# Load the trained autoencoder model
loaded_autoencoder = load_model('face_aging_autoencoder.h5')

# Load a test image
test_image_path = 'data/test/22_0_0_20170117134610684.jpg.chip.jpg'
test_image = cv2.imread(test_image_path)
test_image = cv2.resize(test_image, input_shape[:2])  # Resize to target size
test_image = test_image / 255.0  # Normalize pixel values to the range [0, 1]
test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension

# Apply the autoencoder to the test image
aged_image = loaded_autoencoder.predict(test_image)

# Rescale pixel values back to [0, 255]
aged_image = (aged_image * 255).astype(np.uint8)

# Display the original and aged images side by side in a Jupyter Notebook
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Aged image
aged_image_rgb = cv2.cvtColor(aged_image[0], cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 2)
plt.imshow(aged_image_rgb)
plt.title('Aged Image')
plt.axis('off')

plt.show()