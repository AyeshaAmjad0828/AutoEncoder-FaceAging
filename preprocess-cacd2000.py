
import h5py
import os
import shutil

# Load the .mat file using h5py
mat_data = h5py.File('data/celebrity2000.mat', 'r')

# Extract relevant data structures
celebrity_data = mat_data['celebrityData']
celebrity_image_data = mat_data['celebrityImageData']

# Create a directory to store young and old face pairs
output_directory = 'data/cacd'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Dictionary to store images by celebrity ID
celebrity_images = {}

# Populate the dictionary with images by celebrity ID
for i in range(len(celebrity_image_data['identity'])):
    identity = int(celebrity_image_data['identity'][i][0])
    img_name = celebrity_image_data['name'][i][0]
    year = int(celebrity_image_data['year'][i][0])
    age = int(celebrity_image_data['age'][i][0])

    if identity not in celebrity_images:
        celebrity_images[identity] = []

    celebrity_images[identity].append({'name': img_name, 'year': year, 'age': age})

# Find and copy young and old face pairs of the same celebrities
for i in range(len(celebrity_data['identity'])):
    identity = int(celebrity_data['identity'][i][0])
    birth_year = int(celebrity_data['birth'][i][0])

    if identity in celebrity_images and len(celebrity_images[identity]) >= 2:
        # Sort images by year
        celebrity_images[identity] = sorted(celebrity_images[identity], key=lambda x: x['year'])

        # Find young and old face pairs
        young_face = None
        old_face = None

        for img_data in celebrity_images[identity]:
            if img_data['age'] < (img_data['year'] - birth_year) / 2:
                young_face = img_data
            else:
                old_face = img_data
                break

        # Copy young and old face images to the output directory
        if young_face and old_face:
            young_img_path = os.path.join('images', young_face['name'])
            old_img_path = os.path.join('images', old_face['name'])

            shutil.copy(young_img_path, os.path.join(output_directory, f'young_{identity}_{young_face["name"]}'))
            shutil.copy(old_img_path, os.path.join(output_directory, f'old_{identity}_{old_face["name"]}'))